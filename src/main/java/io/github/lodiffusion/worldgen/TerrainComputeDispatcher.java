package io.github.lodiffusion.worldgen;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lwjgl.opengl.GL15C;
import org.lwjgl.opengl.GL31C;
import org.lwjgl.opengl.GL43C;
import com.rhythmatician.lodiffusion.gpu.BiomePaletteSSBO;
import com.rhythmatician.lodiffusion.gpu.BiomePaletteSerializer;
import com.rhythmatician.lodiffusion.gpu.TerrainShaperMlpSsbo;
import net.lodiffusion.shadow.VoxyRequestDecoder;
import net.lodiffusion.shadow.ShadowRouterJobQueue;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Drives per-chunk GPU terrain density field computation.
 *
 * One call to {@link #dispatch(int, int)} produces a full 16×384×16 density grid
 * in Binding 7 (DensityOutput) for the requested chunk column. Callers must read
 * Binding 7 before issuing the next dispatch, or manage their own double-buffering.
 *
 * <h3>RouterConfig UBO (binding 8, std140, 80 bytes)</h3>
 * Mirrors the {@code RouterConfig} uniform block in {@code terrain_compute.comp}.
 * The static portion (noise indices, gradient params, spline offsets) is uploaded once
 * via {@link #init(ShaderProgramManager, RouterConfig)}. Only the chunk origin (8 bytes)
 * is updated per dispatch via {@code glBufferSubData}, avoiding a full UBO re-upload.
 *
 * <h3>Lifecycle</h3>
 * <pre>
 *   TerrainComputeDispatcher d = new TerrainComputeDispatcher();
 *   d.init(shaderProgramManager, RouterConfig.overworldDefaults(data));
 *   // per chunk:
 *   d.dispatch(chunkX, chunkZ);
 *   // optional debug/parity only:
 *   FloatBuffer result = ssboManager.readDensityDebug();
 *   // on world unload:
 *   d.cleanup();
 * </pre>
 */
public class TerrainComputeDispatcher {
    private static final Logger LOGGER = LogManager.getLogger();

    /** Binding point for the RouterConfig UBO (must match terrain_compute.comp). */
    private static final int ROUTER_CONFIG_BINDING = 8;

    /**
     * std140-compatible layout:
     *   offset  0 : int  chunk_origin_x
     *   offset  4 : int  chunk_origin_z
     *   offset  8 : int  _pad0
     *   offset 12 : int  _pad1
     *   offset 16 : int  nn_continents
     *   offset 20 : int  nn_erosion
     *   offset 24 : int  nn_ridges
     *   offset 28 : int  nn_depth_noise
     *   offset 32 : int  nn_jagged
     *   offset 36 : int  nn_shift_a
     *   offset 40 : int  nn_shift_b
     *   offset 44 : int  _pad2
     *   offset 48 : float grad_from_y
     *   offset 52 : float grad_to_y
     *   offset 56 : float grad_from_value
     *   offset 60 : float grad_to_value
     *   offset 64 : int  spline_offset_offset
     *   offset 68 : int  spline_factor_offset
     *   offset 72 : int  spline_jagged_offset
     *   offset 76 : int  _pad3
     *   offset 80 : int  nn_entrances     (WS-4.1a)
     *   offset 84 : int  nn_cheese_caves  (WS-4.1a)
     *   offset 88 : int  nn_spaghetti_2d  (WS-4.1a)
     *   offset 92 : int  nn_roughness     (WS-4.1a)
     *   offset 96 : int  nn_noodle         (WS-4.1a)
     *   offset 100: int  nn_temperature     (biome: temperature noise index)
     *   offset 104: int  nn_vegetation      (biome: vegetation/humidity noise index)
     *   offset 108: int  biome_palette_count (0 = GPU biome pass disabled)
     *   total     : 112 bytes
     */
    private static final int UBO_SIZE_BYTES = 112;

    // Byte offsets for the mutable chunk-origin fields (updated per dispatch)
    private static final int OFFSET_CHUNK_X = 0;
    @SuppressWarnings("unused")
    private static final int OFFSET_CHUNK_Z = 4;

    private ShaderProgramManager shaderManager;
    private int uboId = 0;
    private boolean ready = false;
    private TerrainShaperMlpSsbo mlpSsbo;
    private BiomePaletteSSBO biomePaletteSSBO;

    // -------------------------------------------------------------------------
    // Initialisation
    // -------------------------------------------------------------------------

    /**
     * Allocates the RouterConfig UBO and uploads the static portion.
     * Call once after {@link ShaderProgramManager#compile()} succeeds.
     *
     * @param shaderManager compiled compute program
     * @param config        static router parameters for this dimension
     */
    public void init(ShaderProgramManager shaderManager, RouterConfig config) {
        this.shaderManager = shaderManager;

        // Allocate UBO
        int[] ids = new int[1];
        GL15C.glGenBuffers(ids);
        if (ids[0] == 0) {
            throw new RuntimeException("TerrainComputeDispatcher: failed to allocate RouterConfig UBO");
        }
        uboId = ids[0];

        // Build the full 80-byte buffer (chunk origin starts at (0,0); updated per dispatch)
        ByteBuffer buf = buildUBO(0, 0, config);

        GL15C.glBindBuffer(GL31C.GL_UNIFORM_BUFFER, uboId);
        GL15C.glBufferData(GL31C.GL_UNIFORM_BUFFER, buf, GL15C.GL_DYNAMIC_DRAW);
        GL31C.glBindBufferBase(GL31C.GL_UNIFORM_BUFFER, ROUTER_CONFIG_BINDING, uboId);
        GL15C.glBindBuffer(GL31C.GL_UNIFORM_BUFFER, 0);

        ready = true;
        LOGGER.info("TerrainComputeDispatcher: RouterConfig UBO allocated ({} bytes) at binding {}",
                UBO_SIZE_BYTES, ROUTER_CONFIG_BINDING);

        // Load pre-trained TerrainShaperMLP weights into GPU memory (SSBO=9, UBO=10)
        try {
            mlpSsbo = new TerrainShaperMlpSsbo();
            LOGGER.info("TerrainComputeDispatcher: TerrainShaperMLP weights loaded (binding 9+10)");
        } catch (Exception e) {
            LOGGER.warn("TerrainComputeDispatcher: failed to load MLP weights, shader will use fallback splines — {}", e.getMessage());
            mlpSsbo = null;
        }
    }

    /**
     * Uploads the biome parameter palette to the GPU (binding 12 + 13).
     * Call this once after {@link #init} when the world's biome source is available.
     *
     * @param biomeSource the runtime {@code MultiNoiseBiomeSource} instance
     * @param config      the current RouterConfig to update with the palette entry count
     * @return an updated RouterConfig with {@code biomePaletteCount} set
     */
    public RouterConfig initBiomePalette(Object biomeSource, RouterConfig config) {
        try {
            java.nio.FloatBuffer palette = BiomePaletteSerializer.buildPalette(biomeSource);
            int count = palette.limit() / BiomePaletteSerializer.ENTRY_STRIDE;
            if (count == 0) {
                LOGGER.warn("TerrainComputeDispatcher: biome palette is empty — GPU biome pass disabled");
                return config;
            }
            biomePaletteSSBO = new BiomePaletteSSBO(palette, count);
            LOGGER.info("TerrainComputeDispatcher: biome palette uploaded ({} entries, bindings 12+13)", count);
            return config.withBiomePalette(config.nnTemperature, config.nnVegetation, count);
        } catch (Exception e) {
            LOGGER.error("TerrainComputeDispatcher: biome palette init failed — GPU biome pass disabled", e);
            return config;
        }
    }

    // -------------------------------------------------------------------------
    // Per-chunk dispatch
    // -------------------------------------------------------------------------

    /**
     * Dispatches the terrain compute shader for one 16×16 chunk column.
     *
     * Updates only the 8-byte chunk-origin region of the UBO, then issues
     * {@code glDispatchCompute(1, 1, 1)} followed by a storage barrier.
     *
     * After this returns the density data is ready in Binding 7.
     *
     * @param chunkX chunk X coordinate (not block X — multiply by 16 for block origin)
     * @param chunkZ chunk Z coordinate
     */
    public void dispatch(int chunkX, int chunkZ) {
        if (!ready) {
            LOGGER.warn("TerrainComputeDispatcher.dispatch called before init — skipping");
            return;
        }

        // Partial UBO update: only the 8-byte origin (avoids re-uploading 80 bytes)
        ByteBuffer origin = ByteBuffer.allocateDirect(8).order(ByteOrder.nativeOrder());
        origin.putInt(chunkX * 16);  // block origin X
        origin.putInt(chunkZ * 16);  // block origin Z
        origin.flip();

        GL15C.glBindBuffer(GL31C.GL_UNIFORM_BUFFER, uboId);
        GL15C.glBufferSubData(GL31C.GL_UNIFORM_BUFFER, OFFSET_CHUNK_X, origin);
        GL15C.glBindBuffer(GL31C.GL_UNIFORM_BUFFER, 0);

        // Bind MLP weights (SSBO=9, UBO=10) before shader execution
        if (mlpSsbo != null) {
            mlpSsbo.bind();
        }

        // Bind biome palette + output SSBOs (bindings 12+13)
        if (biomePaletteSSBO != null) {
            biomePaletteSSBO.bind();
        }

        // Execute
        shaderManager.use();
        GL43C.glDispatchCompute(1, 1, 1);
        GL43C.glMemoryBarrier(GL43C.GL_SHADER_STORAGE_BARRIER_BIT);
    }

    /**
     * Pull next job from ShadowRouterJobQueue and dispatch if available.
     * 
     * This implements the demand-driven pull model: Voxy's missing terrain requests
     * are queued by VoxyShadowBridgeMixin, and this method consumes them for GPU generation.
     * 
     * @return true if a request was processed, false if queue is empty
     */
    public boolean acceptNextRequest() {
        if (!ready) {
            return false;
        }
        
        VoxyRequestDecoder.VoxyNodeRequest req = ShadowRouterJobQueue.dequeueAny();
        if (req == null) {
            return false;
        }
        
        // Convert Voxy world coordinates to chunk coordinates.
        // Voxy stores coordinates in a 16-voxel unit space, so divide by 16 to get chunk coords.
        int chunkX = req.worldX / 16;
        int chunkZ = req.worldZ / 16;
        
        // Dispatch for this request
        dispatch(chunkX, chunkZ);
        
        // Log for debugging (can be disabled later)
        LOGGER.debug("TerrainComputeDispatcher: processed request LOD={} at chunk ({}, {})",
                req.lodLevel, chunkX, chunkZ);
        
        return true;
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------

    public void cleanup() {
        if (mlpSsbo != null) {
            mlpSsbo.cleanup();
            mlpSsbo = null;
        }
        if (biomePaletteSSBO != null) {
            biomePaletteSSBO.cleanup();
            biomePaletteSSBO = null;
        }
        if (uboId != 0) {
            int[] ids = { uboId };
            GL15C.glDeleteBuffers(ids);
            uboId = 0;
            ready = false;
            LOGGER.info("TerrainComputeDispatcher: RouterConfig UBO freed");
        }
    }

    public boolean isReady() {
        return ready;
    }

    // -------------------------------------------------------------------------
    // Internal UBO builder
    // -------------------------------------------------------------------------

    private ByteBuffer buildUBO(int chunkX, int chunkZ, RouterConfig c) {
        ByteBuffer buf = ByteBuffer.allocateDirect(UBO_SIZE_BYTES).order(ByteOrder.nativeOrder());

        // offset 0–15: chunk origin + padding
        buf.putInt(chunkX * 16);
        buf.putInt(chunkZ * 16);
        buf.putInt(0); // _pad0
        buf.putInt(0); // _pad1

        // offset 16–47: named NormalNoise indices (-1 = not mapped → GLSL fallback)
        buf.putInt(c.nnContinents);
        buf.putInt(c.nnErosion);
        buf.putInt(c.nnRidges);
        buf.putInt(c.nnDepthNoise);
        buf.putInt(c.nnJagged);
        buf.putInt(c.nnShiftA);
        buf.putInt(c.nnShiftB);
        buf.putInt(0); // _pad2

        // offset 48–63: YClampedGradient parameters
        buf.putFloat(c.gradFromY);
        buf.putFloat(c.gradToY);
        buf.putFloat(c.gradFromValue);
        buf.putFloat(c.gradToValue);

        // offset 64–79: spline table offsets (-1 = no spline data → GLSL fallback)
        buf.putInt(c.splineOffsetOffset);
        buf.putInt(c.splineFactorOffset);
        buf.putInt(c.splineJaggedOffset);
        buf.putInt(0); // _pad3

        // offset 80–111: cave noise indices (WS-4.1a; -1 = not wired → GLSL skips)
        buf.putInt(c.nnEntrances);
        buf.putInt(c.nnCheeseCaves);
        buf.putInt(c.nnSpaghetti2d);
        buf.putInt(c.nnRoughness);
        buf.putInt(c.nnNoodle);
        buf.putInt(c.nnTemperature);       // was _pad4
        buf.putInt(c.nnVegetation);        // was _pad5
        buf.putInt(c.biomePaletteCount);   // was _pad6

        buf.flip();
        return buf;
    }

    // =========================================================================
    // RouterConfig — mirrors the RouterConfig uniform block in terrain_compute.comp
    // =========================================================================

    /**
     * Carries the static (per-dimension) portion of the RouterConfig UBO.
     *
     * Named noise indices correspond to positions within the flat NormalNoise arrays
     * (normalNoiseInts / normalNoiseFloats) uploaded to Bindings 4 & 5.  A value of
     * {@code -1} signals the GLSL shader to use its simplified fallback path.
     *
     * Use {@link #overworldDefaults()} until {@code NoiseRouterExtractor} exposes
     * per-output indices; then call {@link #withNamedIndices} to wire them in.
     */
    public static class RouterConfig {

        // Named NormalNoise indices (index into binding 4/5 flat arrays)
        public int nnContinents  = -1;
        public int nnErosion     = -1;
        public int nnRidges      = -1;
        public int nnDepthNoise  = -1;
        public int nnJagged      = -1;
        public int nnShiftA      = -1;
        public int nnShiftB      = -1;

        // WS-4.1a: Cave noise indices (-1 = disabled)
        public int nnEntrances   = -1;   // overworld/caves/entrances
        public int nnCheeseCaves = -1;   // overworld/caves/pillars
        public int nnSpaghetti2d = -1;   // overworld/caves/spaghetti_2d
        public int nnRoughness   = -1;   // overworld/caves/spaghetti_roughness_function
        public int nnNoodle      = -1;   // overworld/caves/noodle

        // Biome classification noise indices + palette count
        public int nnTemperature     = -1;  // NoiseRouter.temperature
        public int nnVegetation      = -1;  // NoiseRouter.vegetation (= humidity)
        public int biomePaletteCount =  0;  // 0 = GPU biome pass disabled

        // YClampedGradient for depth
        public float gradFromY     = -64.0f;
        public float gradToY       = 320.0f;
        public float gradFromValue =  1.5f;
        public float gradToValue   = -1.5f;

        // Spline data offsets into Binding 6 (-1 = no spline, use linear fallback)
        public int splineOffsetOffset = -1;
        public int splineFactorOffset = -1;
        public int splineJaggedOffset = -1;

        /**
         * Overworld defaults: Vanilla gradient range, no spline data yet, all
         * NormalNoise indices unmapped.  Triggers the GLSL simplified fallback paths.
         */
        public static RouterConfig overworldDefaults() {
            return new RouterConfig(); // all field defaults match Overworld
        }

        /**
         * Returns a copy with explicit named NormalNoise indices set.
         * Call this once {@code NoiseRouterExtractor} tracks per-output names.
         */
        public RouterConfig withNamedIndices(
                int continents, int erosion, int ridges,
                int depthNoise, int jagged, int shiftA, int shiftB) {
            RouterConfig c = new RouterConfig();
            c.nnContinents = continents;
            c.nnErosion    = erosion;
            c.nnRidges     = ridges;
            c.nnDepthNoise = depthNoise;
            c.nnJagged     = jagged;
            c.nnShiftA     = shiftA;
            c.nnShiftB     = shiftB;
            // copy gradient, spline and cave fields
            c.gradFromY      = this.gradFromY;
            c.gradToY        = this.gradToY;
            c.gradFromValue  = this.gradFromValue;
            c.gradToValue    = this.gradToValue;
            c.splineOffsetOffset = this.splineOffsetOffset;
            c.splineFactorOffset = this.splineFactorOffset;
            c.splineJaggedOffset = this.splineJaggedOffset;
            c.nnEntrances        = this.nnEntrances;
            c.nnCheeseCaves      = this.nnCheeseCaves;
            c.nnSpaghetti2d      = this.nnSpaghetti2d;
            c.nnRoughness        = this.nnRoughness;
            c.nnNoodle           = this.nnNoodle;
            c.nnTemperature      = this.nnTemperature;
            c.nnVegetation       = this.nnVegetation;
            c.biomePaletteCount  = this.biomePaletteCount;
            return c;
        }

        /**
         * Returns a copy with biome classification fields set.
         * Call {@code TerrainComputeDispatcher.initBiomePalette()} which calls this internally.
         */
        public RouterConfig withBiomePalette(int temperature, int vegetation, int paletteCount) {
            RouterConfig c = new RouterConfig();
            c.nnContinents       = this.nnContinents;
            c.nnErosion          = this.nnErosion;
            c.nnRidges           = this.nnRidges;
            c.nnDepthNoise       = this.nnDepthNoise;
            c.nnJagged           = this.nnJagged;
            c.nnShiftA           = this.nnShiftA;
            c.nnShiftB           = this.nnShiftB;
            c.gradFromY          = this.gradFromY;
            c.gradToY            = this.gradToY;
            c.gradFromValue      = this.gradFromValue;
            c.gradToValue        = this.gradToValue;
            c.splineOffsetOffset = this.splineOffsetOffset;
            c.splineFactorOffset = this.splineFactorOffset;
            c.splineJaggedOffset = this.splineJaggedOffset;
            c.nnEntrances        = this.nnEntrances;
            c.nnCheeseCaves      = this.nnCheeseCaves;
            c.nnSpaghetti2d      = this.nnSpaghetti2d;
            c.nnRoughness        = this.nnRoughness;
            c.nnNoodle           = this.nnNoodle;
            c.nnTemperature      = temperature;
            c.nnVegetation       = vegetation;
            c.biomePaletteCount  = paletteCount;
            return c;
        }

        /**
         * Returns a copy with cave noise indices set (WS-4.1a).
         * Call this once {@code NoiseRouterExtractor} exposes cave noise indices.
         */
        public RouterConfig withCaveIndices(
                int entrances, int cheeseCaves, int spaghetti2d,
                int roughness, int noodle) {
            RouterConfig c = new RouterConfig();
            c.nnContinents = this.nnContinents;
            c.nnErosion    = this.nnErosion;
            c.nnRidges     = this.nnRidges;
            c.nnDepthNoise = this.nnDepthNoise;
            c.nnJagged     = this.nnJagged;
            c.nnShiftA     = this.nnShiftA;
            c.nnShiftB     = this.nnShiftB;
            c.gradFromY      = this.gradFromY;
            c.gradToY        = this.gradToY;
            c.gradFromValue  = this.gradFromValue;
            c.gradToValue    = this.gradToValue;
            c.splineOffsetOffset = this.splineOffsetOffset;
            c.splineFactorOffset = this.splineFactorOffset;
            c.splineJaggedOffset = this.splineJaggedOffset;
            c.nnEntrances   = entrances;
            c.nnCheeseCaves = cheeseCaves;
            c.nnSpaghetti2d = spaghetti2d;
            c.nnRoughness   = roughness;
            c.nnNoodle      = noodle;
            return c;
        }

        /**
         * Returns a copy with explicit spline buffer offsets set.
         * Call this once spline data is packed into Binding 6 by the extractor.
         */
        public RouterConfig withSplineOffsets(int offsetSpline, int factorSpline, int jaggedSpline) {
            RouterConfig c = new RouterConfig();
            c.nnContinents = this.nnContinents;
            c.nnErosion    = this.nnErosion;
            c.nnRidges     = this.nnRidges;
            c.nnDepthNoise = this.nnDepthNoise;
            c.nnJagged     = this.nnJagged;
            c.nnShiftA     = this.nnShiftA;
            c.nnShiftB     = this.nnShiftB;
            c.gradFromY      = this.gradFromY;
            c.gradToY        = this.gradToY;
            c.gradFromValue  = this.gradFromValue;
            c.gradToValue    = this.gradToValue;
            c.splineOffsetOffset = offsetSpline;
            c.splineFactorOffset = factorSpline;
            c.splineJaggedOffset = jaggedSpline;
            c.nnEntrances   = this.nnEntrances;
            c.nnCheeseCaves = this.nnCheeseCaves;
            c.nnSpaghetti2d = this.nnSpaghetti2d;
            c.nnRoughness   = this.nnRoughness;
            c.nnNoodle      = this.nnNoodle;
            c.nnTemperature     = this.nnTemperature;
            c.nnVegetation      = this.nnVegetation;
            c.biomePaletteCount = this.biomePaletteCount;
            return c;
        }
    }
}

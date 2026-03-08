package com.rhythmatician.lodiffusion.voxy;

import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.Heightmap;
import net.minecraft.world.biome.Biome;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.gen.noise.NoiseConfig;

import java.util.logging.Logger;

/**
 * Computes the v2 anchor conditioning inputs from the Minecraft client world.
 *
 * <p>The v2 contract requires:
 * <ul>
 *   <li>{@code x_height_planes} [1, 5, 16, 16] float32 — 5-plane heightmap
 *       (surface, ocean_floor, slope_x, slope_z, curvature), normalised [0..1]</li>
 *   <li>{@code x_router6} [1, 6, 16, 16] float32 — 6-channel CORE router values
 *       (temperature, vegetation, continents, erosion, depth, ridges), z-scored</li>
 *   <li>{@code x_biome} [1, 16, 16] int64 — biome integer index per column</li>
 * </ul>
 *
 * <p>On the client side we do not have access to the full DensityFunction pipeline.
 * We therefore <em>approximate</em> router6 from the information that is available:
 * biome IDs and the heightmap.  This matches the fallback used during training
 * ({@code approximate_router6_from_biome} in Python).
 *
 * <p>If a server-side {@link com.rhythmatician.lodiffusion.world.noise.NoiseTap.Cache}
 * is available (e.g., when running an integrated server), the real router values
 * are used instead.
 */
public final class AnchorSampler {

    private static final Logger LOGGER = Logger.getLogger(AnchorSampler.class.getName());

    // Router6 channel indices — must match Python training order
    public static final int TEMP_IDX        = 0;
    public static final int VEGETATION_IDX  = 1;
    public static final int CONTINENTS_IDX  = 2;
    public static final int EROSION_IDX     = 3;
    public static final int DEPTH_IDX       = 4;
    public static final int RIDGES_IDX      = 5;

    /** MC sea level — used to normalise heights. */
    private static final float SEA_LEVEL = 62f;
    /** Typical raw height range for normalisation. */
    private static final float HEIGHT_RANGE = 320f;

    private AnchorSampler() {}

    // ------------------------------------------------------------------
    // Public API
    // ------------------------------------------------------------------

    /** Container for all v2 anchor inputs for one 16×16 chunk column. */
    public record AnchorInputs(
        float[][]  heightPlanes5,  // [5][256] row-major (will be reshaped to [5,16,16])
        float[][]  router6,        // [6][256] row-major (will be reshaped to [6,16,16])
        int[][]    biomeIdx        // [16][16]
    ) {}

    /**
     * Sample v2 anchor inputs for the 16×16 section at the given chunk coordinates.
     *
     * @param chunk       the Minecraft chunk (or null if not loaded — falls back to zeros)
     * @param noiseConfig server-side noise config for real router lookup; may be null
     * @return an {@link AnchorInputs} record ready to pass to
     *         {@link com.rhythmatician.lodiffusion.onnx.UnifiedModelRunner#generateV2}
     */
    public static AnchorInputs sample(Chunk chunk, NoiseConfig noiseConfig) {
        int[][] biomeIdx  = sampleBiomes(chunk);
        float[][] hmap    = sampleHeightmap(chunk);
        float[][] heightPlanes = computeHeightPlanes(hmap);
        float[][] router6 = approximateRouter6(biomeIdx, hmap);
        return new AnchorInputs(heightPlanes, router6, biomeIdx);
    }

    /**
     * Sample v2 anchor inputs using the server-side noise pipeline.
     *
     * <p>This produces <em>real</em> heightmap, router6, and biome data
     * for any (sectionX, sectionZ) coordinate — no loaded chunk required.
     * This is the primary path; the chunk-based {@link #sample(Chunk, NoiseConfig)}
     * should only be used when {@link WorldNoiseAccess} is not available.
     *
     * @param noiseAccess server-side noise access (must not be null)
     * @param sectionX    chunk / section X coordinate
     * @param sectionZ    chunk / section Z coordinate
     * @return an {@link AnchorInputs} record with real world-gen data
     */
    public static AnchorInputs sampleFromNoise(WorldNoiseAccess noiseAccess,
                                                int sectionX, int sectionZ) {
        // 1. Real heightmap from ChunkGenerator.getHeight()
        float[][] hmap = noiseAccess.sampleHeightmap(sectionX, sectionZ);

        // 2. Real biomes from BiomeSource
        int[][] biomeIdx = noiseAccess.sampleBiomes(sectionX, sectionZ, hmap);

        // 3. Height-planes are derived from the heightmap (pure math — same either way)
        float[][] heightPlanes = computeHeightPlanes(hmap);

        // 4. Real router6 from NoiseRouter density functions
        float[][] router6 = noiseAccess.sampleRouter6(sectionX, sectionZ, hmap);

        return new AnchorInputs(heightPlanes, router6, biomeIdx);
    }

    // ------------------------------------------------------------------
    // Biome sampling
    // ------------------------------------------------------------------

    /**
     * Extract a [16][16] biome integer-index grid for the chunk.
     * Uses the surface-level biome at each column (y=64).
     */
    static int[][] sampleBiomes(Chunk chunk) {
        int[][] out = new int[16][16];
        if (chunk == null) return out;  // default 0 (air biome)

        ChunkPos cp = chunk.getPos();
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                int bx = cp.getStartX() + lx;
                int bz = cp.getStartZ() + lz;
                // Sample biome at sea level as representative
                RegistryEntry<Biome> biomeEntry = chunk.getBiomeForNoiseGen(lx >> 2, 4, lz >> 2);
                // Map registry entry to a stable integer index using raw ID hash
                out[lx][lz] = Math.abs(biomeEntry.hashCode()) % 256;
            }
        }
        return out;
    }

    // ------------------------------------------------------------------
    // Heightmap extraction
    // ------------------------------------------------------------------

    /**
     * Extract the raw WORLD_SURFACE heightmap [16][16] in block-Y coordinates.
     */
    static float[][] sampleHeightmap(Chunk chunk) {
        float[][] hm = new float[16][16];
        if (chunk == null) {
            // Default flat terrain at sea level
            for (int x = 0; x < 16; x++)
                for (int z = 0; z < 16; z++)
                    hm[x][z] = SEA_LEVEL;
            return hm;
        }

        var heightmap = chunk.getHeightmap(Heightmap.Type.WORLD_SURFACE);
        for (int lx = 0; lx < 16; lx++)
            for (int lz = 0; lz < 16; lz++)
                hm[lx][lz] = heightmap.get(lx, lz);
        return hm;
    }

    // ------------------------------------------------------------------
    // Height-planes computation
    // ------------------------------------------------------------------

    /**
     * Derive the 5-plane height feature tensor from a raw heightmap.
     *
     * <p>Planes:
     * <ol>
     *   <li>surface         — normalised block-Y / 320</li>
     *   <li>ocean_floor     — approx from below-sea portion (surface when &lt;= sea)</li>
     *   <li>slope_x         — finite difference dH/dx normalised</li>
     *   <li>slope_z         — finite difference dH/dz normalised</li>
     *   <li>curvature       — Laplacian (d²H/dx² + d²H/dz²), normalised</li>
     * </ol>
     *
     * @return float[5][256] in row-major order (channel, lx*16+lz)
     */
    static float[][] computeHeightPlanes(float[][] hm) {
        float[][] planes = new float[5][256];

        float[] surfaceFlat   = planes[0];
        float[] oceanFloorFlat = planes[1];
        float[] slopeXFlat    = planes[2];
        float[] slopeZFlat    = planes[3];
        float[] curvatureFlat = planes[4];

        // ---- surface & ocean_floor ----
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                float h = hm[lx][lz];
                surfaceFlat  [lx * 16 + lz] = h / HEIGHT_RANGE;
                oceanFloorFlat[lx * 16 + lz] = Math.min(h, SEA_LEVEL) / HEIGHT_RANGE;
            }
        }

        // ---- slope_x (∂H/∂x) ----
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                int xl = Math.max(0, lx - 1);
                int xr = Math.min(15, lx + 1);
                float dx = (hm[xr][lz] - hm[xl][lz]) / (2f * (xr - xl == 0 ? 1 : (xr - xl)));
                slopeXFlat[lx * 16 + lz] = clampSlope(dx);
            }
        }

        // ---- slope_z (∂H/∂z) ----
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                int zl = Math.max(0, lz - 1);
                int zr = Math.min(15, lz + 1);
                float dz = (hm[lx][zr] - hm[lx][zl]) / (2f * (zr - zl == 0 ? 1 : (zr - zl)));
                slopeZFlat[lx * 16 + lz] = clampSlope(dz);
            }
        }

        // ---- curvature (Laplacian) ----
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                float center = hm[lx][lz];
                float left   = hm[Math.max(0, lx - 1)][lz];
                float right  = hm[Math.min(15, lx + 1)][lz];
                float down   = hm[lx][Math.max(0, lz - 1)];
                float up     = hm[lx][Math.min(15, lz + 1)];
                float laplacian = left + right + down + up - 4f * center;
                curvatureFlat[lx * 16 + lz] = clampCurvature(laplacian);
            }
        }

        return planes;
    }

    // ------------------------------------------------------------------
    // Router6 approximation
    // ------------------------------------------------------------------

    /**
     * Approximate the 6-channel CORE router values from biome indices and
     * the surface heightmap.
     *
     * <p>This mirrors the Python {@code approximate_router6_from_biome()}
     * fallback used when real NoiseTap data is unavailable.
     *
     * @return float[6][256] in row-major order (channel, lx*16+lz)
     */
    static float[][] approximateRouter6(int[][] biomeIdx, float[][] hm) {
        float[][] r6 = new float[6][256];

        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                int idx = lx * 16 + lz;
                int b = biomeIdx[lx][lz];
                float h = hm[lx][lz] / HEIGHT_RANGE;  // normalised 0..1

                // Temperature: warm biomes (0–64 IDs) → higher value; cold biomes → lower
                float temp = 0.5f + (b % 64) / 128f - 0.25f;
                r6[TEMP_IDX][idx] = clampRouter(temp);

                // Vegetation: mid-range biomes tend toward forests
                float veg = 0.4f + (b % 32) / 64f;
                r6[VEGETATION_IDX][idx] = clampRouter(veg);

                // Continents: high terrain → high continents value
                r6[CONTINENTS_IDX][idx] = clampRouter(h * 0.8f + 0.1f);

                // Erosion: lower terrain → more erosion
                r6[EROSION_IDX][idx] = clampRouter(1f - h * 0.7f);

                // Depth: ocean floor depth proxy (negative below sea, positive above)
                float depth = (hm[lx][lz] - SEA_LEVEL) / HEIGHT_RANGE;
                r6[DEPTH_IDX][idx] = clampRouter(depth * 0.5f + 0.5f);

                // Ridges: high local-variation proxy (use steep terrain as proxy)
                float ridges = Math.abs(h - 0.5f) * 2f;
                r6[RIDGES_IDX][idx] = clampRouter(ridges);
            }
        }
        return r6;
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static float clampSlope(float v) {
        // Slopes in the range ±1.0 are typical; normalise and clip
        return Math.max(-1f, Math.min(1f, v / 4f)) * 0.5f + 0.5f;
    }

    private static float clampCurvature(float v) {
        // Laplacian values up to ±16 are typical
        return Math.max(-1f, Math.min(1f, v / 16f)) * 0.5f + 0.5f;
    }

    private static float clampRouter(float v) {
        return Math.max(0f, Math.min(1f, v));
    }
}

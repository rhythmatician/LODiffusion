package io.github.lodiffusion.worldgen;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.lwjgl.opengl.GL43C;

import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * Manages OpenGL Shader Storage Buffer Objects (SSBOs) for GPU terrain generation.
 *
 * Allocates and uploads 8 GPU buffers (one per binding 0-7) containing noise parameters,
 * permutation tables, and spline data extracted from the NoiseRouter.
 *
 * Memory Layout (std430 alignment rules):
 * - Binding 0: ImprovedNoise origins (vec3 packed as vec4 for alignment)
 * - Binding 1: ImprovedNoise permutation tables (uint arrays)
 * - Binding 2: PerlinNoise octave indices (int array)
 * - Binding 3: PerlinNoise amplitudes (float array)
 * - Binding 4: NormalNoise perlin indices (int array)
 * - Binding 5: NormalNoise value factors (float array)
 * - Binding 6: Spline control point flattening (float array)
 * - Binding 7: Output density grid (read-write, target for compute dispatch)
 */
public class ShaderSSBOManager {
    private static final Logger LOGGER = LogManager.getLogger();

    // SSBO Binding points (must match GLSL shader definitions)
    private static final int IMPROVED_ORIGINS_BINDING = 0;
    private static final int IMPROVED_PERMS_BINDING = 1;
    private static final int PERLIN_INT_BINDING = 2;
    private static final int PERLIN_FLOAT_BINDING = 3;
    private static final int NORMAL_NOISE_INT_BINDING = 4;
    private static final int NORMAL_NOISE_FLOAT_BINDING = 5;
    private static final int SPLINE_DATA_BINDING = 6;
    private static final int DENSITY_OUTPUT_BINDING = 7;

    private static final int BUFFER_COUNT = 8;

    // Shader program manager for compute operations
    private ShaderProgramManager shaderManager = new ShaderProgramManager();

    // Per-chunk compute dispatcher (owns the RouterConfig UBO at binding 8)
    private TerrainComputeDispatcher dispatcher = new TerrainComputeDispatcher();

    // OpenGL buffer IDs (one per binding)
    private int[] bufferIds = new int[BUFFER_COUNT];
    private boolean initialized = false;
    private long lastUploadTime = 0L;

    public ShaderSSBOManager() {
        // Buffers will be created on-demand during first upload
    }

    /**
     * Uploads NoiseRouterData to GPU SSBOs and prepares the compute pipeline.
     *
     * Call this on the render thread after extracting the NoiseRouter.
     * Use GL_STATIC_DRAW for typical dimension/gameplay scenarios.
     *
     * ALIGNMENT NOTE (std430):
     * - vec3 is treated as vec4 for alignment purposes (12 bytes + 4 bytes padding)
     * - This method handles padding automatically for improvedOrigins
     */
    public void uploadNoiseData(NoiseRouterExtractor.NoiseRouterData data) {
        if (data == null) {
            LOGGER.warn("uploadNoiseData called with null NoiseRouterData");
            return;
        }

        LOGGER.info("ShaderSSBOManager: Uploading NoiseRouter data to GPU...");

        // Ensure buffers and shaders are allocated
        if (!initialized) {
            allocateBuffers();
        }

        try {
            // Upload each buffer with appropriate GL settings
            uploadBuffer(IMPROVED_ORIGINS_BINDING, padImprovedOrigins(data.improvedOrigins),
                    "ImprovedNoise Origins (vec3→vec4 padded)");
            uploadBuffer(IMPROVED_PERMS_BINDING, data.improvedPerms,
                    "ImprovedNoise Permutations");
            uploadBuffer(PERLIN_INT_BINDING, data.perlinInts,
                    "PerlinNoise Octave Indices");
            uploadBuffer(PERLIN_FLOAT_BINDING, data.perlinFloats,
                    "PerlinNoise Amplitudes");
            uploadBuffer(NORMAL_NOISE_INT_BINDING, data.normalNoiseInts,
                    "NormalNoise Perlin Indices");
            uploadBuffer(NORMAL_NOISE_FLOAT_BINDING, data.normalNoiseFloats,
                    "NormalNoise Value Factors");
            uploadBuffer(SPLINE_DATA_BINDING, data.splineData,
                    "Spline Control Points");

            // Binding 7 (Density Output) is allocated but left uninitialized (written by compute shader)
            allocateDensityOutput();

            // Compile shaders after SSBOs are ready
            shaderManager.compile();

            // Initialise the per-chunk dispatcher (uploads RouterConfig UBO)
            dispatcher.init(shaderManager, TerrainComputeDispatcher.RouterConfig.overworldDefaults());

            lastUploadTime = System.currentTimeMillis();
            LOGGER.info("ShaderSSBOManager: GPU upload, shader compilation, and dispatcher init complete");
        } catch (Exception e) {
            LOGGER.error("ShaderSSBOManager: Failed to initialize GPU pipeline", e);
            throw new RuntimeException("GPU pipeline initialization failed", e);
        }
    }

    /**
     * Dispatches the compute shader for a single 16×16 chunk column.
     *
     * Sets the chunk origin in the RouterConfig UBO, dispatches one workgroup
     * (matching the shader's local_size 16×1×16 layout), then issues a
     * storage barrier so Binding 7 is readable immediately after this returns.
     *
     * @param chunkX chunk coordinate X (block origin = chunkX * 16)
     * @param chunkZ chunk coordinate Z (block origin = chunkZ * 16)
     */
    public void dispatch(int chunkX, int chunkZ) {
        if (!initialized || !shaderManager.isCompiled() || !dispatcher.isReady()) return;
        dispatcher.dispatch(chunkX, chunkZ);
    }

    /**
     * Pads improvedOrigins FloatBuffer for std430 alignment.
     *
     * Input: 3 floats per instance (x, y, z)
     * Output: 4 floats per instance (x, y, z, padding)
     *
     * This prevents the GPU from misaligning the next instance's X coordinate
     * with the current instance's padding slot.
     */
    private FloatBuffer padImprovedOrigins(FloatBuffer origins) {
        if (origins == null || origins.capacity() == 0) {
            LOGGER.warn("improvedOrigins buffer is empty or null");
            return FloatBuffer.allocate(0);
        }

        int elementCount = origins.capacity() / 3;
        FloatBuffer padded = FloatBuffer.allocate(elementCount * 4);

        origins.rewind();
        for (int i = 0; i < elementCount; i++) {
            padded.put(origins.get()); // x
            padded.put(origins.get()); // y
            padded.put(origins.get()); // z
            padded.put(0.0f);          // padding (unused, but required for alignment)
        }
        padded.rewind();
        return padded;
    }

    /**
     * Allocates GPU buffers on first upload.
     * Individual buffer IDs are created lazily in reinterpret_cast_obtainBufferId().
     */
    private synchronized void allocateBuffers() {
        if (initialized) {
            return;
        }

        try {
            LOGGER.info("ShaderSSBOManager: Buffer allocation system initialized (lazy allocation on first upload)");
            initialized = true;
        } catch (Exception e) {
            LOGGER.error("ShaderSSBOManager: Failed to initialize buffer allocation", e);
            throw new RuntimeException("Buffer allocation failed", e);
        }
    }

    /**
     * Generic buffer upload handler.
     *
     * Binds buffer, allocates storage, and uploads data with GL_STATIC_DRAW.
     * This method is designed to be safe even if called from non-render threads
     * (via RenderSystem queue if needed).
     */
    private void uploadBuffer(int bindingPoint, FloatBuffer data, String debugName) {
        if (data == null || data.capacity() == 0) {
            LOGGER.debug("Skipping empty buffer: {} (binding {})", debugName, bindingPoint);
            return;
        }

        try {
            int bufferId = reinterpret_cast_obtainBufferId(bindingPoint);
            long sizeBytes = (long) data.capacity() * Float.BYTES;

            // Bind buffer to SSBO target and upload
            glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);
            glBufferData(GL_COPY_WRITE_BUFFER, sizeBytes, data, GL_STATIC_DRAW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPoint, bufferId);

            LOGGER.info("ShaderSSBOManager: Uploaded {} ({} KB) to binding {}", 
                    debugName, sizeBytes / 1024, bindingPoint);
        } catch (Exception e) {
            LOGGER.error("ShaderSSBOManager: Failed to upload {} to binding {}", debugName, bindingPoint, e);
            throw new RuntimeException("SSBO upload failed for " + debugName, e);
        }
    }

    /**
     * Uploads int-based buffer (permutations, octave indices, etc.)
     */
    private void uploadBuffer(int bindingPoint, IntBuffer data, String debugName) {
        if (data == null || data.capacity() == 0) {
            LOGGER.debug("Skipping empty buffer: {} (binding {})", debugName, bindingPoint);
            return;
        }

        try {
            int bufferId = reinterpret_cast_obtainBufferId(bindingPoint);
            long sizeBytes = (long) data.capacity() * Integer.BYTES;

            glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);
            glBufferData(GL_COPY_WRITE_BUFFER, sizeBytes, data, GL_STATIC_DRAW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingPoint, bufferId);

            LOGGER.info("ShaderSSBOManager: Uploaded {} ({} KB) to binding {}", 
                    debugName, sizeBytes / 1024, bindingPoint);
        } catch (Exception e) {
            LOGGER.error("ShaderSSBOManager: Failed to upload {} to binding {}", debugName, bindingPoint, e);
            throw new RuntimeException("SSBO upload failed for " + debugName, e);
        }
    }

    /**
     * Allocates output density buffer (binding 7, RW by compute shader).
     * Pre-allocation prevents issues if the shader writes before we read results.
     */
    private void allocateDensityOutput() {
        try {
            // Allocate a 256KB buffer for density output grid
            // (actual size will depend on chunk column dimension)
            int bufferId = reinterpret_cast_obtainBufferId(DENSITY_OUTPUT_BINDING);
            long sizeBytes = 256 * 1024; // 256 KB default (can be resized later)

            glBindBuffer(GL_COPY_WRITE_BUFFER, bufferId);
            glBufferDataNull(GL_COPY_WRITE_BUFFER, sizeBytes, GL_DYNAMIC_DRAW);
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, DENSITY_OUTPUT_BINDING, bufferId);

            LOGGER.info("ShaderSSBOManager: Allocated density output buffer ({} KB) at binding {}", 
                    sizeBytes / 1024, DENSITY_OUTPUT_BINDING);
        } catch (Exception e) {
            LOGGER.error("ShaderSSBOManager: Failed to allocate density output buffer", e);
            throw new RuntimeException("Density output allocation failed", e);
        }
    }

    /**
     * Reads back data from an SSBO into a FloatBuffer.
     * Useful for validation (e.g., checking density output at binding 7).
     *
     * @param bindingPoint The SSBO binding index (0-7)
     * @param elementCount Number of floats to read
     * @return A FloatBuffer containing the GPU-side data
     */
    public FloatBuffer readBuffer(int bindingPoint, int elementCount) {
        if (bindingPoint < 0 || bindingPoint >= BUFFER_COUNT || bufferIds[bindingPoint] == 0) {
            return null;
        }

        try {
            FloatBuffer result = FloatBuffer.allocate(elementCount);
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferIds[bindingPoint]);
            GL43C.glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, result);
            result.rewind();
            return result;
        } catch (Exception e) {
            LOGGER.error("ShaderSSBOManager: Failed to read buffer from binding {}", bindingPoint, e);
            return null;
        }
    }

    /**
     * Retrieves or creates a buffer ID for a binding point.
     * Uses glGenBuffers() to allocate GPU-side space on first access.
     */
    private int reinterpret_cast_obtainBufferId(int bindingPoint) {
        if (bufferIds[bindingPoint] == 0) {
            bufferIds[bindingPoint] = glGenBuffers();
            LOGGER.debug("ShaderSSBOManager: Generated buffer ID {} for binding {}", 
                    bufferIds[bindingPoint], bindingPoint);
        }
        return bufferIds[bindingPoint];
    }

    /**
     * Cleans up GPU resources.
     * Call this when the world is unloaded or the player switches servers.
     */
    public synchronized void cleanup() {
        if (!initialized) {
            return;
        }

        try {
            // Cleanup dispatcher UBO first, then shader program
            dispatcher.cleanup();
            shaderManager.cleanup();

            for (int i = 0; i < bufferIds.length; i++) {
                if (bufferIds[i] != 0) {
                    glDeleteBuffers(bufferIds[i]);
                    LOGGER.debug("ShaderSSBOManager: Deleted buffer {} at binding {}", bufferIds[i], i);
                }
            }
            bufferIds = new int[BUFFER_COUNT];
            initialized = false;
            LOGGER.info("ShaderSSBOManager: GPU resources cleaned up");
        } catch (Exception e) {
            LOGGER.error("ShaderSSBOManager: Error during cleanup", e);
        }
    }

    /**
     * Checks if buffers are currently allocated and valid.
     */
    public boolean isValid() {
        return initialized && System.currentTimeMillis() - lastUploadTime < 300000; // 5 min timeout
    }

    /**
     * Returns the buffer ID for a specific binding (for debugging/inspection).
     */
    public int getBufferId(int bindingPoint) {
        if (bindingPoint < 0 || bindingPoint >= BUFFER_COUNT) {
            return 0;
        }
        return bufferIds[bindingPoint];
    }

    // ============================================================================
    // GL Operations (LWJGL3 GL43C backend — direct calls for buffer management)
    // ============================================================================
    // Called during world load (main thread context), so no RenderSystem wrapping needed

    private void glBindBuffer(int target, int buffer) {
        GL43C.glBindBuffer(target, buffer);
    }

    private void glBufferData(int target, long size, FloatBuffer data, int usage) {
        if (data != null) {
            data.position(0);
            GL43C.glBufferData(target, data, usage);
        }
    }

    private void glBufferData(int target, long size, IntBuffer data, int usage) {
        if (data != null) {
            data.position(0);
            GL43C.glBufferData(target, data, usage);
        }
    }

    private void glBindBufferBase(int target, int index, int buffer) {
        GL43C.glBindBufferBase(target, index, buffer);
    }

    private void glBufferDataNull(int target, long size, int usage) {
        GL43C.glBufferData(target, size, usage);
    }

    private int glGenBuffers() {
        // Generate one buffer and return its ID
        int[] ids = new int[1];
        GL43C.glGenBuffers(ids);
        if (ids[0] == 0) {
            throw new RuntimeException("Failed to generate OpenGL buffer");
        }
        return ids[0];
    }

    private void glDeleteBuffers(int buffer) {
        if (buffer != 0) {
            int[] ids = { buffer };
            GL43C.glDeleteBuffers(ids);
        }
    }

    // ============================================================================
    // GL Constants (LWJGL3 GL43C)
    // ============================================================================
    // These match the OpenGL 4.3 specification values
    private static final int GL_SHADER_STORAGE_BUFFER = GL43C.GL_SHADER_STORAGE_BUFFER;
    private static final int GL_COPY_WRITE_BUFFER = GL43C.GL_COPY_WRITE_BUFFER;
    private static final int GL_STATIC_DRAW = GL43C.GL_STATIC_DRAW;
    private static final int GL_DYNAMIC_DRAW = GL43C.GL_DYNAMIC_DRAW;
}

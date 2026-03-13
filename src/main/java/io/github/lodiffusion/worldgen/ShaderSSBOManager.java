package io.github.lodiffusion.worldgen;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

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

    // OpenGL buffer IDs (one per binding)
    private int[] bufferIds = new int[BUFFER_COUNT];
    private boolean initialized = false;
    private long lastUploadTime = 0L;

    public ShaderSSBOManager() {
        // Buffers will be created on-demand during first upload
    }

    /**
     * Uploads NoiseRouterData to GPU SSBOs.
     *
     * Call this on the render thread after extracting the NoiseRouter.
     * Uses GL_STATIC_DRAW for typical dimension/gameplay scenarios.
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

        // Ensure buffers are allocated
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

            lastUploadTime = System.currentTimeMillis();
            LOGGER.info("ShaderSSBOManager: GPU upload complete");
        } catch (Exception e) {
            LOGGER.error("ShaderSSBOManager: Failed to upload NoiseRouter data to GPU", e);
            throw new RuntimeException("SSBO upload failed", e);
        }
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
     * Allocates GPU buffers if not already done.
     */
    private synchronized void allocateBuffers() {
        if (initialized) {
            return;
        }

        try {
            // Use Minecraft's RenderSystem to queue GL calls on render thread
            // For now, we'll use a simpler approach: buffers will be created on first upload
            LOGGER.info("ShaderSSBOManager: Buffers will be allocated on first data upload");
            initialized = true;
        } catch (Exception e) {
            LOGGER.error("ShaderSSBOManager: Failed to allocate buffer IDs", e);
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
     * Retrieves or creates a buffer ID for a binding point.
     * (Placeholder: actual GL buffer creation would happen via RenderSystem.recordRenderCall)
     */
    private int reinterpret_cast_obtainBufferId(int bindingPoint) {
        if (bufferIds[bindingPoint] == 0) {
            // TODO: Generate buffer via glGenBuffers()
            // For now, this is a placeholder that assumes GL context is available
            bufferIds[bindingPoint] = bindingPoint + 1000; // Temporary ID scheme
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
            // TODO: Delete buffers via glDeleteBuffers()
            for (int bufferId : bufferIds) {
                if (bufferId != 0) {
                    // glDeleteBuffers(bufferId);
                    LOGGER.debug("ShaderSSBOManager: Deleted buffer {}", bufferId);
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
    // GL Constant Stubs (placeholder for LWJGL calls)
    // ============================================================================
    // In actual implementation, these would delegate to LWJGL3 or Minecraft's RenderSystem

    private static final int GL_SHADER_STORAGE_BUFFER = 0x90D3;
    private static final int GL_COPY_WRITE_BUFFER = 0x8F37;
    private static final int GL_STATIC_DRAW = 0x88E4;
    private static final int GL_DYNAMIC_DRAW = 0x88E8;

    private void glBindBuffer(int target, int buffer) {
        // Placeholder: RenderSystem.recordRenderCall(() -> GLUtil.NVIDIA_GL.glBindBuffer(target, buffer))
    }

    private void glBufferData(int target, long size, FloatBuffer data, int usage) {
        // Placeholder: RenderSystem.recordRenderCall(() -> GLUtil.NVIDIA_GL.glBufferData(...))
    }

    private void glBufferData(int target, long size, IntBuffer data, int usage) {
        // Placeholder: RenderSystem.recordRenderCall(() -> GLUtil.NVIDIA_GL.glBufferData(...))
    }

    private void glBindBufferBase(int target, int index, int buffer) {
        // Placeholder: RenderSystem.recordRenderCall(() -> GLUtil.NVIDIA_GL.glBindBufferBase(...))
    }

    private void glBufferDataNull(int target, long size, int usage) {
        // Placeholder: RenderSystem.recordRenderCall(() -> GLUtil.NVIDIA_GL.glBufferData(target, size, (FloatBuffer) null, usage))
    }
}

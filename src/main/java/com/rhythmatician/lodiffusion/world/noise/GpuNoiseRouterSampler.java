package com.rhythmatician.lodiffusion.world.noise;

import com.rhythmatician.lodiffusion.HelloTerrainMod;

import net.minecraft.world.gen.noise.NoiseConfig;

/**
 * GPU-backed {@link NoiseRouterSampler} that reads terrain data computed by the
 * shadow router compute pipeline.
 *
 * <h2>Current status (partial / hybrid)</h2>
 * The shadow router GPU shader currently evaluates {@code finalDensity} and the
 * 5 climate fields internally (for biome classification), but does <b>not</b>
 * expose all 15 {@link RouterField}s as individual SSBO outputs.  Until the
 * shader is extended (WS-future), this sampler operates in <b>hybrid mode</b>:
 *
 * <ul>
 *   <li>Fields the GPU computes: read from SSBO readback (TBD — currently
 *       falls through to CPU).</li>
 *   <li>Fields the GPU does not yet compute: delegated to a
 *       {@link VanillaNoiseRouterSampler} for CPU evaluation.</li>
 * </ul>
 *
 * <p>Once the shader emits all 15 fields at quart resolution, the CPU fallback
 * is removed and this becomes a pure GPU readback path.
 *
 * <p>Marked {@code @Experimental} — the GPU path is not yet validated for
 * bit-parity with vanilla.
 *
 * @see NoiseRouterSampler
 * @see VanillaNoiseRouterSampler
 */
public final class GpuNoiseRouterSampler implements NoiseRouterSampler {

    /**
     * CPU fallback for fields the GPU doesn't currently output.
     * In the future, when the shader outputs all 15 fields, this will be
     * removed and all sampling will come from GPU readback.
     */
    private final VanillaNoiseRouterSampler cpuFallback;

    /**
     * @param noiseConfig the server's NoiseConfig (used for CPU fallback)
     */
    public GpuNoiseRouterSampler(NoiseConfig noiseConfig) {
        this.cpuFallback = new VanillaNoiseRouterSampler(noiseConfig);
    }

    @Override
    public SectionNoiseData sampleSection(int sectionX, int sectionY, int sectionZ) {
        // ── Phase 1 (current): full CPU fallback ──────────────────────
        // TODO: When terrain_compute.comp is extended to emit all 15
        //       RouterField values at quart resolution per section,
        //       replace this with GPU SSBO readback:
        //
        //       1. Enqueue (sectionX, sectionZ) on the ShadowRouterJobQueue
        //       2. Wait for the dispatch to complete (glMemoryBarrier)
        //       3. glGetBufferSubData for the relevant Y-slice
        //       4. Unpack the 15-channel quart-resolution buffer into
        //          float[960] and construct SectionNoiseData
        //
        //       Fields that the GPU already computes internally but does not
        //       yet expose as outputs:
        //         - FINAL_DENSITY (binding 7, block-res — needs quart-res variant)
        //         - TEMPERATURE, VEGETATION, CONTINENTS, EROSION, DEPTH, RIDGES
        //           (evaluated in the biome classifier, not written to SSBO)

        return cpuFallback.sampleSection(sectionX, sectionY, sectionZ);
    }

    @Override
    public String backendName() {
        return "gpu";
    }

    @Override
    public void close() {
        cpuFallback.close();
        HelloTerrainMod.LOGGER.info(
                "[GpuNoiseRouterSampler] Closed (hybrid mode — CPU fallback released)");
    }
}

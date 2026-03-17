package com.rhythmatician.lodiffusion.world.noise;

import com.rhythmatician.lodiffusion.Config;
import com.rhythmatician.lodiffusion.HelloTerrainMod;

import net.minecraft.world.gen.noise.NoiseConfig;

/**
 * Creates and manages {@link NoiseRouterSampler} instances based on the
 * {@code "terrainBackend"} config key.
 *
 * <p>The factory supports <b>hot-swapping</b>: it re-reads the config on every
 * call to {@link #getSampler()}.  When the requested backend changes, the old
 * sampler is closed and a new one is created.  This allows switching between
 * {@code "vanilla"} and {@code "gpu"} at runtime without restarting the world.
 *
 * <h2>Lifecycle</h2>
 * <ol>
 *   <li>{@link #create(NoiseConfig)} — called once at world load.</li>
 *   <li>{@link #getSampler()} — called per-section by the generation service.</li>
 *   <li>{@link #close()} — called on world unload to release resources.</li>
 * </ol>
 *
 * @see NoiseRouterSampler
 * @see Config#terrainBackend()
 */
public final class NoiseRouterSamplerFactory implements AutoCloseable {

    private final NoiseConfig noiseConfig;

    /** Currently active sampler (volatile for hot-swap visibility). */
    private volatile NoiseRouterSampler activeSampler;

    /** The backend name of the currently active sampler. */
    private volatile String activeBackendKey;

    private NoiseRouterSamplerFactory(NoiseConfig noiseConfig) {
        this.noiseConfig = noiseConfig;
    }

    /**
     * Create a factory bound to the given {@link NoiseConfig}.
     *
     * <p>The first sampler is created lazily on the first call to
     * {@link #getSampler()}.
     *
     * @param noiseConfig the server's NoiseConfig (never null)
     * @return a new factory
     */
    public static NoiseRouterSamplerFactory create(NoiseConfig noiseConfig) {
        return new NoiseRouterSamplerFactory(noiseConfig);
    }

    /**
     * Return the active {@link NoiseRouterSampler}, creating or swapping it if
     * the {@code "terrainBackend"} config has changed since the last call.
     *
     * <p>This method is safe to call from any thread.  Sampler creation (which
     * may resolve DensityFunction handles) is synchronized; subsequent reads
     * are lock-free.
     *
     * @return the active sampler, never null
     */
    public NoiseRouterSampler getSampler() {
        String requested = resolveBackendKey(Config.terrainBackend());

        // Fast path: no change
        if (requested.equals(activeBackendKey) && activeSampler != null) {
            return activeSampler;
        }

        // Slow path: create / swap
        synchronized (this) {
            // Double-check after acquiring lock
            if (requested.equals(activeBackendKey) && activeSampler != null) {
                return activeSampler;
            }

            NoiseRouterSampler oldSampler = activeSampler;
            NoiseRouterSampler newSampler = createSampler(requested);

            activeSampler = newSampler;
            activeBackendKey = requested;

            if (oldSampler != null) {
                oldSampler.close();
                HelloTerrainMod.LOGGER.info(
                        "[NoiseRouterSamplerFactory] Switched backend: {} → {}",
                        oldSampler.backendName(), newSampler.backendName());
            } else {
                HelloTerrainMod.LOGGER.info(
                        "[NoiseRouterSamplerFactory] Initialized backend: {}",
                        newSampler.backendName());
            }

            return newSampler;
        }
    }

    @Override
    public void close() {
        synchronized (this) {
            if (activeSampler != null) {
                activeSampler.close();
                HelloTerrainMod.LOGGER.info(
                        "[NoiseRouterSamplerFactory] Closed (backend was: {})",
                        activeSampler.backendName());
                activeSampler = null;
                activeBackendKey = null;
            }
        }
    }

    // ── internals ─────────────────────────────────────────────────────

    /**
     * Resolve {@code "auto"} to a concrete backend name.
     *
     * <p>Currently {@code "auto"} maps to {@code "vanilla"}.  When the shadow
     * router GPU pipeline covers all 15 fields at quart resolution, this will
     * prefer {@code "gpu"} if a GL context is available.
     */
    private static String resolveBackendKey(String raw) {
        if (raw == null || raw.isBlank()) raw = "auto";
        return switch (raw.toLowerCase(java.util.Locale.ROOT)) {
            case "auto"    -> "vanilla";   // TODO: prefer "gpu" when shader is complete
            case "vanilla" -> "vanilla";
            case "gpu"     -> "gpu";
            default -> {
                HelloTerrainMod.LOGGER.warn(
                        "[NoiseRouterSamplerFactory] Unknown terrainBackend '{}', falling back to vanilla",
                        raw);
                yield "vanilla";
            }
        };
    }

    private NoiseRouterSampler createSampler(String backendKey) {
        return switch (backendKey) {
            case "gpu"     -> new GpuNoiseRouterSampler(noiseConfig);
            default        -> new VanillaNoiseRouterSampler(noiseConfig);
        };
    }
}

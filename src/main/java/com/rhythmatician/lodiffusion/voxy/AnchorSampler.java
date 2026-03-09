package com.rhythmatician.lodiffusion.voxy;

import net.minecraft.registry.entry.RegistryEntry;
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
        int[][]    biomeIdx,       // [16][16]
        float[][]  rawHm           // [16][16] surface block-Y (may be null for chunk/synth path)
    ) {
        /** Backwards-compat constructor (rawHm not available). */
        public AnchorInputs(float[][] heightPlanes5, float[][] router6, int[][] biomeIdx) {
            this(heightPlanes5, router6, biomeIdx, null);
        }
    }

    /**
     * Sample v2 anchor inputs for the 16×16 section at the given chunk coordinates.
     *
     * <p><strong>DEPRECATED:</strong> This method uses {@link #approximateRouter6}
     * which produces a distribution mismatch vs real noise-router values.
     * Prefer {@link #sampleFromNoise} which uses real DensityFunction data.
     *
     * @param chunk       the Minecraft chunk (or null if not loaded — falls back to zeros)
     * @param noiseConfig server-side noise config (currently unused — for future real-noise path)
     * @return an {@link AnchorInputs} record with APPROXIMATE router6 data
     * @deprecated Use {@link #sampleFromNoise(WorldNoiseAccess, int, int)} instead
     */
    @Deprecated
    public static AnchorInputs sample(Chunk chunk, NoiseConfig noiseConfig) {
        LOGGER.warning("AnchorSampler.sample() uses approximateRouter6 — "
                + "quality will be degraded.  Use sampleFromNoise() for real data.");
        int[][] biomeIdx  = sampleBiomes(chunk);
        float[][] hmap    = sampleHeightmap(chunk);
        float[][] heightPlanes = computeHeightPlanes(hmap, null);  // no ocean floor available from chunk path
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
        // 1. Real heightmaps — single ChunkNoiseSampler pass yields both
        //    WORLD_SURFACE_WG and OCEAN_FLOOR_WG (~64× faster than 256
        //    individual getHeight() calls).
        float[][][] bothHmaps = noiseAccess.sampleBothHeightmaps(sectionX, sectionZ);
        float[][] hmap         = bothHmaps[0];  // WORLD_SURFACE_WG
        float[][] oceanFloorHm = bothHmaps[1];  // OCEAN_FLOOR_WG

        // 2. Real biomes from BiomeSource → canonical IDs
        String[][] biomeNames = noiseAccess.sampleBiomeNames(sectionX, sectionZ, hmap);
        int[][] biomeIdx = new int[16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                biomeIdx[x][z] = BiomeMapping.toCanonicalId(biomeNames[x][z]);
            }
        }

        // 3. Height-planes use BOTH surface and ocean-floor heightmaps (matches Python)
        float[][] heightPlanes = computeHeightPlanes(hmap, oceanFloorHm);

        // 4. Real router6 from NoiseRouter density functions
        float[][] router6 = noiseAccess.sampleRouter6(sectionX, sectionZ, hmap);

        return new AnchorInputs(heightPlanes, router6, biomeIdx, hmap);
    }

    // ------------------------------------------------------------------
    // Biome sampling
    // ------------------------------------------------------------------

    /**
     * Extract a [16][16] biome integer-index grid for the chunk.
     * Uses the surface-level biome at each column (y=64).
     *
     * <p>Uses {@link BiomeMapping#toCanonicalId} for stable, deterministic
     * encoding that matches the Python training pipeline.
     */
    static int[][] sampleBiomes(Chunk chunk) {
        int[][] out = new int[16][16];
        if (chunk == null) return out;  // default 0

        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                // Sample biome at sea level as representative
                RegistryEntry<Biome> biomeEntry = chunk.getBiomeForNoiseGen(lx >> 2, 4, lz >> 2);
                out[lx][lz] = BiomeMapping.toCanonicalId(biomeEntry);
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
     *   <li>ocean_floor     — real OCEAN_FLOOR_WG / 320 (or clamped approx if unavailable)</li>
     *   <li>slope_x         — finite difference dH/dx normalised</li>
     *   <li>slope_z         — finite difference dH/dz normalised</li>
     *   <li>curvature       — Laplacian (d²H/dx² + d²H/dz²), normalised</li>
     * </ol>
     *
     * @param hm           surface heightmap [16][16] in block-Y
     * @param oceanFloorHm ocean floor heightmap [16][16] in block-Y, or {@code null}
     *                     to fall back to {@code min(surface, SEA_LEVEL)} approximation
     * @return float[5][256] in row-major order (channel, lx*16+lz)
     */
    static float[][] computeHeightPlanes(float[][] hm, float[][] oceanFloorHm) {
        float[][] planes = new float[5][256];

        // Step 1: Normalise surface (matches Python: surf = height / 320.0)
        float[][] surfNorm = new float[16][16];
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                float h = hm[lx][lz];
                surfNorm[lx][lz] = h / HEIGHT_RANGE;
                planes[0][lx * 16 + lz] = surfNorm[lx][lz];                // surface
                // Ocean floor: use real OCEAN_FLOOR_WG data when available,
                // otherwise fall back to the min(surface, SEA_LEVEL) approximation.
                float of = (oceanFloorHm != null)
                        ? oceanFloorHm[lx][lz]
                        : Math.min(h, SEA_LEVEL);
                planes[1][lx * 16 + lz] = of / HEIGHT_RANGE;              // ocean_floor
            }
        }

        // Step 2: slope_x = np.gradient(surfNorm, axis=x)
        // Matches Python: central differences on NORMALISED surface,
        // forward/backward one-sided at boundaries (np.gradient edge_order=1).
        float[][] slopeX = new float[16][16];
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                if (lx == 0) {
                    slopeX[lx][lz] = surfNorm[1][lz] - surfNorm[0][lz];
                } else if (lx == 15) {
                    slopeX[lx][lz] = surfNorm[15][lz] - surfNorm[14][lz];
                } else {
                    slopeX[lx][lz] = (surfNorm[lx + 1][lz] - surfNorm[lx - 1][lz]) / 2f;
                }
                planes[2][lx * 16 + lz] = slopeX[lx][lz];
            }
        }

        // Step 3: slope_z = np.gradient(surfNorm, axis=z)
        float[][] slopeZ = new float[16][16];
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                if (lz == 0) {
                    slopeZ[lx][lz] = surfNorm[lx][1] - surfNorm[lx][0];
                } else if (lz == 15) {
                    slopeZ[lx][lz] = surfNorm[lx][15] - surfNorm[lx][14];
                } else {
                    slopeZ[lx][lz] = (surfNorm[lx][lz + 1] - surfNorm[lx][lz - 1]) / 2f;
                }
                planes[3][lx * 16 + lz] = slopeZ[lx][lz];
            }
        }

        // Step 4: curvature = np.gradient(slope_x, axis=x) + np.gradient(slope_z, axis=z)
        // This is the Laplacian computed as gradient-of-gradient, matching Python exactly.
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                // d(slope_x)/dx
                float dsx;
                if (lx == 0) {
                    dsx = slopeX[1][lz] - slopeX[0][lz];
                } else if (lx == 15) {
                    dsx = slopeX[15][lz] - slopeX[14][lz];
                } else {
                    dsx = (slopeX[lx + 1][lz] - slopeX[lx - 1][lz]) / 2f;
                }
                // d(slope_z)/dz
                float dsz;
                if (lz == 0) {
                    dsz = slopeZ[lx][1] - slopeZ[lx][0];
                } else if (lz == 15) {
                    dsz = slopeZ[lx][15] - slopeZ[lx][14];
                } else {
                    dsz = (slopeZ[lx][lz + 1] - slopeZ[lx][lz - 1]) / 2f;
                }
                planes[4][lx * 16 + lz] = dsx + dsz;
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
     * <p><strong>DEPRECATED:</strong> This produces a fundamentally different
     * distribution than real noise-router values and should not be used for
     * training or high-quality inference.  Use {@link WorldNoiseAccess#sampleRouter6}
     * instead.
     *
     * @return float[6][256] in row-major order (channel, lx*16+lz)
     * @deprecated Distribution mismatch with real DensityFunction values
     */
    @Deprecated
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

    private static float clampRouter(float v) {
        return Math.max(0f, Math.min(1f, v));
    }
}

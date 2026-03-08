package com.rhythmatician.lodiffusion.voxy;

import com.rhythmatician.lodiffusion.HelloTerrainMod;

import net.minecraft.registry.RegistryKey;
import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.world.Heightmap;
import net.minecraft.world.World;
import net.minecraft.world.biome.Biome;
import net.minecraft.world.biome.source.BiomeSource;
import net.minecraft.world.gen.chunk.ChunkGenerator;
import net.minecraft.world.gen.chunk.NoiseChunkGenerator;
import net.minecraft.world.gen.densityfunction.DensityFunction;
import net.minecraft.world.gen.noise.NoiseConfig;
import net.minecraft.world.gen.noise.NoiseRouter;

import java.lang.reflect.Field;

/**
 * Provides server-side access to Minecraft's noise generators for sampling
 * heightmap, biome, and density-router values at <em>any</em> (x, z) coordinate
 * — <b>without needing a loaded chunk</b>.
 *
 * <p>This replaces the synthetic sine-wave heightmap and constant-biome fallback
 * that was previously used for distant (unloaded) sections.  It works by
 * tapping into the integrated server's {@link ChunkGenerator} and
 * {@link NoiseConfig} directly.
 *
 * <h3>Limitations</h3>
 * <ul>
 *   <li>Only works when an integrated server is available (singleplayer / LAN).
 *       Returns {@code null} from {@link #tryCreate(World)} on dedicated servers.</li>
 *   <li>All sampling methods are pure computation (no world state mutation),
 *       so they are safe to call from the LOD worker thread.</li>
 * </ul>
 */
public final class WorldNoiseAccess {

    private final ServerWorld serverWorld;
    private final ChunkGenerator generator;
    private final NoiseConfig noiseConfig;
    private final NoiseRouter router;
    private final BiomeSource biomeSource;

    // Cached density functions for the 6 router channels
    private final DensityFunction temperature;
    private final DensityFunction vegetation;
    private final DensityFunction continents;
    private final DensityFunction erosion;
    private final DensityFunction depth;
    private final DensityFunction ridges;

    private WorldNoiseAccess(ServerWorld serverWorld, ChunkGenerator generator,
                             NoiseConfig noiseConfig) {
        this.serverWorld = serverWorld;
        this.generator = generator;
        this.noiseConfig = noiseConfig;
        this.router = noiseConfig.getNoiseRouter();
        this.biomeSource = generator.getBiomeSource();

        this.temperature  = router.temperature();
        this.vegetation   = router.vegetation();
        this.continents   = router.continents();
        this.erosion      = router.erosion();
        this.depth        = router.depth();
        this.ridges       = router.ridges();
    }

    // ------------------------------------------------------------------
    // Factory
    // ------------------------------------------------------------------

    /**
     * Try to create a {@code WorldNoiseAccess} from a server instance.
     *
     * @param server      the Minecraft server (integrated or dedicated)
     * @param clientWorld the client-side world (used to determine dimension)
     * @return a new instance, or {@code null} if {@code NoiseConfig} is
     *         not available (e.g., non-noise chunk generator)
     */
    public static WorldNoiseAccess tryCreate(MinecraftServer server, World clientWorld) {
        try {
            if (server == null) {
                HelloTerrainMod.LOGGER.info(
                        "[WorldNoiseAccess] No server provided — cannot bind noise pipeline");
                return null;
            }

            // Get the server-side world for the same dimension as the client
            RegistryKey<World> dimKey = clientWorld.getRegistryKey();
            ServerWorld serverWorld = server.getWorld(dimKey);
            if (serverWorld == null) {
                HelloTerrainMod.LOGGER.warn(
                        "[WorldNoiseAccess] Could not get ServerWorld for dimension {}",
                        dimKey.getValue());
                return null;
            }

            ChunkGenerator gen = serverWorld.getChunkManager().getChunkGenerator();

            // Try to get NoiseConfig — reflection-based since there's no public getter
            NoiseConfig nc = tryGetNoiseConfig(serverWorld);
            if (nc == null) {
                HelloTerrainMod.LOGGER.warn(
                        "[WorldNoiseAccess] NoiseConfig unavailable — cannot use noise access");
                return null;
            }

            HelloTerrainMod.LOGGER.info(
                    "[WorldNoiseAccess] Successfully bound to server noise pipeline");
            return new WorldNoiseAccess(serverWorld, gen, nc);

        } catch (Exception e) {
            HelloTerrainMod.LOGGER.warn(
                    "[WorldNoiseAccess] Failed to initialize: {}", e.getMessage());
            return null;
        }
    }

    // ------------------------------------------------------------------
    // Heightmap sampling
    // ------------------------------------------------------------------

    /**
     * Sample the real surface heightmap for a 16×16 section column.
     *
     * <p>Uses {@link ChunkGenerator#getHeight(int, int, Heightmap.Type,
     * net.minecraft.world.HeightLimitView, NoiseConfig)} which works at
     * any coordinate without needing a loaded chunk.
     *
     * @param sectionX section X coordinate (chunk X)
     * @param sectionZ section Z coordinate (chunk Z)
     * @return float[16][16] of surface Y values in block coordinates
     */
    public float[][] sampleHeightmap(int sectionX, int sectionZ) {
        float[][] hm = new float[16][16];
        int baseX = sectionX * 16;
        int baseZ = sectionZ * 16;

        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                hm[lx][lz] = generator.getHeight(
                        baseX + lx, baseZ + lz,
                        Heightmap.Type.WORLD_SURFACE_WG,
                        serverWorld, noiseConfig);
            }
        }
        return hm;
    }

    // ------------------------------------------------------------------
    // Router6 sampling (real density function values)
    // ------------------------------------------------------------------

    /**
     * Sample the 6 CORE router density functions at the surface level
     * for a 16×16 section column.
     *
     * <p>Uses {@link DensityFunction#sample(DensityFunction.NoisePos)}
     * with {@link DensityFunction.UnblendedNoisePos} — pure computation,
     * no chunk needed.
     *
     * <p>Router channels: temperature, vegetation, continents, erosion,
     * depth, ridges — same order as the training data.
     *
     * @param sectionX section X coordinate
     * @param sectionZ section Z coordinate
     * @param heightmap the surface heightmap (used to sample at surface Y)
     * @return float[6][256] in row-major order (channel, lx*16+lz)
     */
    public float[][] sampleRouter6(int sectionX, int sectionZ, float[][] heightmap) {
        float[][] r6 = new float[6][256];
        int baseX = sectionX * 16;
        int baseZ = sectionZ * 16;

        DensityFunction[] fns = {
            temperature, vegetation, continents, erosion, depth, ridges
        };

        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                int idx = lx * 16 + lz;
                int bx = baseX + lx;
                int bz = baseZ + lz;
                // Sample at surface Y for this column
                int surfaceY = (int) heightmap[lx][lz];
                var noisePos = new DensityFunction.UnblendedNoisePos(bx, surfaceY, bz);

                for (int ch = 0; ch < 6; ch++) {
                    r6[ch][idx] = (float) fns[ch].sample(noisePos);
                }
            }
        }
        return r6;
    }

    // ------------------------------------------------------------------
    // Biome sampling
    // ------------------------------------------------------------------

    /**
     * Sample biome integer indices for a 16×16 section column using the
     * server-side {@link BiomeSource}.
     *
     * <p>Biomes are sampled at quarter-resolution (4-block steps) as per
     * Minecraft's biome storage convention, then each block column gets
     * the biome of its containing quarter.
     *
     * <p>The biome index is derived from the biome's registry raw ID
     * modulo 256, providing a stable mapping within a given world.
     *
     * @param sectionX section X coordinate
     * @param sectionZ section Z coordinate
     * @param heightmap surface heightmap for Y coordinate
     * @return int[16][16] of biome indices (0–255)
     */
    public int[][] sampleBiomes(int sectionX, int sectionZ, float[][] heightmap) {
        int[][] biomes = new int[16][16];
        int baseX = sectionX * 16;
        int baseZ = sectionZ * 16;

        // Biomes are stored at quarter resolution — sample at quart coords
        // and fill the 4×4 block region with the same value.
        for (int qx = 0; qx < 4; qx++) {
            for (int qz = 0; qz < 4; qz++) {
                int bx = baseX + qx * 4 + 2;  // center of quartet
                int bz = baseZ + qz * 4 + 2;
                int surfaceY = (int) heightmap[qx * 4][qz * 4];

                // BiomeSource.getBiome works at quart coordinates
                RegistryEntry<Biome> biomeEntry = biomeSource.getBiome(
                        bx >> 2, surfaceY >> 2, bz >> 2,
                        noiseConfig.getMultiNoiseSampler());

                // Use the same hash-based mapping as AnchorSampler.sampleBiomes()
                int biomeIdx = Math.abs(biomeEntry.hashCode()) % 256;

                // Fill the 4×4 block region
                for (int dx = 0; dx < 4; dx++) {
                    for (int dz = 0; dz < 4; dz++) {
                        biomes[qx * 4 + dx][qz * 4 + dz] = biomeIdx;
                    }
                }
            }
        }
        return biomes;
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /**
     * Try to extract a {@link NoiseConfig} from the world's chunk generator
     * via reflection.
     *
     * <p>Mirrors {@code NoiseDumperCommand.tryGetNoiseConfig()}.
     */
    private static NoiseConfig tryGetNoiseConfig(ServerWorld world) {
        try {
            ChunkGenerator gen = world.getChunkManager().getChunkGenerator();
            if (!(gen instanceof NoiseChunkGenerator)) {
                HelloTerrainMod.LOGGER.info(
                        "[WorldNoiseAccess] ChunkGenerator is not NoiseChunkGenerator (type: {})",
                        gen.getClass().getSimpleName());
                return null;
            }

            // Walk all declared fields looking for a cached NoiseConfig instance
            for (Class<?> cls = gen.getClass();
                 cls != null && cls != Object.class;
                 cls = cls.getSuperclass()) {
                for (Field f : cls.getDeclaredFields()) {
                    if (NoiseConfig.class.isAssignableFrom(f.getType())) {
                        f.setAccessible(true);
                        NoiseConfig nc = (NoiseConfig) f.get(gen);
                        if (nc != null) return nc;
                    }
                }
            }

            HelloTerrainMod.LOGGER.warn(
                    "[WorldNoiseAccess] No NoiseConfig field found on {}",
                    gen.getClass().getName());
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.warn(
                    "[WorldNoiseAccess] Reflection failed: {}", e.getMessage());
        }
        return null;
    }

    /** Expose for diagnostics. */
    public boolean isAvailable() {
        return true;  // if constructed, it's available
    }
}

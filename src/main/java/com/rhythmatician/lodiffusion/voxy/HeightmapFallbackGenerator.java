package com.rhythmatician.lodiffusion.voxy;

import java.lang.reflect.Method;
import java.util.logging.Logger;

import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;

/**
 * Ultra-fast fallback terrain generator for when no ONNX models are available.
 *
 * <p>Fills sections with a simple heightmap-based algorithm:
 * <ul>
 *   <li>Below y=0 → deepslate</li>
 *   <li>y=0 to (surface − 3) → stone</li>
 *   <li>Top 3 solid blocks → biome-dependent (sand for desert/beach, else dirt/dirt/grass_block)</li>
 *   <li>Above surface, below sea level → water</li>
 *   <li>Above surface, at or above sea level → air</li>
 * </ul>
 *
 * <p>This bypasses the entire ONNX pipeline and {@link VoxySectionWriter},
 * composing 64-bit Voxy voxels directly via {@link VoxyCompat#composeVoxel}
 * for maximum throughput.  With no compute bottleneck, performance is
 * limited only by Voxy I/O ({@code insertUpdate}).
 *
 * <p>The generator is stateless — all mutable state (block IDs, biome IDs)
 * is held externally in {@link FallbackBlockIds} and passed per call.
 */
public final class HeightmapFallbackGenerator {

    private static final Logger LOGGER = Logger.getLogger(HeightmapFallbackGenerator.class.getName());

    /** Minecraft sea level in block Y coordinates. */
    static final int SEA_LEVEL = 63;

    /** Canonical biome index for minecraft:desert (alphabetical position 12). */
    private static final int BIOME_DESERT = 12;

    /** Canonical biome index for minecraft:beach (alphabetical position 2). */
    private static final int BIOME_BEACH = 2;

    /** Default light value: full sky light, no block light → 0x0F. */
    private static final int DEFAULT_LIGHT = 0x0F;

    private HeightmapFallbackGenerator() {}

    // ------------------------------------------------------------------ //
    //  Block ID resolution
    // ------------------------------------------------------------------ //

    /**
     * Pre-resolved Voxy block IDs for the 7 block types used by the fallback.
     * Resolved once at startup via {@link #resolveBlockIds(Object)}.
     */
    public record FallbackBlockIds(
        int air,
        int stone,
        int deepslate,
        int dirt,
        int grassBlock,
        int sand,
        int water
    ) {}

    /**
     * Resolve Voxy internal block IDs for the 7 block types needed by the
     * fallback generator.  Uses the same {@code Mapper.getIdForBlockState}
     * reflection pattern as {@link VoxyBlockMapper#build}.
     *
     * @param voxyMapper the Voxy Mapper object (from {@link VoxyCompat#getMapper})
     * @return pre-resolved block IDs
     * @throws RuntimeException if reflection fails
     */
    public static FallbackBlockIds resolveBlockIds(Object voxyMapper) {
        try {
            Method getIdMethod = voxyMapper.getClass().getMethod("getIdForBlockState",
                    BlockState.class);

            int air        = (int) getIdMethod.invoke(voxyMapper, Blocks.AIR.getDefaultState());
            int stone      = (int) getIdMethod.invoke(voxyMapper, Blocks.STONE.getDefaultState());
            int deepslate  = (int) getIdMethod.invoke(voxyMapper, Blocks.DEEPSLATE.getDefaultState());
            int dirt       = (int) getIdMethod.invoke(voxyMapper, Blocks.DIRT.getDefaultState());
            int grassBlock = (int) getIdMethod.invoke(voxyMapper, Blocks.GRASS_BLOCK.getDefaultState());
            int sand       = (int) getIdMethod.invoke(voxyMapper, Blocks.SAND.getDefaultState());
            int water      = (int) getIdMethod.invoke(voxyMapper, Blocks.WATER.getDefaultState());

            LOGGER.info("Fallback block IDs resolved: air=" + air + " stone=" + stone
                    + " deepslate=" + deepslate + " dirt=" + dirt + " grass=" + grassBlock
                    + " sand=" + sand + " water=" + water);

            return new FallbackBlockIds(air, stone, deepslate, dirt, grassBlock, sand, water);
        } catch (Exception e) {
            throw new RuntimeException("Failed to resolve fallback block IDs from Voxy Mapper", e);
        }
    }

    // ------------------------------------------------------------------ //
    //  Biome ID resolution (shared with VoxyBlockMapper)
    // ------------------------------------------------------------------ //

    /**
     * Resolve canonical biome IDs → Voxy biome IDs.
     *
     * <p>Same reflection-based approach as {@link VoxyBlockMapper}, but
     * usable without a model vocabulary.  Maps all 54 canonical overworld
     * biomes to their Voxy internal IDs.
     *
     * @param voxyMapper the Voxy Mapper object
     * @return int[54] mapping canonical biome index → Voxy biome ID
     */
    public static int[] resolveBiomeMappings(Object voxyMapper) {
        int[] map = new int[BiomeMapping.size()];
        try {
            Method getBiomeEntries = voxyMapper.getClass().getMethod("getBiomeEntries");
            Object[] entries = (Object[]) getBiomeEntries.invoke(voxyMapper);

            if (entries == null || entries.length == 0) {
                LOGGER.warning("No biome entries from Voxy — all biomes map to 0");
                return map;
            }

            int resolved = 0;
            for (Object entry : entries) {
                String biomeName = (String) entry.getClass().getField("biome").get(entry);
                int voxyId = entry.getClass().getField("id").getInt(entry);

                int canonicalId = BiomeMapping.toCanonicalId(biomeName);
                if (canonicalId != BiomeMapping.UNKNOWN_BIOME_ID) {
                    map[canonicalId] = voxyId;
                    resolved++;
                }
            }

            LOGGER.info("Fallback biome mappings: resolved " + resolved + "/"
                    + BiomeMapping.size() + " from " + entries.length + " Voxy entries");

        } catch (Exception e) {
            LOGGER.warning("Failed to resolve biome mappings: " + e.getMessage()
                    + " — all biomes will use default (0)");
        }
        return map;
    }

    // ------------------------------------------------------------------ //
    //  Section generation
    // ------------------------------------------------------------------ //

    /**
     * Generate a single 16³ section filled according to the heightmap rules.
     *
     * <p>This method creates a {@code VoxelizedSection}, fills its L0 data,
     * computes the mip pyramid, and is ready for {@link VoxyCompat#insertUpdate}.
     *
     * <p>When {@code oceanFloorHm} is provided (non-null), it is used as the
     * real solid ground surface for columns where water is present.  Water is
     * placed between the ocean/river floor and the water surface ({@code rawHm}).
     * The top 3 solid blocks are placed relative to the floor, not the water
     * surface, so riverbeds get sand/dirt/grass correctly.
     *
     * @param sectionX      section X coordinate (blockX / 16)
     * @param sectionY      section Y coordinate (blockY / 16)
     * @param sectionZ      section Z coordinate (blockZ / 16)
     * @param rawHm         [16][16] surface heightmap (water surface) in block Y, indexed [x][z]
     * @param oceanFloorHm  [16][16] ocean/river floor heightmap in block Y, or null
     * @param biomeIdx      [16][16] canonical biome indices, indexed [x][z]
     * @param biomeVoxyIds  [16][16] Voxy biome IDs, indexed [x][z]
     * @param blockIds      pre-resolved Voxy block IDs
     * @param voxyMapper    Voxy Mapper for mip computation
     * @return the filled and mipped {@code VoxelizedSection}, or {@code null}
     *         if the section is entirely air (skip insertion)
     */
    public static Object generateSection(int sectionX, int sectionY, int sectionZ,
                                          float[][] rawHm, float[][] oceanFloorHm,
                                          int[][] biomeIdx,
                                          int[][] biomeVoxyIds,
                                          FallbackBlockIds blockIds,
                                          Object voxyMapper) {
        int baseY = sectionY * 16;

        // Quick check: if the entire section is above the max heightmap AND
        // above sea level, it's all air — skip it.
        if (baseY >= SEA_LEVEL) {
            boolean allAboveSurface = true;
            for (int lx = 0; lx < 16 && allAboveSurface; lx++) {
                for (int lz = 0; lz < 16 && allAboveSurface; lz++) {
                    if (baseY < rawHm[lx][lz]) {
                        allAboveSurface = false;
                    }
                }
            }
            if (allAboveSurface) return null;
        }

        Object section = VoxyCompat.createEmptySection();
        VoxyCompat.setSectionPosition(section, sectionX, sectionY, sectionZ);
        long[] data = VoxyCompat.getSectionData(section);

        int nonAir = 0;

        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                float waterSurfaceY = rawHm[lx][lz];
                // If ocean floor data is available, use it as the solid ground.
                // Otherwise, fall back to the surface heightmap (no water distinction).
                float groundY = oceanFloorHm != null ? oceanFloorHm[lx][lz] : waterSurfaceY;
                int waterSurfaceBlockY = (int) Math.floor(waterSurfaceY);
                int groundBlockY = (int) Math.floor(groundY);

                int canonBiome = biomeIdx[lx][lz];
                int voxyBiome = biomeVoxyIds[lx][lz];

                boolean isSandy = canonBiome == BIOME_DESERT || canonBiome == BIOME_BEACH;

                for (int ly = 0; ly < 16; ly++) {
                    int worldY = baseY + ly;
                    int idx = VoxyCompat.l0Index(lx, ly, lz);

                    int blockId = pickBlockId(worldY, groundBlockY,
                            waterSurfaceBlockY, isSandy, blockIds);

                    data[idx] = VoxyCompat.composeVoxel(blockId, voxyBiome, DEFAULT_LIGHT);

                    if (blockId != blockIds.air()) {
                        nonAir++;
                    }
                }
            }
        }

        if (nonAir == 0) return null;

        VoxyCompat.setNonAirCount(section, nonAir);
        VoxyCompat.mipSection(section, voxyMapper);
        return section;
    }

    /**
     * Determine whether a canonical biome index should use sandy surface blocks.
     * Package-private for testing.
     *
     * @param canonicalBiomeIdx biome index from {@link BiomeMapping}
     * @return true if the biome should use sand instead of grass/dirt
     */
    static boolean isSandyBiome(int canonicalBiomeIdx) {
        return canonicalBiomeIdx == BIOME_DESERT || canonicalBiomeIdx == BIOME_BEACH;
    }

    /**
     * Determine the Voxy block ID for a voxel at a given world Y, considering
     * both the solid ground surface and the water surface.
     *
     * <p>When {@code groundBlockY < waterSurfaceBlockY}, water exists in that
     * column (river, ocean, lake).  Voxels between the ground and the water
     * surface are filled with water.
     *
     * <p>Package-private for testing.
     *
     * @param worldY              absolute block Y coordinate
     * @param groundBlockY        the solid ground height (floor of ocean floor or surface heightmap)
     * @param waterSurfaceBlockY  the water surface height (floor of WORLD_SURFACE_WG heightmap)
     * @param isSandy             true if the biome uses sand
     * @param blockIds            pre-resolved block IDs
     * @return the Voxy block ID to use
     */
    static int pickBlockId(int worldY, int groundBlockY, int waterSurfaceBlockY,
                            boolean isSandy, FallbackBlockIds blockIds) {
        if (worldY >= groundBlockY) {
            // Above solid ground — could be water or air
            if (worldY < waterSurfaceBlockY) {
                // Between ground and water surface — water
                return blockIds.water();
            }
            // Above both ground and water surface
            return (worldY < SEA_LEVEL) ? blockIds.water() : blockIds.air();
        } else if (worldY >= groundBlockY - 3) {
            // Top 3 solid blocks
            if (isSandy) {
                return blockIds.sand();
            }
            int depth = groundBlockY - 1 - worldY;
            return (depth == 0) ? blockIds.grassBlock() : blockIds.dirt();
        } else if (worldY < 0) {
            return blockIds.deepslate();
        } else {
            return blockIds.stone();
        }
    }
}

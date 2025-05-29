package com.rhythmatician.lodiffusion;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for LOD (Level of Detail) integration.
 * Drives integration with Distant Horizons and progressive refinement.
 */
class LODIntegrationTest {

    private DiffusionChunkGenerator generator;

    @BeforeEach
    void setUp() {
        generator = new DiffusionChunkGenerator();
    }

    @Test
    void testLODFactorBasicMapping() {
        // Given: Different LOD levels
        int lod0 = 0; // Highest detail
        int lod1 = 1; // Medium detail
        int lod2 = 2; // Lower detail

        // When/Then: Should map LOD to diffusion factors
        assertDoesNotThrow(() -> {
            // This will drive the creation of LOD mapping logic
            generator.buildSurfaceWithLOD(0, 0, new int[16][16], new String[16], lod0);
            generator.buildSurfaceWithLOD(0, 0, new int[16][16], new String[16], lod1);
            generator.buildSurfaceWithLOD(0, 0, new int[16][16], new String[16], lod2);
        });
    }

    @Test
    void testProgressiveRefinementLOD0ToLOD1() {
        // Given: Heightmap processed at LOD 1 (lower detail)
        int[][] heightmap = new int[16][16];
        String[] biomes = new String[16];

        // Initialize with flat terrain
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                heightmap[x][z] = 64;
            }
        }
        for (int i = 0; i < 16; i++) {
            biomes[i] = "minecraft:plains";
        }

        // Process at LOD 1 first
        generator.buildSurfaceWithLOD(0, 0, heightmap, biomes, 1);

        // Store LOD 1 result
        int[][] lod1Result = new int[16][16];
        for (int x = 0; x < 16; x++) {
            System.arraycopy(heightmap[x], 0, lod1Result[x], 0, 16);
        }

        // When: Process at LOD 0 (higher detail) - should refine, not recompute
        generator.buildSurfaceWithLOD(0, 0, heightmap, biomes, 0);

        // Then: LOD 0 should be based on LOD 1 result (progressive refinement)
        // This drives the progressive refinement requirement
        boolean hasRefinementPattern = false;
        for (int x = 1; x < 15; x++) {
            for (int z = 1; z < 15; z++) {
                // Check if LOD 0 preserves general structure from LOD 1
                int lod1Base = lod1Result[x][z];
                int lod0Refined = heightmap[x][z];

                // Should be refined but not completely different
                if (Math.abs(lod0Refined - lod1Base) <= 5) {
                    hasRefinementPattern = true;
                    break;
                }
            }
            if (hasRefinementPattern) break;
        }

        assertTrue(hasRefinementPattern,
            "LOD 0 should progressively refine LOD 1, not recompute from scratch");
    }

    @Test
    void testLODDiffusionFactorMapping() {
        // Given: Test heightmap data
        int[][] heightmap = new int[16][16];
        String[] biomes = new String[16];

        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                heightmap[x][z] = 60 + (x + z) % 8;
            }
        }
        for (int i = 0; i < 16; i++) {
            biomes[i] = "minecraft:mountains";
        }

        // Store original for comparison
        int[][] original = new int[16][16];
        for (int x = 0; x < 16; x++) {
            System.arraycopy(heightmap[x], 0, original[x], 0, 16);
        }

        // When: Process with different LOD levels
        int[][] lod0Result = new int[16][16];
        int[][] lod2Result = new int[16][16];

        // Copy for LOD 0 test
        for (int x = 0; x < 16; x++) {
            System.arraycopy(original[x], 0, lod0Result[x], 0, 16);
        }
        generator.buildSurfaceWithLOD(0, 0, lod0Result, biomes, 0);

        // Copy for LOD 2 test
        for (int x = 0; x < 16; x++) {
            System.arraycopy(original[x], 0, lod2Result[x], 0, 16);
        }
        generator.buildSurfaceWithLOD(0, 0, lod2Result, biomes, 2);

        // Then: Higher LOD should have more detail variation than lower LOD
        int lod0Variation = calculateVariation(lod0Result);
        int lod2Variation = calculateVariation(lod2Result);

        assertTrue(lod0Variation >= lod2Variation,
            "LOD 0 (high detail) should have equal or more variation than LOD 2 (low detail)");
    }

    private int calculateVariation(int[][] heightmap) {
        int totalVariation = 0;
        for (int x = 1; x < 15; x++) {
            for (int z = 1; z < 15; z++) {
                int current = heightmap[x][z];
                int neighbors = heightmap[x-1][z] + heightmap[x+1][z] +
                              heightmap[x][z-1] + heightmap[x][z+1];
                totalVariation += Math.abs(current * 4 - neighbors);
            }
        }
        return totalVariation;
    }
}

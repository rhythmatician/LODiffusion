package com.rhythmatician.lodiffusion;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for DiffusionModel.
 * Follows TDD approach - testing diffusion algorithm behavior.
 */
class DiffusionModelTest {

    private DiffusionModel model;

    @BeforeEach
    void setUp() {
        model = new DiffusionModel();
    }

    @Test
    void testDiffusionModelExists() {
        // Given/When/Then: DiffusionModel should be instantiable
        assertNotNull(model);
    }

    @Test
    void testRunMethodExists() {
        // Given: A DiffusionModel instance and test heightmap data
        int[][] heightmap = new int[16][16];
        String[] biomes = new String[16];

        // Fill with test data
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                heightmap[x][z] = 64;
            }
        }
        for (int i = 0; i < 16; i++) {
            biomes[i] = "minecraft:plains";
        }

        // When/Then: run method should exist and be callable
        // This drives the creation of the run method signature
        assertDoesNotThrow(() -> {
            model.run(heightmap, biomes);
        });
    }

    @Test
    void testRunMethodModifiesHeightmap() {
        // Given: A DiffusionModel instance and baseline heightmap
        int[][] heightmap = new int[16][16];
        String[] biomes = new String[16];

        // Initialize with flat terrain
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                heightmap[x][z] = 64;
            }
        }
        for (int i = 0; i < 16; i++) {
            biomes[i] = "minecraft:mountains";
        }

        // Store original data for comparison
        int[][] original = new int[16][16];
        for (int x = 0; x < 16; x++) {
            System.arraycopy(heightmap[x], 0, original[x], 0, 16);
        }

        // When: run method processes the heightmap
        model.run(heightmap, biomes);

        // Then: heightmap should be modified by diffusion algorithm
        boolean modified = false;
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (heightmap[x][z] != original[x][z]) {
                    modified = true;
                    break;
                }
            }
            if (modified) break;
        }

        assertTrue(modified, "DiffusionModel.run() should modify heightmap data");
    }

    @Test
    void testRunMethodWithMultipleChannels() {
        // Given: A DiffusionModel instance and multi-channel data
        float[][][] channels = new float[3][16][16]; // 3 channels: height, biome, temperature
        String[] biomes = new String[16];

        // Initialize channels with test data
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                channels[0][x][z] = 64.0f; // Height channel
                channels[1][x][z] = 1.0f;  // Biome channel (plains)
                channels[2][x][z] = 0.8f;  // Temperature channel
            }
        }
        for (int i = 0; i < 16; i++) {
            biomes[i] = "minecraft:plains";
        }

        // Store original data for comparison
        float[][][] original = new float[3][16][16];
        for (int c = 0; c < 3; c++) {
            for (int x = 0; x < 16; x++) {
                System.arraycopy(channels[c][x], 0, original[c][x], 0, 16);
            }
        }

        // When: run method processes the multi-channel data
        assertDoesNotThrow(() -> {
            model.run(channels, biomes);
        });

        // Then: channels should be modified by diffusion algorithm
        boolean modified = false;
        for (int c = 0; c < 3; c++) {
            for (int x = 0; x < 16; x++) {
                for (int z = 0; z < 16; z++) {
                    if (Math.abs(channels[c][x][z] - original[c][x][z]) > 0.001f) {
                        modified = true;
                        break;
                    }
                }
                if (modified) break;
            }
            if (modified) break;
        }

        assertTrue(modified, "DiffusionModel.run() should modify multi-channel data");
    }

    @Test
    void testRunMethodWithLODAndChannels() {
        // Given: A DiffusionModel instance with LOD and multi-channel data
        float[][][] channels = new float[2][16][16]; // 2 channels: height, biome
        String[] biomes = new String[16];
        int lod = 1; // Medium detail

        // Initialize with test data
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                channels[0][x][z] = 64.0f; // Height
                channels[1][x][z] = 2.0f;  // Mountain biome
            }
        }
        for (int i = 0; i < 16; i++) {
            biomes[i] = "minecraft:mountains";
        }

        // When/Then: LOD-aware multi-channel processing should work
        assertDoesNotThrow(() -> {
            model.runWithLOD(lod, channels, biomes);
        });
    }
}

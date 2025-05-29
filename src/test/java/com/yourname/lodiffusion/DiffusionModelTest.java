package com.yourname.lodiffusion;

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
}

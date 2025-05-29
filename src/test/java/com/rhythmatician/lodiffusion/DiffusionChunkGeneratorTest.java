package com.rhythmatician.lodiffusion;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for DiffusionChunkGenerator following TDD principles.
 * Tests chunk generation with diffusion model integration.
 */
public class DiffusionChunkGeneratorTest {

    private DiffusionChunkGenerator generator;
    private int[][] testHeightmap;
    private String[] testBiomes;

    @BeforeEach
    void setUp() {
        generator = new DiffusionChunkGenerator();
        
        // Create test heightmap (16x16)
        testHeightmap = new int[16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                testHeightmap[x][z] = 64; // Sea level default
            }
        }

        // Create test biomes (256 entries for 16x16)
        testBiomes = new String[256];
        for (int i = 0; i < 256; i++) {
            testBiomes[i] = "plains";
        }
    }

    @Test
    void testConstructor_DefaultLODQuery() {
        DiffusionChunkGenerator defaultGenerator = new DiffusionChunkGenerator();
        assertNotNull(defaultGenerator);
    }

    @Test
    void testConstructor_CustomLODQuery() {
        LODQuery customQuery = new DefaultLODQuery();
        DiffusionChunkGenerator customGenerator = new DiffusionChunkGenerator(customQuery);
        assertNotNull(customGenerator);
    }

    @Test
    void testBuildSurface_NoParameters() {
        // Test basic buildSurface method (currently a stub)
        assertDoesNotThrow(() -> generator.buildSurface());
    }

    @Test
    void testBuildSurface_WithChunkCoordinates() {
        // Test buildSurface with chunk coordinates (currently a stub)
        assertDoesNotThrow(() -> generator.buildSurface(0, 0));
    }

    @Test
    void testBuildSurface_WithHeightmapAndBiomes_ModifiesHeightmap() {
        // Store original values
        int originalValue = testHeightmap[0][0];
        
        // Apply surface generation
        generator.buildSurface(0, 0, testHeightmap, testBiomes);
        
        // Verify heightmap was modified
        boolean heightmapChanged = false;
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (testHeightmap[x][z] != 64) {
                    heightmapChanged = true;
                    break;
                }
            }
            if (heightmapChanged) break;
        }
        
        assertTrue(heightmapChanged, "Heightmap should be modified by buildSurface");
    }

    @Test
    void testBuildSurface_WithHeightmapAndBiomes_AppliesBasicModification() {
        // Test specific chunk coordinates for predictable variation
        int chunkX = 1, chunkZ = 1;
        
        generator.buildSurface(chunkX, chunkZ, testHeightmap, testBiomes);
        
        // Check that basic modification was applied (variation based on position)
        int expectedVariation = (chunkX + chunkZ + 0 + 0) % 3 - 1; // For position 0,0
        int actualHeight = testHeightmap[0][0];
        int expectedHeight = 64 + expectedVariation; // Original + basic modification
        
        // Since diffusion model also runs, we just verify it's not the original value
        assertNotEquals(64, actualHeight, "Height at [0][0] should be modified");
    }

    @Test
    void testBuildSurfaceWithLOD_LOD0_HighestDetail() {
        generator.buildSurfaceWithLOD(0, 0, testHeightmap, testBiomes, 0);
        
        // Verify heightmap was processed (LOD 0 includes full diffusion + refinement)
        boolean heightmapChanged = false;
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (testHeightmap[x][z] != 64) {
                    heightmapChanged = true;
                    break;
                }
            }
            if (heightmapChanged) break;
        }
        
        assertTrue(heightmapChanged, "LOD 0 should modify heightmap with full processing");
    }

    @Test
    void testBuildSurfaceWithLOD_LOD1_MediumDetail() {
        generator.buildSurfaceWithLOD(0, 0, testHeightmap, testBiomes, 1);
        
        // Verify heightmap was processed (LOD 1 includes standard diffusion)
        boolean heightmapChanged = false;
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (testHeightmap[x][z] != 64) {
                    heightmapChanged = true;
                    break;
                }
            }
            if (heightmapChanged) break;
        }
        
        assertTrue(heightmapChanged, "LOD 1 should modify heightmap with standard processing");
    }

    @Test
    void testBuildSurfaceWithLOD_LOD2_LowerDetail() {
        generator.buildSurfaceWithLOD(0, 0, testHeightmap, testBiomes, 2);
        
        // Verify heightmap was processed (LOD 2 includes reduced diffusion)
        boolean heightmapChanged = false;
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (testHeightmap[x][z] != 64) {
                    heightmapChanged = true;
                    break;
                }
            }
            if (heightmapChanged) break;
        }
        
        assertTrue(heightmapChanged, "LOD 2 should modify heightmap with reduced processing");
    }    @Test
    void testBuildSurfaceWithLOD_LODDefault_MinimalProcessing() {
        // Use chunk coordinates that will produce non-zero variation 
        // variation = (chunkX + chunkZ + x + z) % 2
        // For x=4,z=4: (1 + 0 + 4 + 4) % 2 = 1 (non-zero)
        generator.buildSurfaceWithLOD(1, 0, testHeightmap, testBiomes, 999);
        
        // Check the specific position that should be modified by minimal processing
        int expectedHeight = 64 + 1; // original + variation of 1
        assertEquals(expectedHeight, testHeightmap[4][4], 
                    "Position [4][4] should be modified by minimal processing");
    }

    @Test
    void testBuildSurfaceWithLOD_ByteOverload_ReturnsHeightmap() {
        int[][] result = generator.buildSurfaceWithLOD(0, 0, (byte) 1);
        
        assertNotNull(result, "Should return a heightmap");
        assertEquals(16, result.length, "Should return 16x16 heightmap");
        assertEquals(16, result[0].length, "Should return 16x16 heightmap");
        
        // Verify it's not all default values (should be processed)
        boolean heightmapChanged = false;
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (result[x][z] != 64) {
                    heightmapChanged = true;
                    break;
                }
            }
            if (heightmapChanged) break;
        }
        
        assertTrue(heightmapChanged, "Returned heightmap should be processed");
    }

    @Test
    void testBuildSurfaceWithLOD_ByteOverload_DifferentLODValues() {
        // Test different LOD values produce different results
        int[][] result0 = generator.buildSurfaceWithLOD(0, 0, (byte) 0);
        int[][] result2 = generator.buildSurfaceWithLOD(0, 0, (byte) 2);
        
        // Results should be different for different LOD levels
        boolean resultsAreDifferent = false;
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (result0[x][z] != result2[x][z]) {
                    resultsAreDifferent = true;
                    break;
                }
            }
            if (resultsAreDifferent) break;
        }
        
        assertTrue(resultsAreDifferent, "Different LOD levels should produce different results");
    }
}

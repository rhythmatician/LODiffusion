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
    }

    @Test
    void testBuildSurfaceWithLOD_HighLOD_MinimalProcessing() {
        // Use chunk coordinates that will produce non-zero variation 
        generator.buildSurfaceWithLOD(1, 0, testHeightmap, testBiomes, 999);
        
        // Check that minimal processing was applied
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
    }

    @Test
    void testGetChunkLOD() {
        // Test the getChunkLOD method
        int lod = generator.getChunkLOD(5, 10);
        assertTrue(lod >= 0, "LOD should be non-negative");
        assertTrue(lod <= 3, "LOD should be reasonable value");
    }

    @Test
    void testGetChunkLODRelativeToPlayer() {
        // Test LOD calculation relative to player position
        int lod = generator.getChunkLODRelativeToPlayer(10, 15, 5, 5);
        assertTrue(lod >= 0, "LOD should be non-negative");
        assertTrue(lod <= 3, "LOD should be reasonable value");
    }

    @Test
    void testBuildSurfaceWithSmartLOD() {
        // Test smart LOD surface building
        assertDoesNotThrow(() -> {
            generator.buildSurfaceWithSmartLOD(2, 3, testHeightmap, testBiomes);
        });
        
        // Verify heightmap was processed
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
        
        assertTrue(heightmapChanged, "Smart LOD should modify heightmap");
    }

    @Test
    void testBuildSurfaceWithLODManager_CoordinateBased() {
        // Test coordinate-based LOD manager integration
        assertDoesNotThrow(() -> {
            generator.buildSurfaceWithLODManager(5, 10, 0, 0, testHeightmap, testBiomes);
        });
        
        // Verify method completed successfully
        assertNotNull(testHeightmap, "Heightmap should remain valid");
    }

    @Test
    void testIsAdvancedLODAvailable() {
        // Test advanced LOD availability check
        boolean isAvailable = generator.isAdvancedLODAvailable();
        // This should return false in test environment without Distant Horizons
        assertFalse(isAvailable, "Advanced LOD should not be available in test environment");
    }

    @Test
    void testGetLODStrategyInfo() {
        // Test LOD strategy info retrieval
        String strategyInfo = generator.getLODStrategyInfo();
        assertNotNull(strategyInfo, "LOD strategy info should not be null");
        assertFalse(strategyInfo.isEmpty(), "LOD strategy info should not be empty");
    }
}

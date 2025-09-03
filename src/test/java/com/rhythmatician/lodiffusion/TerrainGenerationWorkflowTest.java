package com.rhythmatician.lodiffusion;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Integration test demonstrating the complete workflow from Minecraft data
 * to terrain generation using the helper methods.
 */
@Tag("ci")
public class TerrainGenerationWorkflowTest {

    @Test
    public void testCompleteWorkflowFromHeightData() throws IOException {
        // Given: Simulated Minecraft terrain data
        int[][] heightValues = {
            {64, 65, 66, 67, 66, 65, 64, 63},
            {65, 66, 67, 68, 67, 66, 65, 64},
            {66, 67, 68, 69, 68, 67, 66, 65},
            {67, 68, 69, 70, 69, 68, 67, 66},
            {66, 67, 68, 69, 68, 67, 66, 65},
            {65, 66, 67, 68, 67, 66, 65, 64},
            {64, 65, 66, 67, 66, 65, 64, 63},
            {63, 64, 65, 66, 65, 64, 63, 62}
        };
        
        int[][] biomeIds = {
            {1, 1, 1, 1, 1, 1, 1, 1}, // Plains
            {1, 1, 2, 2, 2, 2, 1, 1}, // Plains + Desert
            {1, 2, 2, 6, 6, 2, 2, 1}, // Desert + Forest
            {2, 2, 6, 6, 6, 6, 2, 2}, // Mixed biomes
            {2, 2, 6, 6, 6, 6, 2, 2},
            {1, 2, 2, 6, 6, 2, 2, 1},
            {1, 1, 2, 2, 2, 2, 1, 1},
            {1, 1, 1, 1, 1, 1, 1, 1}
        };
        
        int baseY = 60;
        float chunkX = 100.0f;
        float chunkZ = 200.0f;
        
        // When: Running complete terrain generation workflow
        try (OnnxTerrainGenerator generator = new OnnxTerrainGenerator()) {
            int[][][] generatedTerrain = generator.generateTerrainFromHeights(
                heightValues, biomeIds, baseY, chunkX, chunkZ
            );
            
            // Then: Should have generated valid 16x16x16 terrain
            assertNotNull(generatedTerrain, "Generated terrain should not be null");
            assertEquals(16, generatedTerrain.length, "Should be 16x16x16");
            assertEquals(16, generatedTerrain[0].length, "Should be 16x16x16");
            assertEquals(16, generatedTerrain[0][0].length, "Should be 16x16x16");
            
            // Verify terrain structure makes sense
            boolean hasAir = false;
            boolean hasSolid = false;
            
            for (int x = 0; x < 16; x++) {
                for (int y = 0; y < 16; y++) {
                    for (int z = 0; z < 16; z++) {
                        int blockId = generatedTerrain[x][y][z];
                        assertTrue(blockId >= 0 && blockId < 1104, 
                            "Block ID should be valid: " + blockId);
                        
                        if (blockId == 0) hasAir = true;
                        else hasSolid = true;
                    }
                }
            }
            
            assertTrue(hasAir, "Generated terrain should have air blocks");
            assertTrue(hasSolid, "Generated terrain should have solid blocks");
        }
    }
    
    @Test
    public void testCompleteWorkflowFromBlockData() throws IOException {
        // Given: Simulated parent chunk block data
        int[][][] parentBlocks = new int[8][8][8];
        
        // Create layered terrain: stone at bottom, dirt in middle, grass on top
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                for (int z = 0; z < 8; z++) {
                    if (y < 3) {
                        parentBlocks[x][y][z] = 1; // Stone
                    } else if (y < 5) {
                        parentBlocks[x][y][z] = 3; // Dirt
                    } else if (y < 6) {
                        parentBlocks[x][y][z] = 2; // Grass
                    } else {
                        parentBlocks[x][y][z] = 0; // Air
                    }
                }
            }
        }
        
        int[][] biomeIds = new int[8][8];
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                biomeIds[x][z] = 1; // Plains biome
            }
        }
        
        float chunkX = 50.0f;
        float chunkZ = 75.0f;
        
        // When: Running complete terrain generation workflow
        try (OnnxTerrainGenerator generator = new OnnxTerrainGenerator()) {
            int[][][] generatedTerrain = generator.generateCompleteTerrainFromBlocks(
                parentBlocks, biomeIds, chunkX, chunkZ
            );
            
            // Then: Should have generated valid terrain
            assertNotNull(generatedTerrain, "Generated terrain should not be null");
            assertEquals(16, generatedTerrain.length, "Should be 16x16x16");
            assertEquals(16, generatedTerrain[0].length, "Should be 16x16x16");
            assertEquals(16, generatedTerrain[0][0].length, "Should be 16x16x16");
            
            // Verify terrain has reasonable block distribution
            int airCount = 0;
            int solidCount = 0;
            
            for (int x = 0; x < 16; x++) {
                for (int y = 0; y < 16; y++) {
                    for (int z = 0; z < 16; z++) {
                        int blockId = generatedTerrain[x][y][z];
                        if (blockId == 0) airCount++;
                        else solidCount++;
                    }
                }
            }
            
            assertTrue(airCount > 0, "Should have air blocks");
            assertTrue(solidCount > 0, "Should have solid blocks");
            
            // Air should be more prevalent in upper layers
            int upperAirCount = 0;
            int lowerSolidCount = 0;
            
            for (int x = 0; x < 16; x++) {
                for (int z = 0; z < 16; z++) {
                    if (generatedTerrain[x][12][z] == 0) upperAirCount++;
                    if (generatedTerrain[x][3][z] != 0) lowerSolidCount++;
                }
            }
            
            assertTrue(upperAirCount > lowerSolidCount / 2, 
                "Upper layers should have more air than lower layers have solid");
        }
    }
    
    @Test
    public void testDataValidationIntegration() throws IOException {
        // Test that the validation catches invalid data in the workflow
        
        try (OnnxTerrainGenerator generator = new OnnxTerrainGenerator()) {
            // Test with invalid height values
            int[][] invalidHeights = new int[7][8]; // Wrong size
            
            assertThrows(IllegalArgumentException.class, () -> {
                OnnxTerrainGenerator.createParentHeightmapFromHeights(invalidHeights, 60);
            }, "Should reject invalid height array size");
            
            // Test with invalid biome IDs - this should work due to clamping
            int[][] validHeights = new int[8][8];
            int[][] extremeBiomes = new int[8][8];
            for (int x = 0; x < 8; x++) {
                for (int z = 0; z < 8; z++) {
                    validHeights[x][z] = 64;
                    extremeBiomes[x][z] = (x == 0 && z == 0) ? -100 : 1000; // Extreme values
                }
            }
            
            // Should not throw due to clamping
            assertDoesNotThrow(() -> {
                generator.generateTerrainFromHeights(validHeights, extremeBiomes, 60, 0, 0);
            }, "Should handle extreme biome IDs through clamping");
        }
    }
    
    @Test
    public void testHelperMethodsWithUniformBiome() throws IOException {
        // Test workflow with uniform biome data
        int[][] heights = new int[8][8];
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                heights[x][z] = 64 + (x + z) / 4; // Gentle slope
            }
        }
        
        try (OnnxTerrainGenerator generator = new OnnxTerrainGenerator()) {
            // Test with forest biome
            float[][][] biomeData = OnnxTerrainGenerator.createUniformBiomeData(6); // Forest
            float[][][] heightmap = OnnxTerrainGenerator.createParentHeightmapFromHeights(heights, 60);
            
            // Should not throw and validate correctly
            assertDoesNotThrow(() -> {
                OnnxTerrainGenerator.validateInputData(heightmap, biomeData, new float[]{0, 0});
            });
            
            OnnxTerrainGenerator.TerrainGenerationResult result = generator.generateTerrain(
                heightmap, biomeData, 0.0f, new float[]{0, 0}
            );
            
            assertNotNull(result);
            assertNotNull(result.blockLogits);
            assertNotNull(result.airMask);
            
            // Extract and apply processing
            int[][][] blocks = OnnxTerrainGenerator.extractBlockPredictions(result.blockLogits);
            int[][][] finalTerrain = OnnxTerrainGenerator.applyAirMask(blocks, result.airMask);
            
            assertNotNull(finalTerrain);
            assertEquals(16, finalTerrain.length);
        }
    }
}

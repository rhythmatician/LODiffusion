package com.rhythmatician.lodiffusion;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Test class for data preprocessing helper methods in OnnxTerrainGenerator.
 * Validates the helper methods used for converting Minecraft data to model inputs
 * and processing model outputs back to usable terrain data.
 */
@Tag("ci")
public class DataPreprocessingTest {

    @Test
    public void testCreateParentHeightmap() {
        // Given: A sample chunk with some blocks
        int[][][] chunkBlocks = new int[16][384][16];
        
        // Fill bottom layers with stone (block ID 1), upper with air (block ID 0)
        for (int x = 0; x < 16; x++) {
            for (int y = 0; y < 384; y++) {
                for (int z = 0; z < 16; z++) {
                    chunkBlocks[x][y][z] = (y < 64) ? 1 : 0; // Stone below y=64, air above
                }
            }
        }
        
        // When: Creating parent heightmap from chunk data
        float[][][] heightmap = OnnxTerrainGenerator.createParentHeightmap(chunkBlocks, 0, 60, 0);
        
        // Then: Should have correct shape and values
        assertEquals(8, heightmap.length);
        assertEquals(8, heightmap[0].length);
        assertEquals(8, heightmap[0][0].length);
        
        // Bottom should be solid (y=60-63), top should be air (y=64-67)
        assertEquals(1.0f, heightmap[0][0][0], "Bottom should be solid");
        assertEquals(1.0f, heightmap[0][3][0], "Bottom layers should be solid");
        assertEquals(0.0f, heightmap[0][4][0], "Top layers should be air");
        assertEquals(0.0f, heightmap[0][7][0], "Top should be air");
    }

    @Test
    public void testCreateParentHeightmapFromHeights() {
        // Given: Height values for an 8x8 area
        int[][] heightValues = {
            {64, 65, 66, 65, 64, 63, 64, 65},
            {65, 66, 67, 66, 65, 64, 65, 66},
            {66, 67, 68, 67, 66, 65, 66, 67},
            {65, 66, 67, 66, 65, 64, 65, 66},
            {64, 65, 66, 65, 64, 63, 64, 65},
            {63, 64, 65, 64, 63, 62, 63, 64},
            {64, 65, 66, 65, 64, 63, 64, 65},
            {65, 66, 67, 66, 65, 64, 65, 66}
        };
        int baseY = 60;
        
        // When: Creating heightmap from heights
        float[][][] heightmap = OnnxTerrainGenerator.createParentHeightmapFromHeights(heightValues, baseY);
        
        // Then: Should have correct shape and height-based values
        assertEquals(8, heightmap.length);
        assertEquals(8, heightmap[0].length);
        assertEquals(8, heightmap[0][0].length);
        
        // At position [0][0] (x=0, z=0), height is 64, baseY is 60
        // So y=0-4 (worldY 60-64) should be solid, y=5-7 (worldY 65-67) should be air
        assertEquals(1.0f, heightmap[0][0][0], "Below surface should be solid");
        assertEquals(1.0f, heightmap[0][4][0], "At surface should be solid");
        assertEquals(0.0f, heightmap[0][5][0], "Above surface should be air");
        assertEquals(0.0f, heightmap[0][7][0], "Above surface should be air");
    }

    @Test
    public void testCreateBiomeData() {
        // Given: Biome IDs for an 8x8 area
        int[][] biomeIds = new int[8][8];
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                biomeIds[x][z] = (x + z) % 10; // Varied biome pattern
            }
        }
        
        // When: Creating one-hot biome data
        float[][][] biomeData = OnnxTerrainGenerator.createBiomeData(biomeIds);
        
        // Then: Should have correct shape and one-hot encoding
        assertEquals(256, biomeData.length);
        assertEquals(8, biomeData[0].length);
        assertEquals(8, biomeData[0][0].length);
        
        // Verify one-hot encoding at a few positions
        int expectedBiome00 = biomeIds[0][0]; // Should be 0
        assertEquals(1.0f, biomeData[expectedBiome00][0][0], "Correct biome should be 1.0");
        for (int b = 0; b < 256; b++) {
            if (b != expectedBiome00) {
                assertEquals(0.0f, biomeData[b][0][0], "Other biomes should be 0.0");
            }
        }
        
        // Check position [1][0] - should be biome ID 1
        int expectedBiome10 = biomeIds[1][0]; // Should be 1
        assertEquals(1.0f, biomeData[expectedBiome10][1][0], "Correct biome should be 1.0");
    }

    @Test
    public void testCreateUniformBiomeData() {
        // Given: A single biome ID
        int biomeId = 5; // Forest biome
        
        // When: Creating uniform biome data
        float[][][] biomeData = OnnxTerrainGenerator.createUniformBiomeData(biomeId);
        
        // Then: Should have correct shape and uniform biome
        assertEquals(256, biomeData.length);
        assertEquals(8, biomeData[0].length);
        assertEquals(8, biomeData[0][0].length);
        
        // All positions should have the same biome
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                assertEquals(1.0f, biomeData[biomeId][x][z], "Target biome should be 1.0");
                for (int b = 0; b < 256; b++) {
                    if (b != biomeId) {
                        assertEquals(0.0f, biomeData[b][x][z], "Other biomes should be 0.0");
                    }
                }
            }
        }
    }

    @Test
    public void testExtractBlockPredictions() {
        // Given: Mock block logits with clear winners
        float[][][][] blockLogits = new float[1104][16][16][16];
        
        // Set up some clear predictions
        blockLogits[0][0][0][0] = 10.0f; // Air wins at [0][0][0]
        blockLogits[1][0][0][0] = 1.0f;
        blockLogits[2][0][0][0] = 2.0f;
        
        blockLogits[0][1][1][1] = 1.0f;
        blockLogits[1][1][1][1] = 15.0f; // Stone wins at [1][1][1]
        blockLogits[2][1][1][1] = 3.0f;
        
        blockLogits[0][2][2][2] = 2.0f;
        blockLogits[1][2][2][2] = 3.0f;
        blockLogits[2][2][2][2] = 20.0f; // Grass wins at [2][2][2]
        
        // When: Extracting block predictions
        int[][][] blocks = OnnxTerrainGenerator.extractBlockPredictions(blockLogits);
        
        // Then: Should pick the highest logit block at each position
        assertEquals(16, blocks.length);
        assertEquals(16, blocks[0].length);
        assertEquals(16, blocks[0][0].length);
        
        assertEquals(0, blocks[0][0][0], "Air should win at [0][0][0]");
        assertEquals(1, blocks[1][1][1], "Stone should win at [1][1][1]");
        assertEquals(2, blocks[2][2][2], "Grass should win at [2][2][2]");
    }

    @Test
    public void testApplyAirMask() {
        // Given: Block predictions and air mask
        int[][][] blocks = new int[16][16][16];
        float[][][][] airMask = new float[1][16][16][16];
        
        // Fill with stone blocks
        for (int x = 0; x < 16; x++) {
            for (int y = 0; y < 16; y++) {
                for (int z = 0; z < 16; z++) {
                    blocks[x][y][z] = 1; // Stone
                    airMask[0][x][y][z] = (y < 8) ? 1.0f : 0.0f; // Solid below y=8, air above
                }
            }
        }
        
        // When: Applying air mask
        int[][][] maskedBlocks = OnnxTerrainGenerator.applyAirMask(blocks, airMask);
        
        // Then: Should force air where mask indicates
        assertEquals(16, maskedBlocks.length);
        assertEquals(16, maskedBlocks[0].length);
        assertEquals(16, maskedBlocks[0][0].length);
        
        assertEquals(1, maskedBlocks[0][0][0], "Solid area should keep block");
        assertEquals(1, maskedBlocks[0][7][0], "Solid area should keep block");
        assertEquals(0, maskedBlocks[0][8][0], "Air area should be forced to air");
        assertEquals(0, maskedBlocks[0][15][0], "Air area should be forced to air");
    }

    @Test
    public void testValidateInputDataValid() {
        // Given: Valid input data
        float[][][] parentHeightmap = new float[8][8][8];
        float[][][] biomeData = new float[256][8][8];
        float[] chunkPos = {0.0f, 0.0f};
        
        // Fill with valid values
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                for (int z = 0; z < 8; z++) {
                    parentHeightmap[x][y][z] = (y < 4) ? 1.0f : 0.0f;
                }
            }
        }
        
        // One-hot biome encoding (plains biome = ID 1)
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                for (int b = 0; b < 256; b++) {
                    biomeData[b][x][z] = (b == 1) ? 1.0f : 0.0f;
                }
            }
        }
        
        // When/Then: Should not throw
        assertDoesNotThrow(() -> {
            OnnxTerrainGenerator.validateInputData(parentHeightmap, biomeData, chunkPos);
        });
    }

    @Test
    public void testValidateInputDataInvalidShapes() {
        // Given: Invalid shapes
        float[][][] wrongParent = new float[7][8][8]; // Wrong size
        float[][][] validBiome = new float[256][8][8];
        float[] validChunkPos = {0.0f, 0.0f};
        
        // When/Then: Should throw for wrong parent shape
        assertThrows(IllegalArgumentException.class, () -> {
            OnnxTerrainGenerator.validateInputData(wrongParent, validBiome, validChunkPos);
        });
        
        // Given: Invalid biome shape
        float[][][] validParent = new float[8][8][8];
        float[][][] wrongBiome = new float[255][8][8]; // Wrong size
        
        // When/Then: Should throw for wrong biome shape
        assertThrows(IllegalArgumentException.class, () -> {
            OnnxTerrainGenerator.validateInputData(validParent, wrongBiome, validChunkPos);
        });
    }

    @Test
    public void testValidateInputDataInvalidValues() {
        // Given: Valid shapes but invalid values
        float[][][] parentHeightmap = new float[8][8][8];
        float[][][] biomeData = new float[256][8][8];
        float[] chunkPos = {0.0f, 0.0f};
        
        // Invalid parent heightmap value
        parentHeightmap[0][0][0] = -1.0f; // Invalid negative value
        
        // When/Then: Should throw for invalid parent values
        assertThrows(IllegalArgumentException.class, () -> {
            OnnxTerrainGenerator.validateInputData(parentHeightmap, biomeData, chunkPos);
        });
        
        // Fix parent, break biome
        parentHeightmap[0][0][0] = 1.0f;
        biomeData[0][0][0] = 0.5f; // Invalid - not one-hot
        biomeData[1][0][0] = 0.5f; // Invalid - not one-hot
        
        // When/Then: Should throw for invalid biome encoding
        assertThrows(IllegalArgumentException.class, () -> {
            OnnxTerrainGenerator.validateInputData(parentHeightmap, biomeData, chunkPos);
        });
    }

    @Test
    public void testBiomeDataClampingEdgeCases() {
        // Test biome ID clamping for edge cases
        int[][] edgeCaseBiomes = {
            {-5, 0, 255, 300, 1, 2, 3, 4}, // Negative, min, max, over-max
            {1, 2, 3, 4, 5, 6, 7, 8},
            {5, 6, 7, 8, 9, 10, 11, 12},
            {9, 10, 11, 12, 13, 14, 15, 16},
            {13, 14, 15, 16, 17, 18, 19, 20},
            {17, 18, 19, 20, 21, 22, 23, 24},
            {21, 22, 23, 24, 25, 26, 27, 28},
            {25, 26, 27, 28, 29, 30, 31, 32}
        };
        
        float[][][] biomeData = OnnxTerrainGenerator.createBiomeData(edgeCaseBiomes);
        
        // Verify clamping worked
        assertEquals(1.0f, biomeData[0][0][0], "Negative biome should clamp to 0");
        assertEquals(1.0f, biomeData[0][0][1], "Zero biome should stay 0");
        assertEquals(1.0f, biomeData[255][0][2], "Max biome should stay 255");
        assertEquals(1.0f, biomeData[255][0][3], "Over-max biome should clamp to 255");
    }

    @Test
    public void testUniformBiomeClampingEdgeCases() {
        // Test edge cases for uniform biome creation
        float[][][] negativeData = OnnxTerrainGenerator.createUniformBiomeData(-5);
        float[][][] maxData = OnnxTerrainGenerator.createUniformBiomeData(300);
        
        // Verify clamping
        assertEquals(1.0f, negativeData[0][0][0], "Negative should clamp to 0");
        assertEquals(1.0f, maxData[255][0][0], "Over-max should clamp to 255");
    }
}

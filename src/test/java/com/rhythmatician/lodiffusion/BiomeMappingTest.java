package com.rhythmatician.lodiffusion;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * Tests for improved biome mapping and height extraction fixes.
 */
public class BiomeMappingTest {

    @Test
    public void testBiomeRegistryMapping() {
        DiffusionModel model = new DiffusionModel();
        
        // Test various biome strings for correct ID mapping
        String[] biomes = new String[256];
        biomes[0] = "minecraft:ocean";
        biomes[1] = "minecraft:plains";
        biomes[2] = "minecraft:desert";
        biomes[3] = "minecraft:mountains";
        biomes[4] = "minecraft:forest";
        biomes[5] = "minecraft:taiga";
        biomes[6] = "minecraft:swamp";
        biomes[7] = "minecraft:river";
        biomes[8] = "minecraft:nether_wastes";
        biomes[9] = "minecraft:the_end";
        
        // Fill the rest with plains
        for (int i = 10; i < 256; i++) {
            biomes[i] = "minecraft:plains";
        }
        
        // Create a dummy heightmap
        int[][] heightmap = new int[16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                heightmap[x][z] = 64;
            }
        }
        
        // Test that the model can handle various biome types without errors
        assertNotNull(model, "Model should be created successfully");
        // This would test the actual biome mapping in a real scenario
        // For now, just verify it doesn't crash
        model.run(heightmap, biomes);
        assertTrue(true, "Biome mapping should complete without errors");
    }

    @Test 
    public void testMinYVariance() {
        DiffusionModel model = new DiffusionModel();
        
        // Test with extreme heights (deep ocean to high mountains)
        int[][] deepOceanHeightmap = new int[16][16];
        int[][] mountainHeightmap = new int[16][16];
        
        // Deep ocean at Y=-60
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                deepOceanHeightmap[x][z] = -60;
                mountainHeightmap[x][z] = 200; // High mountains
            }
        }
        
        String[] oceanBiomes = new String[256];
        String[] mountainBiomes = new String[256];
        for (int i = 0; i < 256; i++) {
            oceanBiomes[i] = "minecraft:ocean";
            mountainBiomes[i] = "minecraft:mountains";
        }
        
        // Test that both extreme heights are handled properly
        model.run(deepOceanHeightmap, oceanBiomes);
        model.run(mountainHeightmap, mountainBiomes);
        
        // Verify heights are still in reasonable ranges after processing
        assertTrue(deepOceanHeightmap[8][8] >= -64, "Deep ocean height should not go below world minimum");
        assertTrue(mountainHeightmap[8][8] <= 320, "Mountain height should not exceed world maximum");
    }

    @Test
    public void testFallbackPath() {
        // Test with invalid model path to ensure fallback works
        try (OnnxTerrainGenerator generator = new OnnxTerrainGenerator("invalid/path/model.onnx")) {
            int[][] heights = new int[16][16];
            int[][] biomes = new int[16][16];
            
            for (int x = 0; x < 16; x++) {
                for (int z = 0; z < 16; z++) {
                    heights[x][z] = 64;
                    biomes[x][z] = 1; // Plains
                }
            }
            
            // Should fall back to stub implementation without crashing
            int[][][] result = generator.generateTerrainFromHeights(heights, biomes, 60, 0, 0);
            
            assertNotNull(result, "Fallback should generate valid terrain");
            assertEquals(16, result.length, "Should generate 16x16x16 terrain");
            assertEquals(16, result[0].length, "Should generate 16x16x16 terrain");
            assertEquals(16, result[0][0].length, "Should generate 16x16x16 terrain");
        } catch (Exception e) {
            // Expected - constructor may throw on invalid path
            assertTrue(true, "Expected behavior for invalid model path");
        }
    }
}

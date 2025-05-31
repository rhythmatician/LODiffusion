package com.rhythmatician.lodiffusion.dh;

import com.rhythmatician.lodiffusion.DiffusionModel;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

class StepRunDiffusionModelTest {

    private StepRunDiffusionModel step;

    @BeforeEach
    void setUp() {
        step = new StepRunDiffusionModel();
    }

    @Test
    void testGenerateGroup_ProcessesChunkList() {
        // Arrange
        List<Object> mockChunks = Arrays.asList("chunk1", "chunk2", "chunk3");

        // Act & Assert - should not throw any exceptions
        assertDoesNotThrow(() -> step.generateGroup(mockChunks));
    }

    @Test
    void testGenerateGroup_HandlesEmptyList() {
        // Arrange
        List<Object> emptyChunks = Arrays.asList();        // Act & Assert - should handle empty list gracefully
        assertDoesNotThrow(() -> step.generateGroup(emptyChunks));
    }

    @Test
    void testDiffusionModel_ModifiesHeightmap() {
        // Arrange - Create a 16x16 heightmap with values that will definitely change
        int[][] heightmap = new int[16][16];
        // Initialize with stepped terrain to ensure neighbor averaging creates changes
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                heightmap[x][z] = 64 + x * 2; // Clear height gradient
            }
        }
        String[] biomes = new String[16];
        for (int i = 0; i < 16; i++) {
            biomes[i] = "minecraft:mountains"; // Mountains have non-zero variation
        }        
        
        // Store original values for comparison - check multiple positions
        int originalPos1 = heightmap[2][2]; 
        int originalPos2 = heightmap[5][8];
        int originalPos3 = heightmap[10][5];
        
        // Act - Create DiffusionModel instance and run diffusion
        DiffusionModel model = new DiffusionModel();
        model.run(heightmap, biomes);

        // Assert - Check that at least one value has been modified
        boolean anyModified = (heightmap[2][2] != originalPos1) || 
                             (heightmap[5][8] != originalPos2) || 
                             (heightmap[10][5] != originalPos3);
          assertTrue(anyModified, 
            "Diffusion should modify heightmap values. Original: [" + 
            originalPos1 + "," + originalPos2 + "," + originalPos3 + 
            "], Modified: [" + heightmap[2][2] + "," + heightmap[5][8] + "," + heightmap[10][5] + "]");
    }

    @Test
    void testDiffusionModel_WithLOD_MultiChannel() {
        // Arrange - Create multi-channel data [channel][x][z] with larger array
        float[][][] channels = new float[3][5][5]; // 5x5 to ensure diffusion boundary logic works
        
        // Initialize height channel (channel 0) with more distinct values
        for (int x = 0; x < 5; x++) {
            for (int z = 0; z < 5; z++) {
                channels[0][x][z] = 10.0f + x * 10.0f + z * 2.0f; // More variation
            }
        }
        
        // Initialize biome channel (channel 1)
        for (int x = 0; x < 5; x++) {
            for (int z = 0; z < 5; z++) {
                channels[1][x][z] = 1.0f + x + z * 0.5f;
            }
        }
        
        // Initialize temperature channel (channel 2)
        for (int x = 0; x < 5; x++) {
            for (int z = 0; z < 5; z++) {
                channels[2][x][z] = 0.5f + x * 0.1f + z * 0.1f;
            }
        }
        
        String[] biomes = {"minecraft:plains", "minecraft:forest", "minecraft:mountain", "minecraft:desert", "minecraft:ocean"};
        int lod = 2;
        
        // Store original center values
        float originalHeightCenter = channels[0][2][2]; // Center of 5x5 array
          // Act
        DiffusionModel model = new DiffusionModel();
        model.runWithLOD(lod, channels, biomes);

        // Debug: Print actual values to understand the change
        System.out.println("Original center: " + originalHeightCenter);
        System.out.println("Modified center: " + channels[0][2][2]);
        System.out.println("Difference: " + Math.abs(originalHeightCenter - channels[0][2][2]));

        // Assert - Check that diffusion has modified the data with a more lenient tolerance
        // Since LOD 2 applies minimal processing (0.4 factor), we expect small changes
        assertTrue(Math.abs(originalHeightCenter - channels[0][2][2]) > 0.01f,
            "LOD diffusion should modify the height channel values by at least 0.01f, " +
            "original: " + originalHeightCenter + ", modified: " + channels[0][2][2]);
    }
}

package com.rhythmatician.lodiffusion;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Test class for ONNX model inference integration.
 * Tests DJL-based ONNX runtime for terrain generation.
 */
@Tag("inference")
class OnnxInferenceTest {

    private static final String MODEL_PATH = "artifacts/quick_test/model.onnx";
    
    @BeforeEach
    void setUp() {
        // Ensure model file exists
        Path modelPath = Paths.get(MODEL_PATH);
        assertTrue(modelPath.toFile().exists(), 
            "ONNX model file should exist at: " + MODEL_PATH);
    }

    @Test
    void testOnnxModelExists() {
        // Given: Model path
        Path modelPath = Paths.get(MODEL_PATH);
        
        // When/Then: Model file should exist and be readable
        assertTrue(modelPath.toFile().exists(), "Model file should exist");
        assertTrue(modelPath.toFile().canRead(), "Model file should be readable");
        assertTrue(modelPath.toFile().length() > 0, "Model file should not be empty");
    }

    @Test
    void testOnnxModelLoading() {
        // Given: Model path
        Path modelPath = Paths.get(MODEL_PATH);
        
        // When/Then: Should be able to load model without throwing exceptions
        assertDoesNotThrow(() -> {
            try (OnnxTerrainGenerator generator = new OnnxTerrainGenerator()) {
                assertTrue(generator.isAvailable(), "ONNX terrain generator should be available");
            }
        }, "Model loading should not throw exceptions");
    }

    @Test
    void testOnnxInferenceShapes() {
        // Test that we can run inference with correct input/output shapes
        // This validates the LODiffusion v1 contract:
        // - Input: x_parent [1,1,8,8,8], x_biome [1,256,8,8], timestep [1], chunk_pos [1,3]
        // - Output: block_logits [1,1104,16,16,16], air_mask [1,1,16,16,16]
        
        assertDoesNotThrow(() -> {
            try (OnnxTerrainGenerator generator = new OnnxTerrainGenerator()) {
                // Create test input data
                float[][][] parentHeightmap = new float[8][8][8];
                float[][][] biomeData = new float[256][8][8];
                float timestep = 0.5f;
                float[] chunkPos = {0.0f, 0.0f, 0.0f};
                
                // Initialize with test data
                for (int x = 0; x < 8; x++) {
                    for (int y = 0; y < 8; y++) {
                        for (int z = 0; z < 8; z++) {
                            parentHeightmap[x][y][z] = 64.0f; // Sea level
                        }
                    }
                }
                
                for (int c = 0; c < 256; c++) {
                    for (int x = 0; x < 8; x++) {
                        for (int z = 0; z < 8; z++) {
                            biomeData[c][x][z] = (c == 1) ? 1.0f : 0.0f; // Plains biome
                        }
                    }
                }
                
                // Run inference
                OnnxTerrainGenerator.TerrainGenerationResult result = 
                    generator.generateTerrain(parentHeightmap, biomeData, timestep, chunkPos);
                
                // Verify output shapes
                assertNotNull(result, "Result should not be null");
                assertNotNull(result.blockLogits, "Block logits should not be null");
                assertNotNull(result.airMask, "Air mask should not be null");
                
                // Verify block logits shape [1104][16][16][16]
                assertEquals(1104, result.blockLogits.length, "Block logits should have 1104 block types");
                assertEquals(16, result.blockLogits[0].length, "Block logits should have 16x16x16 output");
                assertEquals(16, result.blockLogits[0][0].length, "Block logits should have 16x16x16 output");
                assertEquals(16, result.blockLogits[0][0][0].length, "Block logits should have 16x16x16 output");
                
                // Verify air mask shape [1][16][16][16]
                assertEquals(1, result.airMask.length, "Air mask should have 1 channel");
                assertEquals(16, result.airMask[0].length, "Air mask should have 16x16x16 output");
                assertEquals(16, result.airMask[0][0].length, "Air mask should have 16x16x16 output");
                assertEquals(16, result.airMask[0][0][0].length, "Air mask should have 16x16x16 output");
            }
        }, "ONNX inference should complete without exceptions");
    }
}

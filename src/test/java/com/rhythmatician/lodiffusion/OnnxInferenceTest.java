package com.rhythmatician.lodiffusion;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Test class for ONNX model inference integration.
 * Tests the progressive LODiffusion models with 4-stage refinement.
 * Currently tests with fallback stubs while DJL integration is being completed.
 */
@Tag("inference")
class OnnxInferenceTest {

    private static final String PROGRESSIVE_MODEL_DIR = "onnx_export";
    private static final String[] PROGRESSIVE_MODELS = {
        "flexible_unet3d_lod4to3.onnx",
        "flexible_unet3d_lod3to2.onnx", 
        "flexible_unet3d_lod2to1.onnx",
        "flexible_unet3d_lod1to0.onnx"
    };
    
    @BeforeEach
    void setUp() {
        // Ensure at least one progressive model exists
        boolean hasAnyModel = false;
        for (String modelName : PROGRESSIVE_MODELS) {
            Path modelPath = Paths.get(PROGRESSIVE_MODEL_DIR, modelName);
            if (modelPath.toFile().exists()) {
                hasAnyModel = true;
                break;
            }
        }
        assertTrue(hasAnyModel, 
            "At least one progressive ONNX model file should exist in: " + PROGRESSIVE_MODEL_DIR);
    }

    @Test
    void testOnnxModelExists() {
        // Test that progressive models exist and are readable
        for (String modelName : PROGRESSIVE_MODELS) {
            Path modelPath = Paths.get(PROGRESSIVE_MODEL_DIR, modelName);
            
            // Skip test if model file doesn't exist (e.g., CI environment without Git LFS)
            assumeTrue(modelPath.toFile().exists(), 
                "Skipping ONNX model test - model file not found: " + modelName + 
                ". This may occur in CI environments without Git LFS.");
            
            // When/Then: Model file should exist and be readable
            assertTrue(modelPath.toFile().canRead(), "Model file should be readable: " + modelName);
            assertTrue(modelPath.toFile().length() > 0, "Model file should not be empty: " + modelName);
            
            // Should be a reasonable size for an ONNX model (> 100KB)
            assertTrue(modelPath.toFile().length() > 100 * 1024, 
                "Model file should be larger than 100KB: " + modelName + 
                " (actual: " + modelPath.toFile().length() + " bytes)");
        }
    }

    @Test
    void testOnnxModelLoading() {
        // Test that progressive models can be loaded
        boolean hasAnyModel = false;
        for (String modelName : PROGRESSIVE_MODELS) {
            Path modelPath = Paths.get(PROGRESSIVE_MODEL_DIR, modelName);
            if (modelPath.toFile().exists()) {
                hasAnyModel = true;
                break;
            }
        }
        
        // Skip test if no model files exist (e.g., CI environment without Git LFS)
        assumeTrue(hasAnyModel, 
            "Skipping ONNX model loading test - no progressive models found. " +
            "This may occur in CI environments without Git LFS. Expected in: " + PROGRESSIVE_MODEL_DIR);
        
        // When/Then: Should be able to load generator without throwing exceptions
        assertDoesNotThrow(() -> {
            try (OnnxTerrainGenerator generator = new OnnxTerrainGenerator()) {
                assertTrue(generator.isAvailable(), "ONNX terrain generator should be available");
                // Note: Progressive models don't have a single "model path" - they have multiple models
            }
        }, "Progressive model loading should not throw exceptions");
    }

    @Test
    void testOnnxInferenceShapes() {
        // Check if any progressive models exist
        boolean hasAnyModel = false;
        for (String modelName : PROGRESSIVE_MODELS) {
            Path modelPath = Paths.get(PROGRESSIVE_MODEL_DIR, modelName);
            if (modelPath.toFile().exists()) {
                hasAnyModel = true;
                break;
            }
        }
        
        // Skip test if no model files exist (e.g., CI environment without Git LFS)
        assumeTrue(hasAnyModel, 
            "Skipping ONNX inference shapes test - no progressive models found. " +
            "This may occur in CI environments without Git LFS. Expected in: " + PROGRESSIVE_MODEL_DIR);
            
        // Test that we can run progressive inference with correct input/output shapes
        // This validates the progressive LODiffusion contract:
        // - Progressive refinement: LOD4→LOD3→LOD2→LOD1→LOD0
        // - Final output: 16×16×16 voxel block
        
        assertDoesNotThrow(() -> {
            try (OnnxTerrainGenerator generator = new OnnxTerrainGenerator()) {
                // Create test input data for progressive generation
                // Progressive generation takes biome IDs and height values
                int[][] biomeIds = new int[16][16];
                int[][] heightValues = new int[16][16];
                
                // Fill with test data - plains biome and simple height pattern
                for (int x = 0; x < 16; x++) {
                    for (int z = 0; z < 16; z++) {
                        biomeIds[x][z] = 1; // Plains biome
                        heightValues[x][z] = 64; // Sea level
                    }
                }
                
                // Run progressive inference (currently using fallback stubs)
                int[][][] result = generator.generateProgressiveTerrain(biomeIds, heightValues, 0.0f, 0.0f);
                
                // Verify output shapes (result should be 16x16x16)
                assertNotNull(result, "Result should not be null");
                
                // Verify progressive output dimensions [16][16][16]
                assertEquals(16, result.length, "Progressive output should be 16×16×16");
                assertEquals(16, result[0].length, "Progressive output should be 16×16×16");
                assertEquals(16, result[0][0].length, "Progressive output should be 16×16×16");
                
                // Verify some basic properties of the generated terrain
                // The stub should generate some solid blocks
                boolean foundSolidBlock = false;
                boolean foundAirBlock = false;
                for (int x = 0; x < 16; x++) {
                    for (int y = 0; y < 16; y++) {
                        for (int z = 0; z < 16; z++) {
                            int blockType = result[x][y][z];
                            if (blockType > 0) {
                                foundSolidBlock = true;
                            } else {
                                foundAirBlock = true;
                            }
                        }
                    }
                }
                
                assertTrue(foundSolidBlock, "Should have some solid blocks in generated terrain");
                assertTrue(foundAirBlock, "Should have some air blocks in generated terrain");
            }
        }, "Progressive ONNX inference should complete without exceptions");
    }
    
    @Test
    void testContractCompliance() {
        // Test that the generator follows the progressive LODiffusion contract exactly
        assertDoesNotThrow(() -> {
            try (OnnxTerrainGenerator generator = new OnnxTerrainGenerator()) {
                // Create test input data
                int[][] biomeIds = new int[16][16];
                int[][] heightValues = new int[16][16];
                
                // Fill with minimal test data
                for (int x = 0; x < 16; x++) {
                    for (int z = 0; z < 16; z++) {
                        biomeIds[x][z] = 1; // Plains biome
                        heightValues[x][z] = 64; // Sea level
                    }
                }
                
                // Test progressive generation contract
                int[][][] result = generator.generateProgressiveTerrain(biomeIds, heightValues, 0.0f, 0.0f);
                
                // Contract compliance: progressive output should be 16×16×16
                assertNotNull(result, "Result should not be null");
                assertEquals(16, result.length, "Must be 16×16×16 output");
                assertEquals(16, result[0].length, "Must be 16×16×16 output");
                assertEquals(16, result[0][0].length, "Must be 16×16×16 output");
                
                // Contract compliance: value ranges (block IDs should be valid)
                for (int x = 0; x < 16; x++) {
                    for (int y = 0; y < 16; y++) {
                        for (int z = 0; z < 16; z++) {
                            int blockType = result[x][y][z];
                            assertTrue(blockType >= 0, 
                                "Block type must be non-negative (air=0, solid>0)");
                            assertTrue(blockType < 1000, 
                                "Block type must be reasonable (< 1000)");
                        }
                    }
                }
            }
        }, "Progressive contract compliance test should pass");
    }
}

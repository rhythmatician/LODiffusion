package com.rhythmatician.lodiffusion.onnx;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

import com.rhythmatician.lodiffusion.world.noise.NoiseTap;

import ai.djl.ndarray.NDManager;
import net.minecraft.world.Heightmap;

/**
 * End-to-end test for the complete 5-model progressive LOD pipeline
 * using the real ONNX models from artifacts/onnx_export_test/
 */
public class EndToEndPipelineTest {

    @Test
    void testCompleteProgressivePipeline() throws Exception {
        Path artifactsDir = Path.of("artifacts", "onnx_export_test");
        
        try (NDManager manager = NDManager.newBaseManager()) {
            // Load all 5 models
            ModelOrchestrator orchestrator = ModelOrchestrator.loadAll(manager,
                artifactsDir.resolve("model0initial.onnx"), 
                artifactsDir.resolve("model0initial.json"),
                artifactsDir.resolve("model1lod4to3.onnx"), 
                artifactsDir.resolve("model1lod4to3.json"),
                artifactsDir.resolve("model2lod3to2.onnx"), 
                artifactsDir.resolve("model2lod3to2.json"),
                artifactsDir.resolve("model3lod2to1.onnx"), 
                artifactsDir.resolve("model3lod2to1.json"),
                artifactsDir.resolve("model4lod1to0.onnx"), 
                artifactsDir.resolve("model4lod1to0.json")
            );
            
            // Create synthetic noise data
            NoiseTap.Cache cache = createSyntheticCache();
            
            // Run the complete pipeline to LOD0 (16³)
            ProgressiveLODPipeline.GenerationResult result = orchestrator.run(cache, 0);
            
            assertNotNull(result);
            assertNotNull(result.blockLogits());
            assertNotNull(result.airMask());
            
            // Debug: Print actual output shapes
            System.out.println("=== DEBUG OUTPUT SHAPES ===");
            System.out.println("blockLogits.length: " + result.blockLogits().length);
            if (result.blockLogits().length > 0) {
                System.out.println("blockLogits[0].length: " + result.blockLogits()[0].length);
                if (result.blockLogits()[0].length > 0) {
                    System.out.println("blockLogits[0][0].length: " + result.blockLogits()[0][0].length);
                    if (result.blockLogits()[0][0].length > 0) {
                        System.out.println("blockLogits[0][0][0].length: " + result.blockLogits()[0][0][0].length);
                        if (result.blockLogits()[0][0][0].length > 0) {
                            System.out.println("blockLogits[0][0][0][0].length: " + result.blockLogits()[0][0][0][0].length);
                        }
                    }
                }
            }
            
            // Verify final output shape is 16³
            assertEquals(1, result.blockLogits().length);
            assertEquals(1104, result.blockLogits()[0].length); // N_blocks (actual model output)
            assertEquals(16, result.blockLogits()[0][0].length);
            assertEquals(16, result.blockLogits()[0][0][0].length);
            assertEquals(16, result.blockLogits()[0][0][0][0].length);
            
            assertEquals(1, result.airMask().length);
            assertEquals(1, result.airMask()[0].length);
            assertEquals(16, result.airMask()[0][0].length);
            assertEquals(16, result.airMask()[0][0][0].length);
            assertEquals(16, result.airMask()[0][0][0][0].length);
            
            // Verify timing info
            assertTrue(result.totalTimeMs() > 0);
            assertEquals(5, result.modelTimesMs().length);
            assertEquals(5, result.resolutions().length);
            
            System.out.println("✅ Complete progressive pipeline test passed!");
            System.out.println("   Total time: " + result.totalTimeMs() + "ms");
            System.out.println("   Model times: " + java.util.Arrays.toString(result.modelTimesMs()));
        }
    }
    
    @Test
    void testPartialPipelineToLOD2() throws Exception {
        Path artifactsDir = Path.of("artifacts", "onnx_export_test");
        
        try (NDManager manager = NDManager.newBaseManager()) {
            ModelOrchestrator orchestrator = ModelOrchestrator.loadAll(manager,
                artifactsDir.resolve("model0initial.onnx"), 
                artifactsDir.resolve("model0initial.json"),
                artifactsDir.resolve("model1lod4to3.onnx"), 
                artifactsDir.resolve("model1lod4to3.json"),
                artifactsDir.resolve("model2lod3to2.onnx"), 
                artifactsDir.resolve("model2lod3to2.json"),
                artifactsDir.resolve("model3lod2to1.onnx"), 
                artifactsDir.resolve("model3lod2to1.json"),
                artifactsDir.resolve("model4lod1to0.onnx"), 
                artifactsDir.resolve("model4lod1to0.json")
            );
            
            NoiseTap.Cache cache = createSyntheticCache();
            
            // Run only to LOD2 (4³)
            ProgressiveLODPipeline.GenerationResult result = orchestrator.run(cache, 2);
            
            assertNotNull(result);
            
            // Verify output shape is 4³
            assertEquals(4, result.blockLogits()[0][0].length);
            assertEquals(4, result.blockLogits()[0][0][0].length);
            assertEquals(4, result.blockLogits()[0][0][0][0].length);
            
            System.out.println("✅ Partial pipeline to LOD2 test passed!");
        }
    }

    private static NoiseTap.Cache createSyntheticCache() {
        // Router maps with core fields
        Map<NoiseTap.RouterField, float[][][]> router = new EnumMap<>(NoiseTap.RouterField.class);
        
        for (NoiseTap.RouterField f : EnumSet.of(
                NoiseTap.RouterField.TEMPERATURE,
                NoiseTap.RouterField.VEGETATION,
                NoiseTap.RouterField.CONTINENTS,
                NoiseTap.RouterField.EROSION,
                NoiseTap.RouterField.DEPTH,
                NoiseTap.RouterField.RIDGES)) {
            router.put(f, createRouterField());
        }
        
        // NOTE: Intentionally NOT adding optional router fields (BARRIER, FLUID_FLOODEDNESS, etc.)
        // so that the input builder only creates the 5 core inputs expected by the ONNX models
        
        // Biomes 4×4×4
        int[][][] biomes4 = new int[4][4][4];
        for (int x = 0; x < 4; x++) {
            for (int z = 0; z < 4; z++) {
                for (int y = 0; y < 4; y++) {
                    biomes4[x][z][y] = (x * 16 + z * 4 + y) % 50; // Various biome IDs
                }
            }
        }
        
        // Heightmaps 16×16
        Map<Heightmap.Type, short[][]> heightmaps = new HashMap<>();
        heightmaps.put(Heightmap.Type.WORLD_SURFACE_WG, createHeightmap(80));
        heightmaps.put(Heightmap.Type.OCEAN_FLOOR_WG, createHeightmap(60));
        heightmaps.put(Heightmap.Type.MOTION_BLOCKING, createHeightmap(85));
        
        return new NoiseTap.Cache(router, biomes4, heightmaps, -64, 384, 100, 200, 12345L);
    }
    
    private static float[][][] createRouterField() {
        float[][][] field = new float[16][16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = 0; y < 16; y++) {
                    // Create some varied noise-like data
                    field[x][z][y] = (float) (Math.sin(x * 0.3) * Math.cos(z * 0.2) + y * 0.1 - 8.0);
                }
            }
        }
        return field;
    }
    
    private static short[][] createHeightmap(int baseHeight) {
        short[][] hm = new short[16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                // Create rolling hills
                hm[x][z] = (short) (baseHeight + 
                    Math.sin(x * 0.4) * 5 + 
                    Math.cos(z * 0.3) * 3 + 
                    Math.sin((x + z) * 0.2) * 2);
            }
        }
        return hm;
    }
}

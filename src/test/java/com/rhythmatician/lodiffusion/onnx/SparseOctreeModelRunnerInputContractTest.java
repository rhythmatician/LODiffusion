package com.rhythmatician.lodiffusion.onnx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.lang.reflect.Constructor;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.Translator;

class SparseOctreeModelRunnerInputContractTest {

    static {
        // Mockito uses Byte Buddy under the hood for inline mocking. On Java 21+,
        // Byte Buddy requires this experimental flag to allow retransforming classes.
        // (See: "Java 21 (65) is not supported by the current version of Byte Buddy" error.)
        System.setProperty("net.bytebuddy.experimental", "true");
    }

    @Test
    void resolveInputOrder_prefersNamedOrderFromConfig() {
        Map<String, int[]> inputs = new LinkedHashMap<>();
        inputs.put("noise_2d", new int[] {1, 6, 4, 4});
        inputs.put("biome_ids", new int[] {1, 4, 2, 4});
        inputs.put("noise_3d", new int[] {1, 13, 4, 2, 4});
        ModelConfig cfg = new ModelConfig(
                "sparse_octree", "1", inputs, null, Map.of("block_logits", new int[] {1, 16, 16, 16, 16}),
                null, null, null, null, null, 256, null, null, null);

        List<String> order = SparseOctreeModelRunner.resolveInputOrder(cfg);

        assertEquals(List.of("noise_2d", "biome_ids", "noise_3d"), order);
    }

    @Test
    @SuppressWarnings({"unchecked"})
    void runInferenceWithBiome_executesTwoInputModel() throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            Predictor<NDList, NDList> predictor = mock(Predictor.class);
            ZooModel<NDList, NDList> model = makeZooModel(predictor);

            // Capture input shapes before the NDManager closes the arrays.
            final long[][] capturedShapes = new long[2][];
            when(predictor.predict(any())).thenAnswer(invocation -> {
                NDList in = invocation.getArgument(0);
                capturedShapes[0] = in.get(0).getShape().getShape().clone();
                capturedShapes[1] = in.get(1).getShape().getShape().clone();
                return new NDList(manager.create(new float[] {0f, 1f}, new Shape(1, 1, 2)));
            });

            SparseOctreeModelRunner runner = newRunner(manager, model,
                    true, new long[] {1, 6, 4, 4},
                    false, null,
                    new long[] {1, 13, 4, 2, 4},
                    List.of("noise_2d", "noise_3d"));
            try {
                int[][][] blocks = runner.runInferenceWithBiome(new float[13 * 4 * 2 * 4], null, null, null);
                assertNotNull(blocks);

                assertArrayEquals(new long[] {1, 6, 4, 4}, capturedShapes[0]);
                assertArrayEquals(new long[] {1, 13, 4, 2, 4}, capturedShapes[1]);
            } finally {
                runner.close();
            }
        }
    }

    @Test
    @SuppressWarnings({"unchecked"})
    void runInferenceWithBiome_executesThreeInputModelAndFallsBackForInvalidBiome() throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            Predictor<NDList, NDList> predictor = mock(Predictor.class);
            ZooModel<NDList, NDList> model = makeZooModel(predictor);

            // Capture input shapes before the NDManager closes the arrays.
            final long[][] capturedShapes = new long[3][];
            final long[][] capturedBiome = new long[1][];
            when(predictor.predict(any())).thenAnswer(invocation -> {
                NDList in = invocation.getArgument(0);
                capturedShapes[0] = in.get(0).getShape().getShape().clone();
                capturedShapes[1] = in.get(1).getShape().getShape().clone();
                capturedShapes[2] = in.get(2).getShape().getShape().clone();
                // Capture biome tensor as a primitive long array copy before it is released
                // (ONNX model expects biome_ids as int64 / torch.long)
                capturedBiome[0] = in.get(1).toLongArray();
                return new NDList(manager.create(new float[] {0f, 1f}, new Shape(1, 1, 2)));
            });

            SparseOctreeModelRunner runner = newRunner(manager, model,
                    true, new long[] {1, 6, 4, 4},
                    true, new long[] {1, 4, 2, 4},
                    new long[] {1, 13, 4, 2, 4},
                    List.of("noise_2d", "biome_ids", "noise_3d"));

            int[][][] invalidBiome = new int[4][2][4];
            invalidBiome[0][0][0] = -1; // invalid value should trigger zeros fallback
            try {
                int[][][] blocks = runner.runInferenceWithBiome(new float[13 * 4 * 2 * 4], invalidBiome, null, null);
                assertNotNull(blocks);

                assertEquals(3, capturedShapes.length);
                assertArrayEquals(new long[] {1, 6, 4, 4}, capturedShapes[0]);
                assertArrayEquals(new long[] {1, 4, 2, 4}, capturedShapes[1]);
                assertArrayEquals(new long[] {1, 13, 4, 2, 4}, capturedShapes[2]);
                assertArrayEquals(new long[4 * 2 * 4], capturedBiome[0]);
            } finally {
                runner.close();
            }
        }
    }

    @Test
    void resolveInputOrder_includesHeightmapInputs() {
        Map<String, int[]> inputs = new LinkedHashMap<>();
        inputs.put("noise_3d", new int[] {1, 15, 4, 2, 4});
        inputs.put("biome_ids", new int[] {1, 4, 2, 4});
        inputs.put("heightmap_surface", new int[] {1, 16, 16});
        inputs.put("heightmap_ocean_floor", new int[] {1, 16, 16});
        ModelConfig cfg = new ModelConfig(
                "sparse_octree", "1", inputs, null, Map.of("block_logits", new int[] {1, 16, 16, 16, 16}),
                null, null, null, null, null, 256, null, null, null);

        List<String> order = SparseOctreeModelRunner.resolveInputOrder(cfg);

        assertEquals(List.of("noise_3d", "biome_ids", "heightmap_surface", "heightmap_ocean_floor"), order);
    }

    @Test
    @SuppressWarnings({"unchecked"})
    void runInferenceWithBiome_passesFiveInputsIncludingHeightmaps() throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            Predictor<NDList, NDList> predictor = mock(Predictor.class);
            ZooModel<NDList, NDList> model = makeZooModel(predictor);

            // Capture all 5 input shapes
            final long[][] capturedShapes = new long[5][];
            when(predictor.predict(any())).thenAnswer(invocation -> {
                NDList in = invocation.getArgument(0);
                for (int i = 0; i < in.size(); i++) {
                    capturedShapes[i] = in.get(i).getShape().getShape().clone();
                }
                return new NDList(manager.create(new float[] {0f, 1f}, new Shape(1, 1, 2)));
            });

            SparseOctreeModelRunner runner = newRunner(manager, model,
                    true, new long[] {1, 6, 4, 4},
                    true, new long[] {1, 4, 2, 4},
                    true, new long[] {1, 16, 16},
                    true, new long[] {1, 16, 16},
                    new long[] {1, 15, 4, 2, 4},
                    List.of("noise_2d", "noise_3d", "biome_ids",
                            "heightmap_surface", "heightmap_ocean_floor"));

            float[][] hmSurface = new float[16][16];
            hmSurface[0][0] = 64.0f;
            float[][] hmOcean = new float[16][16];
            hmOcean[0][0] = 32.0f;
            int[][][] biome = new int[4][2][4];
            try {
                int[][][] blocks = runner.runInferenceWithBiome(
                        new float[15 * 4 * 2 * 4], biome, hmSurface, hmOcean);
                assertNotNull(blocks);

                // Verify all 5 inputs were passed in expected order and shape
                assertArrayEquals(new long[] {1, 6, 4, 4}, capturedShapes[0], "noise_2d shape");
                assertArrayEquals(new long[] {1, 15, 4, 2, 4}, capturedShapes[1], "noise_3d shape");
                assertArrayEquals(new long[] {1, 4, 2, 4}, capturedShapes[2], "biome_ids shape");
                assertArrayEquals(new long[] {1, 16, 16}, capturedShapes[3], "heightmap_surface shape");
                assertArrayEquals(new long[] {1, 16, 16}, capturedShapes[4], "heightmap_ocean_floor shape");
            } finally {
                runner.close();
            }
        }
    }

    private static ZooModel<NDList, NDList> makeZooModel(Predictor<NDList, NDList> predictor) {
        Model base = Model.newInstance("test");
        Translator<NDList, NDList> translator = new NoopTranslator();
        return new ZooModel<>(base, translator) {
            @Override
            public Predictor<NDList, NDList> newPredictor() {
                return predictor;
            }

            @Override
            public void close() {
                // no-op
            }
        };
    }

    private static SparseOctreeModelRunner newRunner(
            NDManager manager,
            ZooModel<NDList, NDList> model,
            boolean hasNoise2d,
            long[] noise2dShape,
            boolean hasBiomeIds,
            long[] biomeShape,
            long[] noise3dShape,
            List<String> inputOrder) throws Exception {
        return newRunner(manager, model, hasNoise2d, noise2dShape,
                hasBiomeIds, biomeShape,
                false, null, false, null,
                noise3dShape, inputOrder);
    }

    private static SparseOctreeModelRunner newRunner(
            NDManager manager,
            ZooModel<NDList, NDList> model,
            boolean hasNoise2d,
            long[] noise2dShape,
            boolean hasBiomeIds,
            long[] biomeShape,
            boolean hasHeightmapSurface,
            long[] heightmapSurfaceShape,
            boolean hasHeightmapOceanFloor,
            long[] heightmapOceanFloorShape,
            long[] noise3dShape,
            List<String> inputOrder) throws Exception {
        Constructor<SparseOctreeModelRunner> ctor = SparseOctreeModelRunner.class.getDeclaredConstructor(
                NDManager.class,
                ZooModel.class,
                BlockVocabulary.class,
                int.class,
                float.class,
                boolean.class,
                long[].class,
                boolean.class,
                long[].class,
                boolean.class,
                long[].class,
                boolean.class,
                long[].class,
                long[].class,
                List.class);
        ctor.setAccessible(true);
        return ctor.newInstance(manager, model, null, 2, 0.6f,
                hasNoise2d, noise2dShape, hasBiomeIds, biomeShape,
                hasHeightmapSurface, heightmapSurfaceShape,
                hasHeightmapOceanFloor, heightmapOceanFloorShape,
                noise3dShape, inputOrder);
    }
}

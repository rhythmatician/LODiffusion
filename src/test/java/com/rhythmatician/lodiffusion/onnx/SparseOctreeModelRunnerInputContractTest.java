package com.rhythmatician.lodiffusion.onnx;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.lang.reflect.Constructor;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
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
                null, null, null, null, null, 256, null, null);

        List<String> order = SparseOctreeModelRunner.resolveInputOrder(cfg);

        assertEquals(List.of("noise_2d", "biome_ids", "noise_3d"), order);
    }

    @Test
    @SuppressWarnings({"unchecked"})
    void runInferenceWithBiome_executesTwoInputModel() throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            Predictor<NDList, NDList> predictor = mock(Predictor.class);
            ZooModel<NDList, NDList> model = makeZooModel(predictor);
            when(predictor.predict(any())).thenReturn(new NDList(
                    manager.create(new float[] {0f, 1f}, new Shape(1, 1, 2))));

            SparseOctreeModelRunner runner = newRunner(manager, model,
                    true, new long[] {1, 6, 4, 4},
                    false, null,
                    List.of("noise_2d", "noise_3d"));
            try {
                int[][][] blocks = runner.runInferenceWithBiome(new float[13 * 4 * 2 * 4], null);
                assertNotNull(blocks);
            } finally {
                runner.close();
            }

            ArgumentCaptor<NDList> captor = ArgumentCaptor.forClass(NDList.class);
            verify(predictor).predict(captor.capture());
            NDList usedInputs = captor.getValue();
            assertEquals(2, usedInputs.size());
            assertArrayEquals(new long[] {1, 6, 4, 4}, usedInputs.get(0).getShape().getShape());
            assertArrayEquals(new long[] {1, 13, 4, 2, 4}, usedInputs.get(1).getShape().getShape());
        }
    }

    @Test
    @SuppressWarnings({"unchecked"})
    void runInferenceWithBiome_executesThreeInputModelAndFallsBackForInvalidBiome() throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            Predictor<NDList, NDList> predictor = mock(Predictor.class);
            ZooModel<NDList, NDList> model = makeZooModel(predictor);
            when(predictor.predict(any())).thenReturn(new NDList(
                    manager.create(new float[] {0f, 1f}, new Shape(1, 1, 2))));

            SparseOctreeModelRunner runner = newRunner(manager, model,
                    true, new long[] {1, 6, 4, 4},
                    true, new long[] {1, 4, 2, 4},
                    List.of("noise_2d", "biome_ids", "noise_3d"));

            int[][][] invalidBiome = new int[4][2][4];
            invalidBiome[0][0][0] = -1; // invalid value should trigger zeros fallback
            try {
                int[][][] blocks = runner.runInferenceWithBiome(new float[13 * 4 * 2 * 4], invalidBiome);
                assertNotNull(blocks);
            } finally {
                runner.close();
            }

            ArgumentCaptor<NDList> captor = ArgumentCaptor.forClass(NDList.class);
            verify(predictor).predict(captor.capture());
            NDList usedInputs = captor.getValue();
            assertEquals(3, usedInputs.size());
            NDArray biomeTensor = usedInputs.get(1);
            assertArrayEquals(new long[] {1, 4, 2, 4}, biomeTensor.getShape().getShape());
            assertArrayEquals(new int[4 * 2 * 4], biomeTensor.toIntArray());
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
                List.class);
        ctor.setAccessible(true);
        return ctor.newInstance(manager, model, null, 2, 0.6f,
                hasNoise2d, noise2dShape, hasBiomeIds, biomeShape, inputOrder);
    }
}

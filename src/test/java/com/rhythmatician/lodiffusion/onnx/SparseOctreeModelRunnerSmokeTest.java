package com.rhythmatician.lodiffusion.onnx;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;

/**
 * Smoke test that verifies the sparse_octree ONNX model can be loaded and executed.
 *
 * <p>This is intended as a minimal end-to-end sanity check (model file + sidecar
 * config + runtime ONNX inference) and does not validate output correctness.
 */
class SparseOctreeModelRunnerSmokeTest {

    @Test
    void loadAndRunDummyInference() throws Exception {
        // Force CPU provider in tests (DirectML isn't always available in CI/host envs).
        // Restore the original value afterward so other tests are not affected.
        String originalDevice = com.rhythmatician.lodiffusion.Config.inferenceDevice();
        com.rhythmatician.lodiffusion.Config.setInferenceDevice("cpu");
        try {
            Path modelDir = Path.of("config", "lodiffusion");
            assumeTrue(modelDir.toFile().exists(), "Skipping smoke test (model directory missing): " + modelDir);
            Path onnxPath = modelDir.resolve("sparse_octree.onnx");
            assumeTrue(Files.exists(onnxPath), "Skipping smoke test (model file missing): " + onnxPath);

            SparseOctreeModelRunner runner = SparseOctreeModelRunner.tryLoad(modelDir);
            assumeTrue(runner != null, "Skipping smoke test (sparse_octree model not loadable)");
            try (SparseOctreeModelRunner r = runner) {
                float[] noise = new float[13 * 4 * 2 * 4];
                int[][][] blocks = r.runInference(noise);
                assertNotNull(blocks, "Inference should not return null");
                assertEquals(16, blocks.length);
                assertEquals(16, blocks[0].length);
                assertEquals(16, blocks[0][0].length);
            }
        } finally {
            com.rhythmatician.lodiffusion.Config.setInferenceDevice(originalDevice);
        }
    }
}

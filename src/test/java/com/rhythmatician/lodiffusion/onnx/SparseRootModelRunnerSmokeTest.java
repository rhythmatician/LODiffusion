package com.rhythmatician.lodiffusion.onnx;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Path;

import org.junit.jupiter.api.Test;

/**
 * Smoke test that verifies the sparse_root ONNX model can be loaded and executed.
 *
 * <p>This is intended as a minimal end-to-end sanity check (model file + sidecar
 * config + runtime ONNX inference) and does not validate output correctness.
 */
class SparseRootModelRunnerSmokeTest {

    @Test
    void loadAndRunDummyInference() throws Exception {
        // Force CPU provider in tests (DirectML isn't always available in CI/host envs).
        // Restore the original value afterward so other tests are not affected.
        String originalDevice = com.rhythmatician.lodiffusion.Config.inferenceDevice();
        com.rhythmatician.lodiffusion.Config.setInferenceDevice("cpu");
        try {
            Path modelDir = Path.of("config", "lodiffusion");
            assertTrue(modelDir.toFile().exists(), "Expected model directory to exist: " + modelDir);

            SparseRootModelRunner runner = SparseRootModelRunner.tryLoad(modelDir);
            assertNotNull(runner, "Expected SparseRootModelRunner to load successfully");
            try {
                float[] noise = new float[13 * 4 * 2 * 4];
                int[][][] blocks = runner.runInference(noise);
                assertNotNull(blocks, "Inference should not return null");
                assertEquals(16, blocks.length);
                assertEquals(16, blocks[0].length);
                assertEquals(16, blocks[0][0].length);
            } finally {
                runner.close();
            }
        } finally {
            com.rhythmatician.lodiffusion.Config.setInferenceDevice(originalDevice);
        }
    }
}

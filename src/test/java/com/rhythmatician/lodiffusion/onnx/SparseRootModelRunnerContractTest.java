package com.rhythmatician.lodiffusion.onnx;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/**
 * Contract tests for sparse-root split expansion semantics.
 */
class SparseRootModelRunnerContractTest {

    @Test
    void splitExpansion_boundaryRespectsStrictGreaterThanThreshold() {
        float threshold = (float) com.rhythmatician.lodiffusion.Config.getDouble(
                SparseRootModelRunner.SPLIT_THRESHOLD_CONFIG_KEY,
                SparseRootModelRunner.DEFAULT_SPLIT_THRESHOLD);
        assertTrue(threshold > 0.0f && threshold < 1.0f,
                "sparseRootSplitThreshold must be in (0,1)");

        float boundaryLogit = SparseRootModelRunner.logitForThreshold(threshold);

        assertFalse(SparseRootModelRunner.shouldExpandNode(boundaryLogit, threshold),
                "Node should not expand when sigmoid(logit) equals threshold exactly");
        assertFalse(SparseRootModelRunner.shouldExpandNode(boundaryLogit - 0.01f, threshold),
                "Node should not expand below threshold");
        assertTrue(SparseRootModelRunner.shouldExpandNode(boundaryLogit + 0.01f, threshold),
                "Node should expand above threshold");
    }
}

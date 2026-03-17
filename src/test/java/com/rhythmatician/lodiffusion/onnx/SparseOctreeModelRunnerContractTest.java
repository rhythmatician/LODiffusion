package com.rhythmatician.lodiffusion.onnx;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/**
 * Contract tests for sparse-root split expansion semantics.
 */
class SparseOctreeModelRunnerContractTest {

    @Test
    void splitExpansion_boundaryRespectsStrictGreaterThanThreshold() {
        float threshold = (float) com.rhythmatician.lodiffusion.Config.getDouble(
                SparseOctreeModelRunner.SPLIT_THRESHOLD_CONFIG_KEY,
                SparseOctreeModelRunner.DEFAULT_SPLIT_THRESHOLD);
        assertTrue(threshold > 0.0f && threshold < 1.0f,
                "sparseRootSplitThreshold must be in (0,1)");

        float boundaryLogit = SparseOctreeModelRunner.logitForThreshold(threshold);

        assertFalse(SparseOctreeModelRunner.shouldExpandNode(boundaryLogit, threshold),
                "Node should not expand when sigmoid(logit) equals threshold exactly");
        assertFalse(SparseOctreeModelRunner.shouldExpandNode(boundaryLogit - 0.01f, threshold),
                "Node should not expand below threshold");
        assertTrue(SparseOctreeModelRunner.shouldExpandNode(boundaryLogit + 0.01f, threshold),
                "Node should expand above threshold");
    }
}

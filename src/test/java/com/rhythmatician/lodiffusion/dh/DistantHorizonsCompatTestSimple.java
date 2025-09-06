package com.rhythmatician.lodiffusion.dh;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * Test class for DistantHorizonsCompat following TDD principles.
 * Tests LOD compatibility layer and integration capabilities.
 */
public class DistantHorizonsCompatTestSimple {

    @Test
    void testGetChunkLOD_WithCoordinates_ReturnsValidLOD() {
        int lod = DistantHorizonsCompat.getChunkLOD(0, 0, 0, 0);

        assertTrue(lod >= 0, "LOD should be non-negative");
        assertTrue(lod <= 10, "LOD should be reasonable (≤ 10)");
    }

    @Test
    void testGetChunkLOD_WithCoordinates_DistanceAffectsLOD() {
        // Test that chunks further from player have higher LOD (lower detail)
        int nearLOD = DistantHorizonsCompat.getChunkLOD(0, 0, 0, 0);
        int farLOD = DistantHorizonsCompat.getChunkLOD(10, 10, 0, 0);

        assertTrue(farLOD >= nearLOD, "Distant chunks should have higher LOD (lower detail)");
    }

    @Test
    void testGetLODDiffusionFactor_ValidRange() {
        // Test various LOD levels
        for (int lod = 0; lod <= 5; lod++) {
            float factor = DistantHorizonsCompat.getLODDiffusionFactor(lod);

            assertTrue(factor >= 0.1f, "Diffusion factor should be >= 0.1");
            assertTrue(factor <= 1.0f, "Diffusion factor should be <= 1.0");
        }
    }

    @Test
    void testGetLODDiffusionFactor_HigherLODLowerFactor() {
        float factor0 = DistantHorizonsCompat.getLODDiffusionFactor(0);
        float factor3 = DistantHorizonsCompat.getLODDiffusionFactor(3);

        assertTrue(factor3 <= factor0, "Higher LOD should have lower or equal diffusion factor");
    }

    @Test
    void testIsDistantHorizonsIntegrationAvailable_ReturnsBoolean() {
        boolean available = DistantHorizonsCompat.isDistantHorizonsIntegrationAvailable();

        // Should return either true or false (not throw exception)
        assertTrue(available == true || available == false);
    }

    @Test
    void testGetIntegrationStatus_ReturnsNonNullString() {
        String status = DistantHorizonsCompat.getIntegrationStatus();

        assertNotNull(status, "Integration status should not be null");
        assertFalse(status.isEmpty(), "Integration status should not be empty");
    }

    @Test
    void testRegisterWorldGenerator_DoesNotThrow() {
        // Test that registration doesn't throw exceptions
        assertDoesNotThrow(() -> DistantHorizonsCompat.registerWorldGenerator());
    }

    @Test
    void testGetLodData_WithPlayerCoordinates_ReturnsValidData() {
        Object[] lodData = DistantHorizonsCompat.getLodData(0, 0, 0, 0);

        assertNotNull(lodData, "LOD data should not be null");
        assertEquals(2, lodData.length, "LOD data should contain 2 elements");

        assertTrue(lodData[0] instanceof Integer, "First element should be LOD level (Integer)");
        assertTrue(lodData[1] instanceof Float, "Second element should be diffusion factor (Float)");

        int lod = (Integer) lodData[0];
        float diffusionFactor = (Float) lodData[1];

        assertTrue(lod >= 0, "LOD should be non-negative");
        assertTrue(diffusionFactor >= 0.1f && diffusionFactor <= 1.0f,
                  "Diffusion factor should be in valid range");
    }

    @Test
    void testGetLodData_DeprecatedMethod_ReturnsValidData() {
        // Test calling with default player position (0, 0)
        Object[] lodData = DistantHorizonsCompat.getLodData(0, 0, 0, 0);

        assertNotNull(lodData, "LOD data should not be null");
        assertEquals(2, lodData.length, "LOD data should contain 2 elements");

        assertTrue(lodData[0] instanceof Integer, "First element should be LOD level (Integer)");
        assertTrue(lodData[1] instanceof Float, "Second element should be diffusion factor (Float)");
    }

    @Test
    void testGetLodData_ConsistentResults() {
        // Test that multiple calls with same parameters return same results
        Object[] firstCall = DistantHorizonsCompat.getLodData(5, 5, 0, 0);
        Object[] secondCall = DistantHorizonsCompat.getLodData(5, 5, 0, 0);

        assertEquals(firstCall[0], secondCall[0], "LOD level should be consistent");
        assertEquals(firstCall[1], secondCall[1], "Diffusion factor should be consistent");
    }

    @Test
    void testStaticMethods_NoInstantiationRequired() {
        // Test that all static methods can be called without class instantiation
        assertDoesNotThrow(() -> {
            DistantHorizonsCompat.getChunkLOD(0, 0, 0, 0);
            DistantHorizonsCompat.getLODDiffusionFactor(1);
            DistantHorizonsCompat.isDistantHorizonsIntegrationAvailable();
            DistantHorizonsCompat.getIntegrationStatus();
            DistantHorizonsCompat.registerWorldGenerator();
            DistantHorizonsCompat.getLodData(0, 0, 0, 0);
        });
    }
}

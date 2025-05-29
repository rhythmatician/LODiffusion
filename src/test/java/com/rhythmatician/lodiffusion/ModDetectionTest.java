package com.rhythmatician.lodiffusion;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for ModDetection utility following TDD principles.
 * Tests mod detection capabilities and LOD strategy information.
 */
public class ModDetectionTest {

    @Test
    void testIsDistantHorizonsAvailable_ReturnsBoolean() {
        // Test that the method returns a boolean value (doesn't throw exception)
        assertDoesNotThrow(() -> {
            boolean result = ModDetection.isDistantHorizonsAvailable();
            // Result should be either true or false (not null)
            assertTrue(result == true || result == false);
        });
    }

    @Test
    void testGetLODStrategyInfo_ReturnsNonNullString() {
        String strategyInfo = ModDetection.getLODStrategyInfo();
        
        assertNotNull(strategyInfo, "LOD strategy info should not be null");
        assertFalse(strategyInfo.isEmpty(), "LOD strategy info should not be empty");
    }

    @Test
    void testGetLODStrategyInfo_ReturnsExpectedMessages() {
        String strategyInfo = ModDetection.getLODStrategyInfo();
        
        // Should return one of the two expected messages
        boolean isValidMessage = strategyInfo.equals("Distant Horizons detected - advanced LOD available") ||
                                strategyInfo.equals("Using fallback distance-based LOD calculation");
        
        assertTrue(isValidMessage, "Should return a recognized LOD strategy message");
    }

    @Test
    void testGetLODStrategyInfo_ConsistentWithAvailabilityCheck() {
        boolean isAvailable = ModDetection.isDistantHorizonsAvailable();
        String strategyInfo = ModDetection.getLODStrategyInfo();
        
        if (isAvailable) {
            assertEquals("Distant Horizons detected - advanced LOD available", strategyInfo,
                        "Strategy info should match availability when DH is available");
        } else {
            assertEquals("Using fallback distance-based LOD calculation", strategyInfo,
                        "Strategy info should match availability when DH is not available");
        }
    }

    @Test
    void testModDetection_StaticMethodsWork() {
        // Test that both static methods can be called without instantiation
        assertDoesNotThrow(() -> {
            ModDetection.isDistantHorizonsAvailable();
            ModDetection.getLODStrategyInfo();
        });
    }
}

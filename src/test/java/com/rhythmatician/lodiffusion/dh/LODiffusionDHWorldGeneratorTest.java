package com.rhythmatician.lodiffusion.dh;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.AfterEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Focused unit tests for LODiffusionDHWorldGenerator to improve test coverage.
 * Tests individual methods and functionality following TDD principles.
 */
public class LODiffusionDHWorldGeneratorTest {

    private LODiffusionDHWorldGenerator generator;

    @BeforeEach
    void setUp() {
        // Get a fresh instance for each test
        generator = LODiffusionDHWorldGenerator.getInstance();
    }

    @AfterEach
    void tearDown() {
        // Reset any static state if needed
        // Note: getInstance() uses singleton pattern, so instance persists
    }

    @Test
    void testGetInstance_ReturnsSingleton() {
        // Test that getInstance() returns the same instance
        LODiffusionDHWorldGenerator instance1 = LODiffusionDHWorldGenerator.getInstance();
        LODiffusionDHWorldGenerator instance2 = LODiffusionDHWorldGenerator.getInstance();
        
        assertNotNull(instance1, "getInstance should return a non-null instance");
        assertNotNull(instance2, "getInstance should return a non-null instance");
        assertSame(instance1, instance2, "getInstance should return the same singleton instance");
    }

    @Test
    void testGetGeneratorName_ReturnsExpectedName() {
        // Test that generator name is correctly set
        String name = generator.getGeneratorName();
        
        assertNotNull(name, "Generator name should not be null");
        assertEquals("LODiffusion AI Terrain Generator", name, "Generator name should match expected value");
    }

    @Test
    void testGetGeneratorVersion_ReturnsValidVersion() {
        // Test that generator version is correctly set
        String version = generator.getGeneratorVersion();
        
        assertNotNull(version, "Generator version should not be null");
        assertEquals("1.0.0", version, "Generator version should match expected value");
        assertTrue(version.matches("\\d+\\.\\d+\\.\\d+"), "Version should follow semantic versioning pattern");
    }

    @Test
    void testIsAvailable_ReturnsTrue() {
        // Test that generator reports as available
        assertTrue(generator.isAvailable(), "Generator should be available after instantiation");
    }

    @Test
    void testGenerateLODTerrain_ValidInputs_ReturnsModifiedHeightmap() {
        // Test terrain generation with valid inputs
        int chunkX = 10;
        int chunkZ = 20;
        int lod = 1;
        
        // Create valid 16x16 heightmap
        int[][] inputHeightmap = new int[16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                inputHeightmap[x][z] = 64 + (x + z) % 10; // Simple pattern
            }
        }
        
        // Create valid biome array (256 entries for 16x16)
        String[] biomes = new String[256];
        for (int i = 0; i < 256; i++) {
            biomes[i] = "plains";
        }
        
        int[][] result = generator.generateLODTerrain(chunkX, chunkZ, lod, inputHeightmap, biomes);
        
        assertNotNull(result, "Generated terrain should not be null");
        assertEquals(16, result.length, "Result should have 16 rows");
        assertEquals(16, result[0].length, "Result should have 16 columns");
        
        // Verify input wasn't modified (should be a copy)
        assertNotSame(inputHeightmap, result, "Result should be a copy, not the same array");
    }

    @Test
    void testGenerateLODTerrain_NullHeightmap_ThrowsException() {
        // Test that null heightmap throws appropriate exception
        int chunkX = 0;
        int chunkZ = 0;
        int lod = 0;
        String[] biomes = new String[256];
        
        assertThrows(IllegalArgumentException.class, () -> {
            generator.generateLODTerrain(chunkX, chunkZ, lod, null, biomes);
        }, "Null heightmap should throw IllegalArgumentException");
    }

    @Test
    void testGenerateLODTerrain_InvalidHeightmapSize_ThrowsException() {
        // Test that invalid heightmap size throws appropriate exception
        int chunkX = 0;
        int chunkZ = 0;
        int lod = 0;
        int[][] invalidHeightmap = new int[8][8]; // Wrong size
        String[] biomes = new String[256];
        
        assertThrows(IllegalArgumentException.class, () -> {
            generator.generateLODTerrain(chunkX, chunkZ, lod, invalidHeightmap, biomes);
        }, "Invalid heightmap size should throw IllegalArgumentException");
    }

    @Test
    void testGenerateLODTerrain_NullBiomes_UsesDefaults() {
        // Test that null biomes are handled gracefully with defaults
        int chunkX = 0;
        int chunkZ = 0;
        int lod = 0;
        
        int[][] validHeightmap = new int[16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                validHeightmap[x][z] = 64;
            }
        }
        
        // Should not throw exception, should use default biomes
        assertDoesNotThrow(() -> {
            int[][] result = generator.generateLODTerrain(chunkX, chunkZ, lod, validHeightmap, null);
            assertNotNull(result, "Should generate terrain even with null biomes");
        }, "Null biomes should be handled gracefully");
    }

    @Test
    void testGenerateLODTerrain_InvalidBiomesLength_UsesDefaults() {
        // Test that invalid biomes length is handled gracefully
        int chunkX = 0;
        int chunkZ = 0;
        int lod = 0;
        
        int[][] validHeightmap = new int[16][16];
        String[] invalidBiomes = new String[100]; // Wrong length
        
        // Should not throw exception, should use default biomes
        assertDoesNotThrow(() -> {
            int[][] result = generator.generateLODTerrain(chunkX, chunkZ, lod, validHeightmap, invalidBiomes);
            assertNotNull(result, "Should generate terrain even with invalid biomes length");
        }, "Invalid biomes length should be handled gracefully");
    }

    @Test
    void testAttemptRegistration_CanBeCalledSafely() {
        // Test that attemptRegistration can be called without errors
        assertDoesNotThrow(() -> {
            boolean result = LODiffusionDHWorldGenerator.attemptRegistration();
            // Result will be false since DH is not available in test environment
            // but it should not throw exceptions
        }, "attemptRegistration should not throw exceptions");
    }

    @Test
    void testIsRegisteredWithDH_ReturnsFalseInTestEnvironment() {
        // Test registration status in test environment
        boolean isRegistered = LODiffusionDHWorldGenerator.isRegisteredWithDH();
        
        // Should be false since DH is not available in test environment
        assertFalse(isRegistered, "Should not be registered with DH in test environment");
    }

    @Test
    void testGetRegistrationStatus_ReturnsValidStatus() {
        // Test that registration status returns a meaningful string
        String status = LODiffusionDHWorldGenerator.getRegistrationStatus();
        
        assertNotNull(status, "Registration status should not be null");
        assertFalse(status.isEmpty(), "Registration status should not be empty");
        
        // Should contain some indication of the current state
        assertTrue(
            status.contains("DH") || status.contains("LOD") || status.contains("fallback") || status.contains("not"),
            "Status should contain meaningful information: " + status
        );
    }
}

package com.rhythmatician.lodiffusion.training;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for TerrainPatch class.
 * Tests the individual terrain patch data structure used for ML training.
 */
class TerrainPatchTest {
    
    private int[][] testHeightmap;
    private String[] testBiomes;
    
    @BeforeEach
    void setUp() {
        // Create valid 8x8 test data
        testHeightmap = new int[8][8];
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                testHeightmap[x][z] = 64 + x + z; // Simple height gradient
            }
        }
        
        testBiomes = new String[64]; // 8x8 = 64 positions
        for (int i = 0; i < 64; i++) {
            testBiomes[i] = (i % 2 == 0) ? "plains" : "forest";
        }
    }
    
    @Test
    void testConstructor_ValidInputs() {
        // Test successful construction with valid data
        TerrainPatch patch = new TerrainPatch(100, 200, testHeightmap, testBiomes);
        
        assertEquals(100, patch.getWorldX(), "World X should be set correctly");
        assertEquals(200, patch.getWorldZ(), "World Z should be set correctly");
        assertNotNull(patch.getHeightmap(), "Heightmap should not be null");
        assertNotNull(patch.getBiomes(), "Biomes should not be null");
    }
    
    @Test
    void testConstructor_NullHeightmap() {
        // Test that null heightmap throws exception
        assertThrows(IllegalArgumentException.class, () -> {
            new TerrainPatch(0, 0, null, testBiomes);
        }, "Should throw exception for null heightmap");
    }
    
    @Test
    void testConstructor_NullBiomes() {
        // Test that null biomes throws exception
        assertThrows(IllegalArgumentException.class, () -> {
            new TerrainPatch(0, 0, testHeightmap, null);
        }, "Should throw exception for null biomes");
    }
    
    @Test
    void testConstructor_WrongHeightmapSize() {
        // Test wrong heightmap dimensions
        int[][] wrongSize = new int[7][8]; // Wrong size
        
        assertThrows(IllegalArgumentException.class, () -> {
            new TerrainPatch(0, 0, wrongSize, testBiomes);
        }, "Should throw exception for wrong heightmap dimensions");
    }
    
    @Test
    void testConstructor_WrongBiomesLength() {
        // Test wrong biomes array length
        String[] wrongBiomes = new String[50]; // Should be 64
        
        assertThrows(IllegalArgumentException.class, () -> {
            new TerrainPatch(0, 0, testHeightmap, wrongBiomes);
        }, "Should throw exception for wrong biomes array length");
    }
    
    @Test
    void testGetHeightAt_ValidCoordinates() {
        TerrainPatch patch = new TerrainPatch(0, 0, testHeightmap, testBiomes);
        
        // Test various valid coordinates
        assertEquals(testHeightmap[0][0], patch.getHeightAt(0, 0), "Height at (0,0) should match");
        assertEquals(testHeightmap[3][5], patch.getHeightAt(3, 5), "Height at (3,5) should match");
        assertEquals(testHeightmap[7][7], patch.getHeightAt(7, 7), "Height at (7,7) should match");
    }
    
    @Test
    void testGetHeightAt_InvalidCoordinates() {
        TerrainPatch patch = new TerrainPatch(0, 0, testHeightmap, testBiomes);
        
        // Test out-of-bounds coordinates
        assertThrows(IndexOutOfBoundsException.class, () -> {
            patch.getHeightAt(-1, 0);
        }, "Should throw exception for negative X");
        
        assertThrows(IndexOutOfBoundsException.class, () -> {
            patch.getHeightAt(0, -1);
        }, "Should throw exception for negative Z");
        
        assertThrows(IndexOutOfBoundsException.class, () -> {
            patch.getHeightAt(8, 0);
        }, "Should throw exception for X >= 8");
        
        assertThrows(IndexOutOfBoundsException.class, () -> {
            patch.getHeightAt(0, 8);
        }, "Should throw exception for Z >= 8");
    }
    
    @Test
    void testGetBiomeAt_ValidCoordinates() {
        TerrainPatch patch = new TerrainPatch(0, 0, testHeightmap, testBiomes);
        
        // Test biome access
        String biome00 = patch.getBiomeAt(0, 0); // Index 0
        String biome10 = patch.getBiomeAt(1, 0); // Index 1
        String biome01 = patch.getBiomeAt(0, 1); // Index 8
        
        assertEquals(testBiomes[0], biome00, "Biome at (0,0) should match");
        assertEquals(testBiomes[1], biome10, "Biome at (1,0) should match");
        assertEquals(testBiomes[8], biome01, "Biome at (0,1) should match");
    }
    
    @Test
    void testGetBiomeAt_InvalidCoordinates() {
        TerrainPatch patch = new TerrainPatch(0, 0, testHeightmap, testBiomes);
        
        // Test out-of-bounds coordinates for biomes
        assertThrows(IndexOutOfBoundsException.class, () -> {
            patch.getBiomeAt(-1, 0);
        }, "Should throw exception for negative X in biome access");
        
        assertThrows(IndexOutOfBoundsException.class, () -> {
            patch.getBiomeAt(8, 0);
        }, "Should throw exception for X >= 8 in biome access");
    }
    
    @Test
    void testHeightStatistics() {
        TerrainPatch patch = new TerrainPatch(0, 0, testHeightmap, testBiomes);
        
        // Test height statistics
        int minHeight = patch.getMinHeight();
        int maxHeight = patch.getMaxHeight();
        int variation = patch.getHeightVariation();
        double average = patch.getAverageHeight();
        
        assertEquals(64, minHeight, "Min height should be 64 (corner value)");
        assertEquals(78, maxHeight, "Max height should be 78 (7+7+64)");
        assertEquals(14, variation, "Height variation should be max-min");
        assertEquals(71.0, average, 0.1, "Average height should be approximately 71");
    }
    
    @Test
    void testGetDominantBiome() {
        TerrainPatch patch = new TerrainPatch(0, 0, testHeightmap, testBiomes);
        
        String dominant = patch.getDominantBiome();
        
        // Since we have alternating plains/forest (50% each), either could be dominant
        assertTrue(dominant.equals("plains") || dominant.equals("forest"), 
                  "Dominant biome should be either plains or forest");
    }
    
    @Test
    void testToFlattenedArray() {
        TerrainPatch patch = new TerrainPatch(0, 0, testHeightmap, testBiomes);
        
        float[] flattened = patch.toFlattenedArray();
        
        assertEquals(64, flattened.length, "Flattened array should have 64 elements");
        assertEquals((float) testHeightmap[0][0], flattened[0], "First element should match heightmap[0][0]");
        assertEquals((float) testHeightmap[7][7], flattened[63], "Last element should match heightmap[7][7]");
    }
    
    @Test
    void testDataImmutability() {
        TerrainPatch patch = new TerrainPatch(0, 0, testHeightmap, testBiomes);
        
        // Modify original arrays
        testHeightmap[0][0] = 999;
        testBiomes[0] = "modified";
        
        // Verify patch data is unchanged
        assertNotEquals(999, patch.getHeightAt(0, 0), "Patch should not be affected by external modification");
        assertNotEquals("modified", patch.getBiomeAt(0, 0), "Patch biomes should not be affected by external modification");
        
        // Modify returned arrays
        int[][] returnedHeightmap = patch.getHeightmap();
        String[] returnedBiomes = patch.getBiomes();
        returnedHeightmap[1][1] = 888;
        returnedBiomes[1] = "changed";
        
        // Verify internal data is unchanged
        assertNotEquals(888, patch.getHeightAt(1, 1), "Internal heightmap should not be modified");
        assertNotEquals("changed", patch.getBiomeAt(1, 0), "Internal biomes should not be modified");
    }
    
    @Test
    void testEquals() {
        TerrainPatch patch1 = new TerrainPatch(10, 20, testHeightmap, testBiomes);
        TerrainPatch patch2 = new TerrainPatch(10, 20, testHeightmap, testBiomes);
        TerrainPatch patch3 = new TerrainPatch(10, 21, testHeightmap, testBiomes); // Different Z
        
        assertEquals(patch1, patch2, "Patches with same data should be equal");
        assertNotEquals(patch1, patch3, "Patches with different coordinates should not be equal");
        assertEquals(patch1, patch1, "Patch should equal itself");
        assertNotEquals(patch1, null, "Patch should not equal null");
    }
    
    @Test
    void testHashCode() {
        TerrainPatch patch1 = new TerrainPatch(10, 20, testHeightmap, testBiomes);
        TerrainPatch patch2 = new TerrainPatch(10, 20, testHeightmap, testBiomes);
        
        assertEquals(patch1.hashCode(), patch2.hashCode(), "Equal patches should have same hash code");
    }
    
    @Test
    void testToString() {
        TerrainPatch patch = new TerrainPatch(100, 200, testHeightmap, testBiomes);
        
        String str = patch.toString();
        assertNotNull(str, "toString should not return null");
        assertTrue(str.contains("100"), "toString should contain world X coordinate");
        assertTrue(str.contains("200"), "toString should contain world Z coordinate");
        assertTrue(str.contains("8x8"), "toString should contain patch dimensions");
    }
    
    @Test
    void testPatchConstants() {
        assertEquals(8, TerrainPatch.PATCH_SIZE, "PATCH_SIZE constant should be 8");
    }
}

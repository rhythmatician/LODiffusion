package com.rhythmatician.lodiffusion.training;

import fixtures.TestWorldFixtures;
import fixtures.TerrainPatchDatasetFixture;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import java.io.File;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for TerrainPatchDataset class.
 * Tests the dataset functionality for loading and managing terrain patches for ML training.
 */
class TerrainPatchDatasetTest {

    private TerrainPatchDataset dataset;

    @BeforeEach
    void setUp() {
        dataset = new TerrainPatchDataset();
    }

    @Test
    void testConstructor() {
        // Test initial state after construction
        assertNotNull(dataset, "Dataset should be created successfully");
        assertEquals(0, dataset.getPatchCount(), "Initial patch count should be 0");
        assertFalse(dataset.isLoaded(), "Dataset should not be loaded initially");
        assertEquals("Dataset not loaded", dataset.getDatasetSummary(), "Summary should indicate not loaded");
    }

    @Test
    void testConstants() {
        // Test that constants match expected values
        assertEquals(8, TerrainPatchDataset.PATCH_SIZE, "PATCH_SIZE should be 8");
        assertEquals(16, TerrainPatchDataset.CHUNK_SIZE, "CHUNK_SIZE should be 16");
        assertEquals(4, TerrainPatchDataset.PATCHES_PER_CHUNK, "PATCHES_PER_CHUNK should be 4 (2x2 grid)");
    }

    @Test
    void testLoadFromWorldData_NoWorldDataAvailable() {
        // Test behavior when world data is not available
        File[] regionFiles = TestWorldFixtures.getTestDataRegionFiles();

        if (regionFiles.length == 0) {
            assertThrows(IllegalArgumentException.class, () -> {
                dataset.loadFromWorldData(regionFiles);
            }, "Should throw IllegalArgumentException when region files array is empty");

            assertFalse(dataset.isLoaded(), "Dataset should not be marked as loaded after failed load");
            assertEquals(0, dataset.getPatchCount(), "Patch count should remain 0 after failed load");
        }
    }

    @Test
    void testLoadFromWorldData_WithWorldDataAvailable() {
        // Test loading when world data is available
        if (TestWorldFixtures.isTestDataAvailable()) {
            File[] regionFiles = TestWorldFixtures.getTestDataRegionFiles();
            int patchesLoaded = dataset.loadFromWorldData(regionFiles);

            assertTrue(patchesLoaded > 0, "Should load at least some patches when world data is available");
            assertTrue(dataset.isLoaded(), "Dataset should be marked as loaded");
            assertEquals(patchesLoaded, dataset.getPatchCount(), "Patch count should match loaded count");

            // Verify patches are actually created
            assertTrue(dataset.getPatchCount() > 0, "Should have patches after loading");
            assertNotNull(dataset.getAllPatches(), "getAllPatches should not return null");
            assertFalse(dataset.getAllPatches().isEmpty(), "getAllPatches should not be empty");
        }
    }    @Test
    void testGetPatch_ValidIndex() {
        // Use cached dataset for faster test execution
        if (TerrainPatchDatasetFixture.isCachedDatasetAvailable()) {
            int patchCount = TerrainPatchDatasetFixture.getCachedPatchCount();
            
            if (patchCount > 0) {
                TerrainPatch patch = TerrainPatchDatasetFixture.getCachedPatch(0);
                assertNotNull(patch, "First patch should not be null");
                assertEquals(8, patch.getHeightmap().length, "Patch should have 8x8 heightmap");
                assertEquals(64, patch.getBiomes().length, "Patch should have 64 biome entries");
            }
        }
    }

    @Test
    void testGetPatch_InvalidIndex() {
        // Test invalid indices on empty dataset
        assertThrows(IndexOutOfBoundsException.class, () -> {
            dataset.getPatch(0);
        }, "Should throw exception for index 0 on empty dataset");

        assertThrows(IndexOutOfBoundsException.class, () -> {
            dataset.getPatch(-1);
        }, "Should throw exception for negative index");
          assertThrows(IndexOutOfBoundsException.class, () -> {
            dataset.getPatch(10);
        }, "Should throw exception for index beyond dataset size");
    }    @Test
    void testGetAllPatches_ImmutableCopy() {
        // Use fresh dataset since we need to call methods on the dataset object
        TerrainPatchDataset testDataset = TerrainPatchDatasetFixture.getFreshDataset();
        if (testDataset != null) {
            List<TerrainPatch> patches1 = testDataset.getAllPatches();
            List<TerrainPatch> patches2 = testDataset.getAllPatches();

            assertNotSame(patches1, patches2, "getAllPatches should return different list instances");
            assertEquals(patches1.size(), patches2.size(), "Both lists should have same size");

            // Verify modifying returned list doesn't affect dataset
            int originalSize = patches1.size();
            if (originalSize > 0) {
                patches1.clear(); // Modify returned list
                assertEquals(originalSize, testDataset.getPatchCount(), "Dataset size should be unchanged");
            }
        }
    }    @Test
    void testClear() {
        // Use fresh dataset since we need to modify it (clear)
        TerrainPatchDataset testDataset = TerrainPatchDatasetFixture.getFreshDataset();
        if (testDataset != null) {
            int initialCount = testDataset.getPatchCount();
            
            if (initialCount > 0) {
                assertTrue(testDataset.isLoaded(), "Dataset should be loaded before clear");
                
                // Clear the dataset
                testDataset.clear();
                
                assertEquals(0, testDataset.getPatchCount(), "Patch count should be 0 after clear");
                assertFalse(testDataset.isLoaded(), "Dataset should not be loaded after clear");
                assertTrue(testDataset.getAllPatches().isEmpty(), "All patches list should be empty after clear");
            }
        }
    }    @Test
    void testGetDatasetSummary_LoadedState() {
        // Use fresh dataset since we need to call methods on the dataset object
        TerrainPatchDataset testDataset = TerrainPatchDatasetFixture.getFreshDataset();
        if (testDataset != null) {
            String summary = testDataset.getDatasetSummary();
            assertNotNull(summary, "Summary should not be null");
            assertTrue(summary.contains("TerrainPatchDataset Summary"), "Summary should contain title");
            assertTrue(summary.contains("Total patches:"), "Summary should contain patch count");
            assertTrue(summary.contains("8x8"), "Summary should contain patch size");
            assertTrue(summary.contains("Ready for training"), "Summary should indicate ready status");
        }
    }

    @Test
    void testGetDatasetSummary_NotLoadedState() {
        // Test summary when not loaded
        String summary = dataset.getDatasetSummary();        assertEquals("Dataset not loaded", summary, "Summary should indicate not loaded");
    }    @Test
    void testPatchCoordinateValidation() {
        // Use cached dataset for faster test execution
        if (TerrainPatchDatasetFixture.isCachedDatasetAvailable()) {
            int patchCount = TerrainPatchDatasetFixture.getCachedPatchCount();
            
            if (patchCount > 0) {
                // Verify patches have reasonable coordinates
                for (int i = 0; i < Math.min(3, patchCount); i++) {
                    TerrainPatch patch = TerrainPatchDatasetFixture.getCachedPatch(i);
                    
                    // Patch coordinates should be multiples of 8 (patch alignment)
                    assertTrue(patch.getWorldX() % 8 == 0,
                              "Patch world X should be aligned to 8-block boundaries");
                    assertTrue(patch.getWorldZ() % 8 == 0,
                              "Patch world Z should be aligned to 8-block boundaries");
                }
            }
        }
    }    @Test
    void testPatchDataValidation() {
        // Use cached dataset for faster test execution
        if (TerrainPatchDatasetFixture.isCachedDatasetAvailable()) {
            int patchCount = TerrainPatchDatasetFixture.getCachedPatchCount();
            
            if (patchCount > 0) {
                TerrainPatch firstPatch = TerrainPatchDatasetFixture.getCachedPatch(0);
                
                // Validate heightmap data
                int[][] heightmap = firstPatch.getHeightmap();
                assertEquals(8, heightmap.length, "Heightmap should have 8 rows");
                for (int x = 0; x < 8; x++) {
                    assertEquals(8, heightmap[x].length, "Each heightmap row should have 8 columns");
                    for (int z = 0; z < 8; z++) {
                        assertTrue(heightmap[x][z] > 0, "Height values should be positive");
                        assertTrue(heightmap[x][z] < 320, "Height values should be reasonable (< world height)");
                    }
                }
                
                // Validate biome data
                String[] biomes = firstPatch.getBiomes();
                assertEquals(64, biomes.length, "Should have 64 biome entries for 8x8 patch");
                for (String biome : biomes) {
                    assertNotNull(biome, "Biome entries should not be null");
                    assertFalse(biome.trim().isEmpty(), "Biome entries should not be empty");
                }
            }
        }
    }    @Test
    void testMultipleLoadCalls() {
        // Use fresh dataset since we need to call loadFromWorldData multiple times
        if (TestWorldFixtures.isTestDataAvailable()) {
            TerrainPatchDataset testDataset = new TerrainPatchDataset();
            File[] regionFiles = TestWorldFixtures.getTestDataRegionFiles();
            
            // Load data twice
            int firstLoad = testDataset.loadFromWorldData(regionFiles);
            int secondLoad = testDataset.loadFromWorldData(regionFiles);
            
            // Second load should replace first load
            assertEquals(firstLoad, secondLoad, "Multiple loads should produce same result");
            assertEquals(secondLoad, testDataset.getPatchCount(), "Patch count should match last load");
            assertTrue(testDataset.isLoaded(), "Dataset should remain loaded after multiple loads");
        }
    }

    @Test
    void testPatchesPerChunkCalculation() {
        // Verify the mathematical relationship between chunks and patches
        // 16x16 chunk divided into 8x8 patches = 2x2 = 4 patches per chunk
        assertEquals(4, TerrainPatchDataset.PATCHES_PER_CHUNK,
                    "Should have 4 patches per chunk (2x2 grid of 8x8 patches in 16x16 chunk)");

        // Verify calculation
        int patchesPerRow = TerrainPatchDataset.CHUNK_SIZE / TerrainPatchDataset.PATCH_SIZE;
        int calculatedPatchesPerChunk = patchesPerRow * patchesPerRow;
        assertEquals(TerrainPatchDataset.PATCHES_PER_CHUNK, calculatedPatchesPerChunk,                    "PATCHES_PER_CHUNK should match calculated value");
    }    @Test
    void testPatchDistribution() {
        // Use cached dataset for faster test execution
        if (TerrainPatchDatasetFixture.isCachedDatasetAvailable()) {
            int patchCount = TerrainPatchDatasetFixture.getCachedPatchCount();
            
            if (patchCount >= 4) {
                // Check that patches from the same chunk have expected coordinate relationships
                // This is a smoke test for the patch generation logic
                List<TerrainPatch> allPatches = TerrainPatchDatasetFixture.getCachedPatches();
                
                // Count patches with different coordinate patterns
                java.util.Set<String> coordSet = new java.util.HashSet<>();
                
                for (int i = 0; i < Math.min(20, allPatches.size()); i++) {
                    TerrainPatch patch = allPatches.get(i);
                    String coordKey = patch.getWorldX() + "," + patch.getWorldZ();
                    coordSet.add(coordKey);
                }
                
                assertTrue(coordSet.size() > 1, "Should have patches at different coordinates");
            }
        }
    }
}

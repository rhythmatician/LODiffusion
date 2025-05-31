package fixtures;

import com.rhythmatician.lodiffusion.training.TerrainPatchDataset;
import com.rhythmatician.lodiffusion.training.TerrainPatch;
import java.io.File;
import java.util.List;

/**
 * Test fixture that provides a cached, pre-loaded TerrainPatchDataset for tests.
 * This fixture loads the test data once and provides access to the loaded dataset,
 * significantly improving test performance by avoiding repeated NBT parsing.
 */
public class TerrainPatchDatasetFixture {
    
    private static TerrainPatchDataset cachedDataset = null;
    private static boolean cacheInitialized = false;
    private static boolean cacheLoadSuccessful = false;
    
    /**
     * Get a cached, pre-loaded TerrainPatchDataset.
     * This method loads the test data once and returns the same dataset instance
     * for all subsequent calls, improving test performance.
     * 
     * @return The cached dataset, or null if test data is not available
     */
    public static synchronized TerrainPatchDataset getCachedDataset() {
        if (!cacheInitialized) {
            initializeCache();
        }
        
        return cacheLoadSuccessful ? cachedDataset : null;
    }
    
    /**
     * Check if a cached dataset is available.
     * This is faster than calling getCachedDataset() when you only need to check availability.
     * 
     * @return true if a cached dataset is available and loaded successfully
     */
    public static synchronized boolean isCachedDatasetAvailable() {
        if (!cacheInitialized) {
            initializeCache();
        }
        
        return cacheLoadSuccessful;
    }
    
    /**
     * Get the number of patches in the cached dataset.
     * 
     * @return Number of patches, or 0 if dataset is not available
     */
    public static int getCachedPatchCount() {
        TerrainPatchDataset dataset = getCachedDataset();
        return dataset != null ? dataset.getPatchCount() : 0;
    }
    
    /**
     * Get a specific patch from the cached dataset.
     * 
     * @param index The patch index
     * @return The terrain patch at the specified index
     * @throws IllegalStateException if cached dataset is not available
     * @throws IndexOutOfBoundsException if index is invalid
     */
    public static TerrainPatch getCachedPatch(int index) {
        TerrainPatchDataset dataset = getCachedDataset();
        if (dataset == null) {
            throw new IllegalStateException("Cached dataset is not available - test data may be missing");
        }
        
        return dataset.getPatch(index);
    }
    
    /**
     * Get all patches from the cached dataset.
     * 
     * @return List of all terrain patches, or empty list if dataset is not available
     */
    public static List<TerrainPatch> getCachedPatches() {
        TerrainPatchDataset dataset = getCachedDataset();
        return dataset != null ? dataset.getAllPatches() : List.of();
    }
    
    /**
     * Get a fresh copy of the dataset for tests that need to modify it.
     * This creates a new TerrainPatchDataset instance and loads the same data,
     * but returns a new object that can be safely modified without affecting other tests.
     * 
     * @return A fresh TerrainPatchDataset instance, or null if test data is not available
     */
    public static TerrainPatchDataset getFreshDataset() {
        if (!TestWorldFixtures.isTestDataAvailable()) {
            return null;
        }
        
        TerrainPatchDataset freshDataset = new TerrainPatchDataset();
        File[] regionFiles = TestWorldFixtures.getTestDataRegionFiles();
        
        try {
            freshDataset.loadFromWorldData(regionFiles);
            return freshDataset;
        } catch (Exception e) {
            System.err.println("Failed to load fresh dataset: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Clear the cache and force re-initialization on next access.
     * This is useful for tests that need to reset the fixture state.
     */
    public static synchronized void clearCache() {
        cachedDataset = null;
        cacheInitialized = false;
        cacheLoadSuccessful = false;
    }
    
    /**
     * Get information about the cached dataset.
     * 
     * @return Summary string describing the cached dataset state
     */
    public static String getCacheInfo() {
        if (!cacheInitialized) {
            return "Cache not initialized";
        }
        
        if (!cacheLoadSuccessful) {
            return "Cache initialization failed - test data may be unavailable";
        }
        
        TerrainPatchDataset dataset = cachedDataset;
        return String.format("Cached dataset loaded successfully: %d patches, ready for testing",
                           dataset.getPatchCount());
    }
    
    /**
     * Initialize the cache by loading test data.
     * This is called automatically on first access.
     */
    private static void initializeCache() {
        cacheInitialized = true;
        cacheLoadSuccessful = false;
        
        if (!TestWorldFixtures.isTestDataAvailable()) {
            System.out.println("TerrainPatchDatasetFixture: Test data not available, cache will be empty");
            return;
        }
        
        try {
            cachedDataset = new TerrainPatchDataset();
            File[] regionFiles = TestWorldFixtures.getTestDataRegionFiles();
            
            int patchCount = cachedDataset.loadFromWorldData(regionFiles);
            
            if (patchCount > 0) {
                cacheLoadSuccessful = true;
                System.out.println("TerrainPatchDatasetFixture: Successfully cached " + patchCount + " patches");
            } else {
                System.out.println("TerrainPatchDatasetFixture: No patches loaded from test data");
            }
            
        } catch (Exception e) {
            System.err.println("TerrainPatchDatasetFixture: Failed to initialize cache: " + e.getMessage());
            cachedDataset = null;
        }
    }
}

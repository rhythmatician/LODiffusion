package com.rhythmatician.lodiffusion.training;

import com.rhythmatician.lodiffusion.world.ChunkDataExtractor;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Dataset for terrain patches extracted from Minecraft world files.
 * Converts 16x16 chunks into 8x8 patches for machine learning training.
 * 
 * This class supports the training pipeline by providing properly formatted
 * terrain data for the diffusion model training process.
 */
public class TerrainPatchDataset {
    
    public static final int PATCH_SIZE = 8;
    public static final int CHUNK_SIZE = 16;
    public static final int PATCHES_PER_CHUNK = (CHUNK_SIZE / PATCH_SIZE) * (CHUNK_SIZE / PATCH_SIZE); // 4 patches per chunk
    
    private final List<TerrainPatch> patches;
    private boolean isDataLoaded;
    
    /**
     * Constructor for TerrainPatchDataset.
     * Initializes an empty dataset ready for patch loading.
     */
    public TerrainPatchDataset() {
        this.patches = new ArrayList<>();
        this.isDataLoaded = false;
    }
    
    /**
     * Load terrain patches from available world data.
     * Converts real Minecraft chunks into 8x8 training patches.
     * 
     * @return Number of patches loaded
     * @throws IllegalStateException if world data is not available
     */
    public int loadFromWorldData() {
        if (!ChunkDataExtractor.isWorldDataAvailable()) {
            throw new IllegalStateException("World data not available for loading patches");
        }
        
        // Clear existing patches
        patches.clear();
        
        File[] regionFiles = ChunkDataExtractor.getAvailableRegionFiles();
        int totalPatchesLoaded = 0;
        
        for (File regionFile : regionFiles) {
            int[] regionCoords = ChunkDataExtractor.parseRegionCoordinates(regionFile);
            int regionX = regionCoords[0];
            int regionZ = regionCoords[1];
            
            // Process chunks in this region (32x32 chunks per region)
            for (int localChunkX = 0; localChunkX < 32; localChunkX++) {
                for (int localChunkZ = 0; localChunkZ < 32; localChunkZ++) {
                    try {
                        // TODO: Replace with actual NBT parsing when implemented
                        // For now, create mock patches with coordinates
                        int[] worldCoords = ChunkDataExtractor.getWorldChunkCoordinates(
                            regionX, regionZ, localChunkX, localChunkZ);
                        
                        List<TerrainPatch> chunkPatches = createMockPatchesFromChunk(
                            worldCoords[0], worldCoords[1]);
                        patches.addAll(chunkPatches);
                        totalPatchesLoaded += chunkPatches.size();
                        
                    } catch (Exception e) {
                        // Skip failed chunks but continue processing
                        System.err.println("Failed to process chunk at local coords [" + 
                                         localChunkX + ", " + localChunkZ + "]: " + e.getMessage());
                    }
                }
            }
        }
        
        isDataLoaded = true;
        return totalPatchesLoaded;
    }
    
    /**
     * Create 4 patches from a single 16x16 chunk.
     * Splits the chunk into 2x2 grid of 8x8 patches.
     * 
     * @param chunkX World chunk X coordinate
     * @param chunkZ World chunk Z coordinate 
     * @return List of 4 terrain patches
     */
    private List<TerrainPatch> createMockPatchesFromChunk(int chunkX, int chunkZ) {
        List<TerrainPatch> chunkPatches = new ArrayList<>();
        
        // Create 2x2 grid of 8x8 patches from 16x16 chunk
        for (int patchX = 0; patchX < 2; patchX++) {
            for (int patchZ = 0; patchZ < 2; patchZ++) {
                // Calculate patch world coordinates
                int patchWorldX = chunkX * 16 + patchX * 8;
                int patchWorldZ = chunkZ * 16 + patchZ * 8;
                
                // Create mock heightmap data (will be replaced with real NBT data)
                int[][] patchHeightmap = createMockHeightmap(patchWorldX, patchWorldZ);
                String[] patchBiomes = createMockBiomes(patchWorldX, patchWorldZ);
                
                TerrainPatch patch = new TerrainPatch(
                    patchWorldX, patchWorldZ, patchHeightmap, patchBiomes);
                chunkPatches.add(patch);
            }
        }
        
        return chunkPatches;
    }
    
    /**
     * Create mock heightmap data for testing purposes.
     * TODO: Replace with real NBT extraction.
     */
    private int[][] createMockHeightmap(int worldX, int worldZ) {
        int[][] heightmap = new int[PATCH_SIZE][PATCH_SIZE];
        
        for (int x = 0; x < PATCH_SIZE; x++) {
            for (int z = 0; z < PATCH_SIZE; z++) {
                // Create realistic-looking terrain variation
                int baseHeight = 64;
                int variation = (worldX + worldZ + x + z) % 20 - 10; // ±10 variation
                heightmap[x][z] = Math.max(1, baseHeight + variation);
            }
        }
        
        return heightmap;
    }
    
    /**
     * Create mock biome data for testing purposes.
     * TODO: Replace with real NBT extraction.
     */
    private String[] createMockBiomes(int worldX, int worldZ) {
        String[] biomes = new String[PATCH_SIZE * PATCH_SIZE]; // 64 biome entries for 8x8 patch
        
        // Simple biome assignment based on coordinates
        String biome = ((worldX + worldZ) % 3 == 0) ? "plains" : 
                      ((worldX + worldZ) % 3 == 1) ? "forest" : "hills";
        
        for (int i = 0; i < biomes.length; i++) {
            biomes[i] = biome;
        }
        
        return biomes;
    }
    
    /**
     * Get the number of patches in the dataset.
     * @return Number of loaded patches
     */
    public int getPatchCount() {
        return patches.size();
    }
    
    /**
     * Get a specific patch by index.
     * @param index Patch index (0-based)
     * @return TerrainPatch at the specified index
     * @throws IndexOutOfBoundsException if index is invalid
     */
    public TerrainPatch getPatch(int index) {
        if (index < 0 || index >= patches.size()) {
            throw new IndexOutOfBoundsException("Patch index " + index + " out of bounds [0, " + patches.size() + ")");
        }
        return patches.get(index);
    }
    
    /**
     * Get all patches in the dataset.
     * @return List of all terrain patches
     */
    public List<TerrainPatch> getAllPatches() {
        return new ArrayList<>(patches); // Return copy to prevent external modification
    }
    
    /**
     * Check if dataset has been loaded with data.
     * @return true if data has been loaded, false otherwise
     */
    public boolean isLoaded() {
        return isDataLoaded;
    }
    
    /**
     * Get summary statistics about the dataset.
     * @return String describing dataset contents
     */
    public String getDatasetSummary() {
        if (!isDataLoaded) {
            return "Dataset not loaded";
        }
        
        StringBuilder summary = new StringBuilder();
        summary.append("TerrainPatchDataset Summary:\n");
        summary.append("- Total patches: ").append(patches.size()).append("\n");
        summary.append("- Patch size: ").append(PATCH_SIZE).append("x").append(PATCH_SIZE).append("\n");
        summary.append("- Patches per chunk: ").append(PATCHES_PER_CHUNK).append("\n");
        summary.append("- Estimated chunks processed: ~").append(patches.size() / PATCHES_PER_CHUNK).append("\n");
        summary.append("- Data status: Ready for training");
        
        return summary.toString();
    }
    
    /**
     * Clear all loaded patches and reset the dataset.
     */
    public void clear() {
        patches.clear();
        isDataLoaded = false;
    }
}

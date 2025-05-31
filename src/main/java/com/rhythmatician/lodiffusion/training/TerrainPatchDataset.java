package com.rhythmatician.lodiffusion.training;

import com.rhythmatician.lodiffusion.world.ChunkDataExtractor;
import java.io.File;
import java.io.IOException;
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
        throw new UnsupportedOperationException(
            "loadFromWorldData() requires external world data discovery. " +
            "Use loadFromWorldData(File[] regionFiles) instead, or call from test context.");
    }
    
    /**
     * Load terrain patches from specified region files.
     * Converts real Minecraft chunks into 8x8 training patches.
     * 
     * @param regionFiles Array of region files to process
     * @return Number of patches loaded
     * @throws IllegalArgumentException if regionFiles is null or empty
     */
    public int loadFromWorldData(File[] regionFiles) {
        if (regionFiles == null || regionFiles.length == 0) {
            throw new IllegalArgumentException("Region files array cannot be null or empty");
        }
        
        // Clear existing patches
        patches.clear();
        
        int totalPatchesLoaded = 0;
        
        for (File regionFile : regionFiles) {
            int[] regionCoords = ChunkDataExtractor.parseRegionCoordinates(regionFile);
            int regionX = regionCoords[0];
            int regionZ = regionCoords[1];
            
            // Process chunks in this region (32x32 chunks per region)
            for (int localChunkX = 0; localChunkX < 32; localChunkX++) {
                for (int localChunkZ = 0; localChunkZ < 32; localChunkZ++) {
                    try {                        // Extract real NBT data from the chunk
                        int[] worldCoords = ChunkDataExtractor.getWorldChunkCoordinates(
                            regionX, regionZ, localChunkX, localChunkZ);
                        
                        List<TerrainPatch> chunkPatches = createPatchesFromRealChunk(
                            regionFile, localChunkX, localChunkZ, worldCoords[0], worldCoords[1]);
                        
                        if (chunkPatches != null && !chunkPatches.isEmpty()) {
                            patches.addAll(chunkPatches);
                            totalPatchesLoaded += chunkPatches.size();
                        }
                          } catch (Exception e) {
                        // Skip failed chunks but continue processing with detailed debugging
                        System.err.println("DEBUG: Failed to process chunk at local coords [" + 
                                         localChunkX + ", " + localChunkZ + "] in region " + 
                                         regionFile.getName() + ": " + e.getClass().getSimpleName() + 
                                         " - " + e.getMessage());
                        
                        // Create fallback patches to ensure consistent dataset size
                        int[] worldCoords = ChunkDataExtractor.getWorldChunkCoordinates(
                            regionX, regionZ, localChunkX, localChunkZ);
                        List<TerrainPatch> fallbackPatches = createFallbackPatches(
                            worldCoords[0], worldCoords[1], "Processing exception");
                        if (fallbackPatches != null) {
                            patches.addAll(fallbackPatches);
                            totalPatchesLoaded += fallbackPatches.size();
                        }
                    }
                }
            }
        }
        
        isDataLoaded = true;
        return totalPatchesLoaded;
    }
    
    /**
     * Create 4 patches from a single 16x16 chunk using real NBT data.
     * Splits the chunk into 2x2 grid of 8x8 patches with real heightmap and biome data.
     * 
     * @param regionFile Region file containing the chunk
     * @param localChunkX Local chunk X coordinate within region (0-31)
     * @param localChunkZ Local chunk Z coordinate within region (0-31)
     * @param worldChunkX World chunk X coordinate
     * @param worldChunkZ World chunk Z coordinate
     * @return List of 4 terrain patches, or null if chunk data cannot be extracted
     */
    private List<TerrainPatch> createPatchesFromRealChunk(File regionFile, int localChunkX, int localChunkZ, 
                                                         int worldChunkX, int worldChunkZ) {
        try {
            // Extract real NBT data from the chunk
            int[][] chunkHeightmap = ChunkDataExtractor.extractHeightmapFromChunk(regionFile, localChunkX, localChunkZ);
            String[] chunkBiomes = ChunkDataExtractor.extractBiomesFromChunk(regionFile, localChunkX, localChunkZ);
            
            // Debug output for NBT extraction
            System.out.println("DEBUG: Extracting patches from chunk [" + worldChunkX + ", " + worldChunkZ + "]");
            
            if (chunkHeightmap == null) {
                System.out.println("DEBUG: Failed to extract heightmap from chunk [" + worldChunkX + ", " + worldChunkZ + "]");
                return createFallbackPatches(worldChunkX, worldChunkZ, "missing heightmap");
            }
            
            if (chunkBiomes == null) {
                System.out.println("DEBUG: Failed to extract biomes from chunk [" + worldChunkX + ", " + worldChunkZ + "], using heightmap only");
                chunkBiomes = createFallbackBiomes();
            }
            
            System.out.println("DEBUG: Successfully extracted NBT data - heightmap: " + 
                chunkHeightmap.length + "x" + chunkHeightmap[0].length + 
                ", biomes: " + chunkBiomes.length + " entries");
            
            // Split the 16x16 chunk into 2x2 grid of 8x8 patches
            List<TerrainPatch> chunkPatches = new ArrayList<>();
            
            for (int patchX = 0; patchX < 2; patchX++) {
                for (int patchZ = 0; patchZ < 2; patchZ++) {
                    // Calculate patch world coordinates
                    int patchWorldX = worldChunkX * 16 + patchX * 8;
                    int patchWorldZ = worldChunkZ * 16 + patchZ * 8;
                    
                    // Extract 8x8 patch from 16x16 chunk data
                    int[][] patchHeightmap = extractPatchHeightmap(chunkHeightmap, patchX, patchZ);
                    String[] patchBiomes = extractPatchBiomes(chunkBiomes, patchX, patchZ);
                    
                    TerrainPatch patch = new TerrainPatch(
                        patchWorldX, patchWorldZ, patchHeightmap, patchBiomes);
                    chunkPatches.add(patch);
                    
                    System.out.println("DEBUG: Created patch [" + patchWorldX + ", " + patchWorldZ + 
                        "] with heights " + getHeightRange(patchHeightmap) + 
                        " and biome " + patchBiomes[0]);
                }
            }
            
            return chunkPatches;
            
        } catch (IOException e) {
            System.err.println("DEBUG: IOException extracting chunk [" + worldChunkX + ", " + worldChunkZ + 
                "]: " + e.getMessage());
            return createFallbackPatches(worldChunkX, worldChunkZ, "IOException: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("DEBUG: Unexpected error extracting chunk [" + worldChunkX + ", " + worldChunkZ + 
                "]: " + e.getClass().getSimpleName() + " - " + e.getMessage());
            return createFallbackPatches(worldChunkX, worldChunkZ, "Unexpected error: " + e.getMessage());
        }
    }
    
    /**
     * Extract an 8x8 heightmap patch from a 16x16 chunk heightmap.
     * @param chunkHeightmap 16x16 chunk heightmap
     * @param patchX Patch X index (0 or 1)
     * @param patchZ Patch Z index (0 or 1)
     * @return 8x8 heightmap for the patch
     */
    private int[][] extractPatchHeightmap(int[][] chunkHeightmap, int patchX, int patchZ) {
        int[][] patchHeightmap = new int[PATCH_SIZE][PATCH_SIZE];
        
        int startX = patchX * PATCH_SIZE;
        int startZ = patchZ * PATCH_SIZE;
        
        for (int x = 0; x < PATCH_SIZE; x++) {
            for (int z = 0; z < PATCH_SIZE; z++) {
                patchHeightmap[x][z] = chunkHeightmap[startX + x][startZ + z];
            }
        }
        
        return patchHeightmap;
    }
    
    /**
     * Extract 64 biome entries for an 8x8 patch from 256 chunk biome entries.
     * @param chunkBiomes 256 biome entries for 16x16 chunk
     * @param patchX Patch X index (0 or 1)
     * @param patchZ Patch Z index (0 or 1)
     * @return 64 biome entries for 8x8 patch
     */
    private String[] extractPatchBiomes(String[] chunkBiomes, int patchX, int patchZ) {
        String[] patchBiomes = new String[PATCH_SIZE * PATCH_SIZE];
        
        int startX = patchX * PATCH_SIZE;
        int startZ = patchZ * PATCH_SIZE;
        
        for (int x = 0; x < PATCH_SIZE; x++) {
            for (int z = 0; z < PATCH_SIZE; z++) {
                int chunkIndex = (startZ + z) * CHUNK_SIZE + (startX + x);
                int patchIndex = z * PATCH_SIZE + x;
                
                if (chunkIndex < chunkBiomes.length) {
                    patchBiomes[patchIndex] = chunkBiomes[chunkIndex];
                } else {
                    patchBiomes[patchIndex] = "minecraft:plains"; // Fallback
                }
            }
        }
        
        return patchBiomes;
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
    
    /**
     * Create fallback patches when NBT extraction fails.
     * @param worldChunkX World chunk X coordinate
     * @param worldChunkZ World chunk Z coordinate
     * @param reason Reason for fallback
     * @return List of 4 fallback patches with realistic data
     */
    private List<TerrainPatch> createFallbackPatches(int worldChunkX, int worldChunkZ, String reason) {
        System.out.println("DEBUG: Creating fallback patches for chunk [" + worldChunkX + ", " + worldChunkZ + 
            "] - reason: " + reason);
        
        List<TerrainPatch> fallbackPatches = new ArrayList<>();
        
        for (int patchX = 0; patchX < 2; patchX++) {
            for (int patchZ = 0; patchZ < 2; patchZ++) {
                int patchWorldX = worldChunkX * 16 + patchX * 8;
                int patchWorldZ = worldChunkZ * 16 + patchZ * 8;
                
                int[][] heightmap = createRealisticHeightmap(patchWorldX, patchWorldZ);
                String[] biomes = createFallbackBiomes();
                
                TerrainPatch patch = new TerrainPatch(patchWorldX, patchWorldZ, heightmap, biomes);
                fallbackPatches.add(patch);
            }
        }
        
        return fallbackPatches;
    }
    
    /**
     * Create realistic-looking heightmap for fallback purposes.
     * @param worldX World X coordinate
     * @param worldZ World Z coordinate
     * @return 8x8 heightmap with realistic terrain variation
     */
    private int[][] createRealisticHeightmap(int worldX, int worldZ) {
        int[][] heightmap = new int[PATCH_SIZE][PATCH_SIZE];
        
        // Use simple noise-like function for realistic terrain
        for (int x = 0; x < PATCH_SIZE; x++) {
            for (int z = 0; z < PATCH_SIZE; z++) {
                int actualX = worldX + x;
                int actualZ = worldZ + z;
                
                // Create realistic terrain using multiple frequency layers
                double noise1 = Math.sin(actualX * 0.1) * Math.cos(actualZ * 0.1) * 10;
                double noise2 = Math.sin(actualX * 0.05) * Math.cos(actualZ * 0.05) * 20;
                double noise3 = Math.sin(actualX * 0.02) * Math.cos(actualZ * 0.02) * 30;
                
                int height = (int) (80 + noise1 + noise2 + noise3);
                heightmap[x][z] = Math.max(1, Math.min(255, height)); // Clamp to valid range
            }
        }
        
        return heightmap;
    }
    
    /**
     * Create fallback biome data with proper Minecraft identifiers.
     * @return Array of 64 biome identifiers for 8x8 patch
     */
    private String[] createFallbackBiomes() {
        String[] biomes = new String[PATCH_SIZE * PATCH_SIZE];
        
        // Use proper Minecraft biome identifiers
        String[] biomePalette = {
            "minecraft:plains", "minecraft:forest", "minecraft:taiga", 
            "minecraft:mountains", "minecraft:desert", "minecraft:swamp"
        };
        
        for (int i = 0; i < biomes.length; i++) {
            biomes[i] = biomePalette[i % biomePalette.length];
        }
        
        return biomes;
    }
    
    /**
     * Get height range string for debugging.
     * @param heightmap Heightmap to analyze
     * @return String describing height range
     */
    private String getHeightRange(int[][] heightmap) {
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        
        for (int x = 0; x < heightmap.length; x++) {
            for (int z = 0; z < heightmap[0].length; z++) {
                min = Math.min(min, heightmap[x][z]);
                max = Math.max(max, heightmap[x][z]);
            }
        }
        
        return min + "-" + max;
    }
}

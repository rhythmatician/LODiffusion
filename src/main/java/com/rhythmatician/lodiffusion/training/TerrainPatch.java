package com.rhythmatician.lodiffusion.training;

/**
 * Represents a single 8x8 terrain patch for machine learning training.
 * Contains heightmap data, biome information, and metadata about the patch location.
 */
public class TerrainPatch {
    
    public static final int PATCH_SIZE = 8;
    
    private final int worldX;
    private final int worldZ;
    private final int[][] heightmap;
    private final String[] biomes;
    
    /**
     * Constructor for TerrainPatch.
     * 
     * @param worldX World X coordinate of the patch (bottom-left corner)
     * @param worldZ World Z coordinate of the patch (bottom-left corner)
     * @param heightmap 8x8 heightmap data
     * @param biomes Biome data array (64 entries for 8x8 positions)
     * @throws IllegalArgumentException if data dimensions are incorrect
     */
    public TerrainPatch(int worldX, int worldZ, int[][] heightmap, String[] biomes) {
        validateInputs(heightmap, biomes);
        
        this.worldX = worldX;
        this.worldZ = worldZ;
        this.heightmap = copyHeightmap(heightmap);
        this.biomes = biomes.clone();
    }
    
    /**
     * Validate input data dimensions.
     */
    private static void validateInputs(int[][] heightmap, String[] biomes) {
        if (heightmap == null) {
            throw new IllegalArgumentException("Heightmap cannot be null");
        }
        if (heightmap.length != PATCH_SIZE) {
            throw new IllegalArgumentException("Heightmap must be " + PATCH_SIZE + "x" + PATCH_SIZE + 
                                             ", got " + heightmap.length + " rows");
        }
        for (int i = 0; i < heightmap.length; i++) {
            if (heightmap[i] == null || heightmap[i].length != PATCH_SIZE) {
                throw new IllegalArgumentException("Heightmap row " + i + " must have " + PATCH_SIZE + " columns");
            }
        }
        
        if (biomes == null) {
            throw new IllegalArgumentException("Biomes array cannot be null");
        }
        if (biomes.length != PATCH_SIZE * PATCH_SIZE) {
            throw new IllegalArgumentException("Biomes array must have " + (PATCH_SIZE * PATCH_SIZE) + 
                                             " entries, got " + biomes.length);
        }
    }
    
    /**
     * Create a deep copy of the heightmap to prevent external modification.
     */
    private static int[][] copyHeightmap(int[][] original) {
        int[][] copy = new int[PATCH_SIZE][PATCH_SIZE];
        for (int x = 0; x < PATCH_SIZE; x++) {
            System.arraycopy(original[x], 0, copy[x], 0, PATCH_SIZE);
        }
        return copy;
    }
    
    /**
     * Get the world X coordinate of this patch.
     * @return World X coordinate (bottom-left corner)
     */
    public int getWorldX() {
        return worldX;
    }
    
    /**
     * Get the world Z coordinate of this patch.
     * @return World Z coordinate (bottom-left corner)
     */
    public int getWorldZ() {
        return worldZ;
    }
    
    /**
     * Get the heightmap data for this patch.
     * @return 8x8 heightmap array (copy to prevent modification)
     */
    public int[][] getHeightmap() {
        return copyHeightmap(heightmap);
    }
    
    /**
     * Get height at specific coordinates within the patch.
     * @param x Local X coordinate (0-7)
     * @param z Local Z coordinate (0-7)
     * @return Height value at the specified position
     * @throws IndexOutOfBoundsException if coordinates are invalid
     */
    public int getHeightAt(int x, int z) {
        if (x < 0 || x >= PATCH_SIZE || z < 0 || z >= PATCH_SIZE) {
            throw new IndexOutOfBoundsException("Coordinates [" + x + ", " + z + "] out of bounds [0, " + PATCH_SIZE + ")");
        }
        return heightmap[x][z];
    }
    
    /**
     * Get the biome data for this patch.
     * @return Biome array (copy to prevent modification)
     */
    public String[] getBiomes() {
        return biomes.clone();
    }
    
    /**
     * Get biome at specific coordinates within the patch.
     * @param x Local X coordinate (0-7)
     * @param z Local Z coordinate (0-7)
     * @return Biome identifier at the specified position
     * @throws IndexOutOfBoundsException if coordinates are invalid
     */
    public String getBiomeAt(int x, int z) {
        if (x < 0 || x >= PATCH_SIZE || z < 0 || z >= PATCH_SIZE) {
            throw new IndexOutOfBoundsException("Coordinates [" + x + ", " + z + "] out of bounds [0, " + PATCH_SIZE + ")");
        }
        int index = z * PATCH_SIZE + x; // Convert 2D coordinates to 1D array index
        return biomes[index];
    }
    
    /**
     * Calculate minimum height in this patch.
     * @return Minimum height value
     */
    public int getMinHeight() {
        int min = Integer.MAX_VALUE;
        for (int x = 0; x < PATCH_SIZE; x++) {
            for (int z = 0; z < PATCH_SIZE; z++) {
                min = Math.min(min, heightmap[x][z]);
            }
        }
        return min;
    }
    
    /**
     * Calculate maximum height in this patch.
     * @return Maximum height value
     */
    public int getMaxHeight() {
        int max = Integer.MIN_VALUE;
        for (int x = 0; x < PATCH_SIZE; x++) {
            for (int z = 0; z < PATCH_SIZE; z++) {
                max = Math.max(max, heightmap[x][z]);
            }
        }
        return max;
    }
    
    /**
     * Calculate height variation (max - min) in this patch.
     * @return Height variation value
     */
    public int getHeightVariation() {
        return getMaxHeight() - getMinHeight();
    }
    
    /**
     * Calculate average height in this patch.
     * @return Average height value
     */
    public double getAverageHeight() {
        int sum = 0;
        for (int x = 0; x < PATCH_SIZE; x++) {
            for (int z = 0; z < PATCH_SIZE; z++) {
                sum += heightmap[x][z];
            }
        }
        return (double) sum / (PATCH_SIZE * PATCH_SIZE);
    }
    
    /**
     * Get the dominant biome in this patch.
     * @return Most common biome identifier
     */
    public String getDominantBiome() {
        // Count biome occurrences
        java.util.Map<String, Integer> biomeCount = new java.util.HashMap<>();
        for (String biome : biomes) {
            biomeCount.put(biome, biomeCount.getOrDefault(biome, 0) + 1);
        }
        
        // Find most common biome
        String dominant = null;
        int maxCount = 0;
        for (java.util.Map.Entry<String, Integer> entry : biomeCount.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                dominant = entry.getKey();
            }
        }
        
        return dominant;
    }
    
    /**
     * Convert this patch to a flattened array format for ML training.
     * @return Flattened array containing heightmap data (64 floats)
     */
    public float[] toFlattenedArray() {
        float[] flattened = new float[PATCH_SIZE * PATCH_SIZE];
        int index = 0;
        
        for (int z = 0; z < PATCH_SIZE; z++) {
            for (int x = 0; x < PATCH_SIZE; x++) {
                flattened[index++] = (float) heightmap[x][z];
            }
        }
        
        return flattened;
    }
    
    /**
     * Get patch summary information.
     * @return String describing this patch
     */
    @Override
    public String toString() {
        return String.format("TerrainPatch[world=(%d,%d), size=%dx%d, height=%.1f±%d, biome=%s]",
                           worldX, worldZ, PATCH_SIZE, PATCH_SIZE, 
                           getAverageHeight(), getHeightVariation(), getDominantBiome());
    }
    
    /**
     * Check equality based on world coordinates and data content.
     */
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        
        TerrainPatch other = (TerrainPatch) obj;
        return worldX == other.worldX && 
               worldZ == other.worldZ &&
               java.util.Arrays.deepEquals(heightmap, other.heightmap) &&
               java.util.Arrays.equals(biomes, other.biomes);
    }
    
    /**
     * Generate hash code based on coordinates and data.
     */
    @Override
    public int hashCode() {
        int result = Integer.hashCode(worldX);
        result = 31 * result + Integer.hashCode(worldZ);
        result = 31 * result + java.util.Arrays.deepHashCode(heightmap);
        result = 31 * result + java.util.Arrays.hashCode(biomes);
        return result;
    }
}

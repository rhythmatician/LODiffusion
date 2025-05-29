package com.yourname.lodiffusion;

public class DiffusionModel {
  
  /**
   * Run diffusion algorithm on heightmap data.
   * @param heightmap 16x16 heightmap data to process
   * @param biomes Biome data for context
   */
  public void run(int[][] heightmap, String[] biomes) {
    // Basic diffusion implementation - smooth terrain with some noise
    // This is minimal implementation to pass tests, will be enhanced later
    
    // Apply simple diffusion-style smoothing and variation
    for (int x = 1; x < 15; x++) {
      for (int z = 1; z < 15; z++) {
        // Calculate average of neighbors
        int neighbors = heightmap[x-1][z] + heightmap[x+1][z] + 
                       heightmap[x][z-1] + heightmap[x][z+1];
        int average = neighbors / 4;
        
        // Apply diffusion with some variation based on biome
        int variation = getBiomeVariation(biomes, x, z);
        heightmap[x][z] = (heightmap[x][z] + average) / 2 + variation;
      }
    }
  }
  
  /**
   * Get height variation based on biome type.
   * @param biomes Array of biome data
   * @param x X coordinate
   * @param z Z coordinate  
   * @return Height variation to apply
   */
  private int getBiomeVariation(String[] biomes, int x, int z) {
    // Simple biome-based variation with null safety
    int biomeIndex = Math.min(x + z, biomes.length - 1);
    String biome = biomes[biomeIndex];
    
    // Handle null biomes gracefully
    if (biome == null) {
      biome = "minecraft:plains"; // Default fallback
    }
    
    if (biome.contains("mountain")) {
      return (x + z) % 5 - 2; // -2 to 2 variation
    } else if (biome.contains("plains")) {
      return (x + z) % 3 - 1; // -1 to 1 variation
    } else {
      return (x + z) % 4 - 2; // -2 to 1 variation
    }
  }
  
  // TODO: load ONNX via DJL for sophisticated AI-based diffusion
}

package com.rhythmatician.lodiffusion;

public class DiffusionModel {

  // LOD → diffusion pass mappings as documented in project requirements
  private static final int[] LOD_DIFFUSION_PASSES = {
    4,  // LOD 0: 4 passes for highest detail
    3,  // LOD 1: 3 passes for high detail  
    2,  // LOD 2: 2 passes for medium detail
    1,  // LOD 3: 1 pass for low detail
    0   // LOD 4+: No diffusion passes for very low detail
  };

  private static final float[] LOD_NOISE_INTENSITY = {
    1.0f,  // LOD 0: Full noise intensity
    0.8f,  // LOD 1: Reduced noise
    0.5f,  // LOD 2: Half noise
    0.2f,  // LOD 3: Minimal noise
    0.0f   // LOD 4+: No noise
  };

  /**
   * Run diffusion algorithm on heightmap data with tile-aware processing.
   * @param heightmap 16x16 heightmap data to process
   * @param biomes Biome data for context
   */
  public void run(int[][] heightmap, String[] biomes) {
    // Enhanced diffusion implementation with multiple passes and noise
    int passes = LOD_DIFFUSION_PASSES[0]; // Use highest detail passes for standard run
    float noiseIntensity = LOD_NOISE_INTENSITY[0];

    // Apply multiple diffusion passes for better quality
    for (int pass = 0; pass < passes; pass++) {
      for (int x = 1; x < 15; x++) {
        for (int z = 1; z < 15; z++) {
          // Calculate weighted average of neighbors with tile-aware behavior
          int neighbors = heightmap[x-1][z] + heightmap[x+1][z] +
                         heightmap[x][z-1] + heightmap[x][z+1];
          int average = neighbors / 4;

          // Apply progressive diffusion with decreasing intensity per pass
          float passStrength = 1.0f - (pass * 0.3f);
          int variation = Math.round(getBiomeVariation(biomes, x, z) * noiseIntensity * passStrength);
          
          // Tile-aware processing: reduce diffusion at tile boundaries (every 16 blocks)
          float tileEdgeFactor = getTileEdgeFactor(x, z);
          int diffusedValue = Math.round(((heightmap[x][z] + (average * passStrength)) / (1.0f + passStrength)) * tileEdgeFactor);
          
          heightmap[x][z] = diffusedValue + variation;
        }
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

  /**
   * Run diffusion algorithm on multi-channel data.
   * Supports processing height, biome, temperature, and other terrain channels simultaneously.
   * @param channels 3D array of channel data [channel][x][z]
   * @param biomes Biome data for context
   */
  public void run(float[][][] channels, String[] biomes) {
    // Multi-channel diffusion processing
    int numChannels = channels.length;
    int width = channels[0].length;
    int height = channels[0][0].length;

    // Apply diffusion to each channel with appropriate weighting
    for (int c = 0; c < numChannels; c++) {
      float channelWeight = getChannelWeight(c);
      
      for (int x = 1; x < width - 1; x++) {
        for (int z = 1; z < height - 1; z++) {
          // Calculate average of neighbors for this channel
          float neighbors = channels[c][x-1][z] + channels[c][x+1][z] +
                           channels[c][x][z-1] + channels[c][x][z+1];
          float average = neighbors / 4.0f;

          // Apply channel-specific diffusion
          float variation = getChannelVariation(c, biomes, x, z);
          channels[c][x][z] = (channels[c][x][z] + average * channelWeight) / (1.0f + channelWeight) + variation;
        }
      }
    }

    // Apply inter-channel correlations (e.g., height affects temperature)
    applyChannelCorrelations(channels, biomes);
  }

  /**
   * Run diffusion algorithm with LOD-aware processing on multi-channel data.
   * Uses predefined LOD mappings for optimal performance at each detail level.
   * @param lod Level of Detail (0 = highest detail, higher = lower detail)
   * @param channels 3D array of channel data [channel][x][z]
   * @param biomes Biome data for context
   */
  public void runWithLOD(int lod, float[][][] channels, String[] biomes) {
    // Get LOD-specific parameters from mappings
    int lodIndex = Math.min(lod, LOD_DIFFUSION_PASSES.length - 1);
    int passes = LOD_DIFFUSION_PASSES[lodIndex];
    float noiseIntensity = LOD_NOISE_INTENSITY[lodIndex];
    
    // Skip processing for very low detail levels
    if (passes == 0) {
      return;
    }
    
    // Apply multiple passes based on LOD level
    for (int pass = 0; pass < passes; pass++) {
      float passStrength = 1.0f - (pass * 0.2f); // Progressive weakening per pass
      
      int numChannels = channels.length;
      int width = channels[0].length;
      int height = channels[0][0].length;

      for (int c = 0; c < numChannels; c++) {
        float channelWeight = getChannelWeight(c) * passStrength;
        
        for (int x = 1; x < width - 1; x++) {
          for (int z = 1; z < height - 1; z++) {
            float neighbors = channels[c][x-1][z] + channels[c][x+1][z] +
                             channels[c][x][z-1] + channels[c][x][z+1];
            float average = neighbors / 4.0f;

            // Apply tile-aware processing for seamless chunk transitions
            float tileEdgeFactor = getTileEdgeFactor(x, z);
            float variation = getChannelVariation(c, biomes, x, z) * noiseIntensity * passStrength;
            
            channels[c][x][z] = (channels[c][x][z] + average * channelWeight * tileEdgeFactor) / (1.0f + channelWeight) + variation;
          }
        }
      }
    }

    // Apply LOD-scaled inter-channel correlations
    applyChannelCorrelations(channels, biomes, noiseIntensity);
  }

  /**
   * Get diffusion weight for a specific channel.
   * @param channelIndex Channel index (0=height, 1=biome, 2=temperature, etc.)
   * @return Weight factor for diffusion intensity
   */
  private float getChannelWeight(int channelIndex) {
    switch (channelIndex) {
      case 0: return 0.5f; // Height channel - moderate diffusion
      case 1: return 0.2f; // Biome channel - light diffusion
      case 2: return 0.3f; // Temperature channel - moderate diffusion
      default: return 0.4f; // Default for other channels
    }
  }

  /**
   * Get LOD factor for scaling diffusion intensity.
   * Uses the predefined LOD noise intensity mappings.
   * @param lod Level of Detail
   * @return Factor to scale diffusion (lower LOD = less processing)
   */
  private float getLODFactor(int lod) {
    int lodIndex = Math.min(lod, LOD_NOISE_INTENSITY.length - 1);
    return LOD_NOISE_INTENSITY[lodIndex];
  }

  /**
   * Get channel-specific variation based on biome and position.
   * @param channelIndex Channel index
   * @param biomes Biome data
   * @param x X coordinate
   * @param z Z coordinate
   * @return Variation to apply
   */
  private float getChannelVariation(int channelIndex, String[] biomes, int x, int z) {
    int biomeIndex = Math.min(x + z, biomes.length - 1);
    String biome = biomes[biomeIndex];
    if (biome == null) biome = "minecraft:plains";

    switch (channelIndex) {
      case 0: // Height channel
        return getBiomeVariation(biomes, x, z) * 0.1f;
      case 1: // Biome channel  
        return (x + z) % 3 * 0.05f - 0.05f; // Small biome variation
      case 2: // Temperature channel
        if (biome.contains("mountain")) {
          return -0.1f; // Cooler in mountains
        } else if (biome.contains("desert")) {
          return 0.15f; // Hotter in deserts
        }
        return 0.0f;
      default:
        return (x + z) % 5 * 0.02f - 0.04f; // Generic small variation
    }
  }

  /**
   * Calculate tile edge factor for tile-aware processing.
   * Reduces diffusion near chunk/tile boundaries to prevent artifacts.
   * @param x X coordinate within chunk
   * @param z Z coordinate within chunk
   * @return Factor from 0.5 to 1.0 (lower near edges)
   */
  private float getTileEdgeFactor(int x, int z) {
    // Reduce diffusion strength near chunk boundaries (first/last 2 blocks)
    float edgeDistance = Math.min(Math.min(x, 15 - x), Math.min(z, 15 - z));
    if (edgeDistance <= 1) {
      return 0.6f; // Reduced diffusion at edges
    } else if (edgeDistance <= 2) {
      return 0.8f; // Slightly reduced diffusion near edges
    }
    return 1.0f; // Full diffusion in center
  }

  /**
   * Apply correlations between different channels.
   * @param channels Multi-channel data
   * @param biomes Biome data
   */
  private void applyChannelCorrelations(float[][][] channels, String[] biomes) {
    applyChannelCorrelations(channels, biomes, 1.0f);
  }

  /**
   * Apply correlations between different channels with LOD scaling.
   * @param channels Multi-channel data
   * @param biomes Biome data
   * @param lodFactor Scaling factor based on LOD
   */
  private void applyChannelCorrelations(float[][][] channels, String[] biomes, float lodFactor) {
    if (channels.length < 2) return; // Need at least 2 channels for correlations

    int width = channels[0].length;
    int height = channels[0][0].length;

    // Apply height-temperature correlation (higher = cooler)
    if (channels.length >= 3) {
      for (int x = 1; x < width - 1; x++) {
        for (int z = 1; z < height - 1; z++) {
          float heightNormalized = (channels[0][x][z] - 64.0f) / 100.0f; // Normalize height
          float tempAdjustment = -heightNormalized * 0.1f * lodFactor; // Higher = cooler
          channels[2][x][z] += tempAdjustment;
        }
      }
    }

    // Apply other inter-channel correlations as needed
    // This is where sophisticated AI-based correlations would be applied in the future
  }
}

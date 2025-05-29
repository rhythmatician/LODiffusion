package com.rhythmatician.lodiffusion;

import com.rhythmatician.lodiffusion.dh.DistantHorizonsCompat;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.util.math.ChunkPos;

public class DiffusionChunkGenerator {

  private final DiffusionModel diffusionModel;
  private final LODQuery lodQuery;

  public DiffusionChunkGenerator() {
    this.diffusionModel = new DiffusionModel();
    this.lodQuery = new DefaultLODQuery();
  }

  /**
   * Constructor that allows injection of custom LOD strategy.
   * @param lodQuery The LOD query strategy to use
   */
  public DiffusionChunkGenerator(LODQuery lodQuery) {
    this.diffusionModel = new DiffusionModel();
    this.lodQuery = lodQuery;
  }

  /**
   * Stub method for building surface terrain using diffusion.
   * This will be expanded to integrate with Fabric chunk generation API.
   */
  public void buildSurface() {
    // TODO: Implement diffusion-based surface generation
    // Will integrate with vanilla heightmap and biome data
  }

  /**
   * Build surface terrain for specified chunk coordinates.
   * @param chunkX X coordinate of the chunk
   * @param chunkZ Z coordinate of the chunk
   */
  public void buildSurface(int chunkX, int chunkZ) {
    // TODO: Implement chunk-coordinate specific diffusion generation
    // This will eventually call the main buildSurface with appropriate data
  }

  /**
   * Build surface terrain with provided heightmap and biome data.
   * This is the main method for Fabric integration.
   * @param chunkX X coordinate of the chunk
   * @param chunkZ Z coordinate of the chunk
   * @param heightmap 16x16 heightmap data from vanilla generation
   * @param biomes Biome data for the chunk
   */
  public void buildSurface(int chunkX, int chunkZ, int[][] heightmap, String[] biomes) {
    // Apply basic height modification first (for backwards compatibility)
    for (int x = 0; x < 16; x++) {
      for (int z = 0; z < 16; z++) {
        // Apply basic height modification based on position and chunk coordinates
        int variation = (chunkX + chunkZ + x + z) % 3 - 1; // -1, 0, or 1
        heightmap[x][z] += variation;
      }
    }

    // Now apply sophisticated diffusion processing
    diffusionModel.run(heightmap, biomes);
  }

  /**
   * Build surface terrain with LOD-aware processing.
   * Implements progressive refinement - higher LOD refines lower LOD output.
   * @param chunkX X coordinate of the chunk
   * @param chunkZ Z coordinate of the chunk
   * @param heightmap 16x16 heightmap data to process
   * @param biomes Biome data for the chunk
   * @param lod Level of Detail (0 = highest detail, higher = lower detail)
   */
  public void buildSurfaceWithLOD(int chunkX, int chunkZ, int[][] heightmap, String[] biomes, int lod) {
    // Apply LOD-specific processing based on level
    switch (lod) {
      case 0:
        // Highest detail - apply full diffusion with refinement
        applyBasicModification(chunkX, chunkZ, heightmap);
        diffusionModel.run(heightmap, biomes);
        applyHighDetailRefinement(heightmap, biomes);
        break;
      case 1:
        // Medium detail - standard diffusion
        applyBasicModification(chunkX, chunkZ, heightmap);
        diffusionModel.run(heightmap, biomes);
        break;
      case 2:
        // Lower detail - reduced diffusion intensity
        applyBasicModification(chunkX, chunkZ, heightmap);
        applyReducedDiffusion(heightmap, biomes);
        break;
      default:
        // Very low detail - minimal processing
        applyMinimalProcessing(chunkX, chunkZ, heightmap);
        break;
    }
  }

  /**
   * Build surface terrain with LOD-aware processing.
   * Overload that creates default heightmap and biomes.
   * @param chunkX X coordinate of the chunk
   * @param chunkZ Z coordinate of the chunk
   * @param lod Level of Detail (0 = highest detail, higher = lower detail)
   * @return Generated heightmap data
   */
  public int[][] buildSurfaceWithLOD(int chunkX, int chunkZ, byte lod) {
    // Create default heightmap
    int[][] heightmap = new int[16][16];
    for (int x = 0; x < 16; x++) {
      for (int z = 0; z < 16; z++) {
        heightmap[x][z] = 64; // Sea level default
      }
    }

    // Create default biomes
    String[] biomes = new String[256]; // 16x16 biome array
    for (int i = 0; i < 256; i++) {
      biomes[i] = "plains"; // Default biome
    }

    // Use existing LOD processing
    buildSurfaceWithLOD(chunkX, chunkZ, heightmap, biomes, (int) lod);

    return heightmap;
  }

  /**
   * Apply basic height modification (shared across LOD levels).
   */
  private void applyBasicModification(int chunkX, int chunkZ, int[][] heightmap) {
    for (int x = 0; x < 16; x++) {
      for (int z = 0; z < 16; z++) {
        int variation = (chunkX + chunkZ + x + z) % 3 - 1; // -1, 0, or 1
        heightmap[x][z] += variation;
      }
    }
  }

  /**
   * Apply high detail refinement for LOD 0.
   */
  private void applyHighDetailRefinement(int[][] heightmap, String[] biomes) {
    // Add fine-grained detail for highest LOD
    for (int x = 1; x < 15; x++) {
      for (int z = 1; z < 15; z++) {
        // Apply more pronounced noise for high detail
        int refinement = (x * z + x + z) % 5 - 2; // Increased variation range
        heightmap[x][z] += refinement;

        // Additional micro-detail for corners
        if ((x + z) % 2 == 0) {
          int microDetail = (x + z) % 3 - 1;
          heightmap[x][z] += microDetail;
        }
      }
    }
  }

  /**
   * Apply reduced diffusion for LOD 2.
   */
  private void applyReducedDiffusion(int[][] heightmap, String[] biomes) {
    // Simplified diffusion for lower detail levels with minimal variation
    for (int x = 2; x < 14; x += 2) { // Process every other point
      for (int z = 2; z < 14; z += 2) {
        int neighbors = heightmap[x-1][z] + heightmap[x+1][z] +
                       heightmap[x][z-1] + heightmap[x][z+1];
        // More conservative smoothing to reduce variation
        heightmap[x][z] = (heightmap[x][z] * 3 + neighbors / 4) / 4;
      }
    }
  }

  /**
   * Apply minimal processing for very low LOD.
   */
  private void applyMinimalProcessing(int chunkX, int chunkZ, int[][] heightmap) {
    // Minimal modification for very low detail
    for (int x = 4; x < 12; x += 4) {
      for (int z = 4; z < 12; z += 4) {
        int variation = (chunkX + chunkZ + x + z) % 2;
        heightmap[x][z] += variation;
      }
    }
  }

  /**
   * Get LOD level for a specific chunk position using the configured strategy.
   * @param chunkX X coordinate of the chunk
   * @param chunkZ Z coordinate of the chunk
   * @return LOD level (0 = highest detail, higher = lower detail)
   */
  public int getChunkLOD(int chunkX, int chunkZ) {
    // Use simple distance-based calculation when no player context available
    if (lodQuery instanceof DefaultLODQuery) {
      return ((DefaultLODQuery) lodQuery).getSimpleLOD(chunkX, chunkZ);
    }
    // Fallback to distance from origin
    return lodQuery.getLOD(chunkX, chunkZ, 0, 0);
  }

  /**
   * Get LOD level relative to a specific player position using configured strategy.
   * @param chunkX X coordinate of the chunk
   * @param chunkZ Z coordinate of the chunk
   * @param playerChunkX Player's chunk X position
   * @param playerChunkZ Player's chunk Z position
   * @return LOD level relative to player
   */
  public int getChunkLODRelativeToPlayer(int chunkX, int chunkZ, int playerChunkX, int playerChunkZ) {
    return lodQuery.getLOD(chunkX, chunkZ, playerChunkX, playerChunkZ);
  }

  /**
   * Build surface terrain using smart LOD determination.
   * Automatically determines LOD based on chunk position using the configured strategy.
   * @param chunkX X coordinate of the chunk
   * @param chunkZ Z coordinate of the chunk
   * @param heightmap 16x16 heightmap data to process
   * @param biomes Biome data for the chunk
   */
  public void buildSurfaceWithSmartLOD(int chunkX, int chunkZ, int[][] heightmap, String[] biomes) {
    // Get LOD from the configured strategy (distance-based fallback or DH if available)
    int lod = getChunkLOD(chunkX, chunkZ);

    // Use existing LOD-aware processing
    buildSurfaceWithLOD(chunkX, chunkZ, heightmap, biomes, lod);
  }

  /**
   * Build surface terrain using LODManager integration with multi-channel processing.
   * This is the main method that integrates with Distant Horizons LODManager
   * and uses the enhanced multi-channel DiffusionModel.
   * 
   * @param player The player entity for LOD calculation
   * @param chunkPos Chunk position to process
   * @param heightmap 16x16 heightmap data to process
   * @param biomes Biome data for the chunk
   * @param temperatureData Optional temperature channel data (can be null)
   */
  public void buildSurfaceWithLODManager(ServerPlayerEntity player, ChunkPos chunkPos, 
                                        int[][] heightmap, String[] biomes, 
                                        float[][] temperatureData) {
    // Get LOD from Distant Horizons or fallback calculation
    int lod = DistantHorizonsCompat.getChunkLOD(player, chunkPos);
    
    // Convert heightmap to float array for multi-channel processing
    float[][] heightChannel = convertToFloatArray(heightmap);
    
    // Prepare multi-channel data
    float[][][] channels;
    if (temperatureData != null) {
      // Use 3 channels: height, biome, temperature
      channels = new float[3][16][16];
      channels[0] = heightChannel;
      channels[1] = generateBiomeChannel(biomes);
      channels[2] = temperatureData;
    } else {
      // Use 2 channels: height, biome
      channels = new float[2][16][16];
      channels[0] = heightChannel;
      channels[1] = generateBiomeChannel(biomes);
    }
    
    // Apply basic modification first
    applyBasicModification(chunkPos.x, chunkPos.z, heightmap);
    
    // Use LOD-aware multi-channel diffusion
    diffusionModel.runWithLOD(lod, channels, biomes);
    
    // Convert back to int array and apply LOD-specific refinements
    convertBackToIntArray(channels[0], heightmap);
    
    // Apply additional LOD-specific processing
    applyLODSpecificProcessing(lod, heightmap, biomes);
  }

  /**
   * Build surface terrain using LODManager integration with coordinate-based LOD.
   * Overload for cases where player entity is not available.
   * 
   * @param chunkX Chunk X coordinate
   * @param chunkZ Chunk Z coordinate
   * @param playerChunkX Player's chunk X coordinate
   * @param playerChunkZ Player's chunk Z coordinate
   * @param heightmap 16x16 heightmap data to process
   * @param biomes Biome data for the chunk
   */
  public void buildSurfaceWithLODManager(int chunkX, int chunkZ, int playerChunkX, int playerChunkZ,
                                        int[][] heightmap, String[] biomes) {
    // Get LOD using coordinate-based calculation
    int lod = DistantHorizonsCompat.getChunkLOD(chunkX, chunkZ, playerChunkX, playerChunkZ);
    
    // Convert heightmap to float array for multi-channel processing
    float[][] heightChannel = convertToFloatArray(heightmap);
    float[][] biomeChannel = generateBiomeChannel(biomes);
    
    // Prepare 2-channel data (height, biome)
    float[][][] channels = new float[2][16][16];
    channels[0] = heightChannel;
    channels[1] = biomeChannel;
    
    // Apply basic modification first
    applyBasicModification(chunkX, chunkZ, heightmap);
    
    // Use LOD-aware multi-channel diffusion
    diffusionModel.runWithLOD(lod, channels, biomes);
    
    // Convert back to int array
    convertBackToIntArray(channels[0], heightmap);
    
    // Apply additional LOD-specific processing
    applyLODSpecificProcessing(lod, heightmap, biomes);
  }

  /**
   * Convert int heightmap to float array for multi-channel processing.
   */
  private float[][] convertToFloatArray(int[][] heightmap) {
    if (heightmap.length != 16 || heightmap[0].length != 16) {
      throw new IllegalArgumentException("Heightmap must be a 16x16 array.");
    }
    float[][] result = new float[16][16];
    for (int x = 0; x < 16; x++) {
      for (int z = 0; z < 16; z++) {
        result[x][z] = (float) heightmap[x][z];
      }
    }
    return result;
  }

  /**
   * Convert float heightmap back to int array.
   */
  private void convertBackToIntArray(float[][] floatArray, int[][] heightmap) {
    for (int x = 0; x < 16; x++) {
      for (int z = 0; z < 16; z++) {
        heightmap[x][z] = Math.round(floatArray[x][z]);
      }
    }
  }

  /**
   * Generate biome channel data from biome string array.
   * Maps biome names to numerical values for processing.
   */
  private float[][] generateBiomeChannel(String[] biomes) {
    float[][] biomeChannel = new float[16][16];
    for (int i = 0; i < 256; i++) {
      int x = i % 16;
      int z = i / 16;
      biomeChannel[x][z] = getBiomeValue(biomes[i]);
    }
    return biomeChannel;
  }

  /**
   * Map biome name to numerical value for channel processing.
   */
  private float getBiomeValue(String biome) {
    switch (biome.toLowerCase()) {
      case "ocean": return 0.1f;
      case "plains": return 0.3f;
      case "desert": return 0.5f;
      case "forest": return 0.7f;
      case "mountains": return 0.9f;
      default: return 0.3f; // Default to plains value
    }
  }

  /**
   * Apply LOD-specific processing after multi-channel diffusion.
   */
  private void applyLODSpecificProcessing(int lod, int[][] heightmap, String[] biomes) {
    switch (lod) {
      case 0:
        applyHighDetailRefinement(heightmap, biomes);
        break;
      case 1:
        // Medium detail - no additional processing needed
        break;
      case 2:
        applyReducedDiffusion(heightmap, biomes);
        break;
      default:
        // Low detail - minimal processing only
        break;
    }
  }



  /**
   * Check if advanced LOD features are available.
   * @return true if Distant Horizons is available, false if using fallback
   */
  public boolean isAdvancedLODAvailable() {
    return ModDetection.isDistantHorizonsAvailable();
  }

  /**
   * Get information about the current LOD strategy.
   * @return Description of the LOD strategy being used
   */
  public String getLODStrategyInfo() {
    return ModDetection.getLODStrategyInfo();
  }
}

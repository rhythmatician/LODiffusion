package com.rhythmatician.lodiffusion.dh;

import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.util.math.ChunkPos;

/**
 * Handles compatibility with Distant Horizons.
 * All DH-specific code should be in this layer.
 * 
 * Provides a unified interface for LOD operations that works both with and without DH.
 */
public class DistantHorizonsCompat {

    private static final LODManagerCompat lodManagerCompat = new LODManagerCompat();

    /**
     * Gets the LOD level for a chunk relative to a player.
     * Uses Distant Horizons if available, otherwise falls back to distance calculation.
     *
     * @param player The player to calculate distance from
     * @param chunkPos The chunk position to get LOD for
     * @return LOD level (0 = highest detail, higher = lower detail)
     */
    public static int getChunkLOD(ServerPlayerEntity player, ChunkPos chunkPos) {
        return lodManagerCompat.getChunkLOD(player, chunkPos);
    }

    /**
     * Gets the LOD level for specific chunk coordinates relative to player coordinates.
     *
     * @param chunkX Chunk X coordinate
     * @param chunkZ Chunk Z coordinate
     * @param playerChunkX Player's chunk X coordinate
     * @param playerChunkZ Player's chunk Z coordinate
     * @return LOD level (0 = highest detail, higher = lower detail)
     */
    public static int getChunkLOD(int chunkX, int chunkZ, int playerChunkX, int playerChunkZ) {
        return lodManagerCompat.getChunkLOD(chunkX, chunkZ, playerChunkX, playerChunkZ);
    }

    /**
     * Gets the diffusion factor for a given LOD level.
     * Maps LOD levels to appropriate diffusion intensity for the DiffusionModel.
     *
     * @param lod The LOD level (0 = highest detail, higher = lower detail)
     * @return Diffusion factor between 0.1 and 1.0
     */
    public static float getLODDiffusionFactor(int lod) {
        return lodManagerCompat.getLODDiffusionFactor(lod);
    }

    /**
     * Checks if Distant Horizons integration is available and functional.
     *
     * @return true if DH is loaded and LOD API is accessible
     */
    public static boolean isDistantHorizonsIntegrationAvailable() {
        return lodManagerCompat.isDistantHorizonsIntegrationAvailable();
    }

    /**
     * Gets a description of the current LOD integration status.
     *
     * @return String describing the LOD strategy being used
     */
    public static String getIntegrationStatus() {
        return lodManagerCompat.getIntegrationStatus();
    }    /**
     * Registers the custom LOD generator with Distant Horizons.
     * Uses the LODiffusionDHWorldGenerator for actual DH API integration.
     */
    public static void registerWorldGenerator() {
        LODiffusionDHWorldGenerator generator = LODiffusionDHWorldGenerator.getInstance();
        
        if (isDistantHorizonsIntegrationAvailable()) {
            System.out.println("Registering LODiffusion generator with Distant Horizons");
            boolean success = generator.attemptRegistration();
            if (success) {
                System.out.println("Successfully registered LODiffusion with DH API");
            } else {
                System.out.println("Failed to register with DH API - using fallback LOD calculation");
            }
        } else {
            System.out.println("DH not available - using standalone LOD calculation");
        }
    }

    /**
     * Queries LOD data for a specific chunk.
     * Provides consistent interface whether using DH or fallback calculation.
     *
     * @param chunkX Chunk X coordinate
     * @param chunkZ Chunk Z coordinate
     * @param playerChunkX Player's chunk X coordinate (for distance calculation)
     * @param playerChunkZ Player's chunk Z coordinate (for distance calculation)
     * @return LOD data array containing [lod_level, diffusion_factor]
     */
    public static Object[] getLodData(int chunkX, int chunkZ, int playerChunkX, int playerChunkZ) {
        int lod = getChunkLOD(chunkX, chunkZ, playerChunkX, playerChunkZ);
        float diffusionFactor = getLODDiffusionFactor(lod);
        
        return new Object[]{lod, diffusionFactor};
    }

    /**
     * Legacy method for backward compatibility.
     * @deprecated Use getLodData(int, int, int, int) with player coordinates instead
     */
    @Deprecated
    public static Object[] getLodData(int chunkX, int chunkZ) {
        // Use origin as default player position
        return getLodData(chunkX, chunkZ, 0, 0);
    }
}


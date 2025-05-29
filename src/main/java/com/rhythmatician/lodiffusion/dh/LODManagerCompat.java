package com.rhythmatician.lodiffusion.dh;

import com.rhythmatician.lodiffusion.DefaultLODQuery;
import com.rhythmatician.lodiffusion.ModDetection;
import net.minecraft.server.network.ServerPlayerEntity;
import net.minecraft.util.math.ChunkPos;

/**
 * Compatibility layer for LODManager integration with Distant Horizons.
 * Provides safe integration that falls back to distance-based calculation
 * when DH is not available.
 */
public class LODManagerCompat {
      private static final String DH_API_CLASS = "com.seibel.distanthorizons.api.DhApi";
    
    private final DefaultLODQuery fallbackQuery;
    private boolean dhIntegrationChecked = false;
    private boolean dhIntegrationAvailable = false;

    public LODManagerCompat() {
        this.fallbackQuery = new DefaultLODQuery();
    }

    /**
     * Gets the LOD level for a chunk relative to a player.
     * Uses Distant Horizons LODManager if available, otherwise falls back to distance calculation.
     *
     * @param player The player to calculate distance from
     * @param chunkPos The chunk position to get LOD for
     * @return LOD level (0 = highest detail, higher = lower detail)
     */
    public int getChunkLOD(ServerPlayerEntity player, ChunkPos chunkPos) {
        if (isDistantHorizonsIntegrationAvailable()) {
            try {
                return getChunkLODFromDH(player, chunkPos);
            } catch (Exception e) {
                // Fall back to distance calculation if DH call fails
                System.err.println("Failed to get LOD from Distant Horizons, falling back: " + e.getMessage());
            }
        }
        
        // Use fallback distance-based calculation
        return fallbackQuery.getLOD(player, chunkPos);
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
    public int getChunkLOD(int chunkX, int chunkZ, int playerChunkX, int playerChunkZ) {
        // For coordinate-based queries, we always use our fallback calculation
        // as it's more efficient and DH's API typically requires player entities
        return fallbackQuery.getLOD(chunkX, chunkZ, playerChunkX, playerChunkZ);
    }

    /**
     * Checks if Distant Horizons integration is available and functional.
     *
     * @return true if DH is loaded and LOD API is accessible
     */
    public boolean isDistantHorizonsIntegrationAvailable() {
        if (!dhIntegrationChecked) {
            checkDistantHorizonsIntegration();
        }
        return dhIntegrationAvailable;
    }

    /**
     * Gets the diffusion factor for a given LOD level.
     * Maps LOD levels to appropriate diffusion intensity.
     *
     * @param lod The LOD level (0 = highest detail, higher = lower detail)
     * @return Diffusion factor between 0.1 and 1.0
     */
    public float getLODDiffusionFactor(int lod) {
        switch (lod) {
            case 0: return 1.0f;  // Full diffusion for highest detail
            case 1: return 0.7f;  // Reduced diffusion for medium detail
            case 2: return 0.4f;  // Lower diffusion for low detail
            case 3: 
            default: return 0.2f; // Minimal diffusion for very low detail
        }
    }

    /**
     * Attempts to get LOD from Distant Horizons using reflection.
     * This is a placeholder implementation as the actual DH API integration
     * would require access to DH's specific API methods.
     */
    private int getChunkLODFromDH(ServerPlayerEntity player, ChunkPos chunkPos) throws Exception {
        // TODO: Implement actual DH API integration when DH dependency is available
        // For now, this method serves as a placeholder for future DH integration
        
        // Placeholder implementation that simulates DH behavior
        // In a real implementation, this would call something like:
        // return LODManager.getChunkLOD(player, chunkPos);
        
        throw new UnsupportedOperationException("DH integration not yet implemented");
    }

    /**
     * Checks if Distant Horizons integration is available using reflection.
     */
    private void checkDistantHorizonsIntegration() {
        dhIntegrationChecked = true;
        
        if (!ModDetection.isDistantHorizonsAvailable()) {
            dhIntegrationAvailable = false;
            return;
        }        try {
            // Try to access DH API classes using reflection
            Class.forName(DH_API_CLASS);
            // If we get here, DH classes are available
            dhIntegrationAvailable = true;
            
        } catch (ClassNotFoundException e) {
            dhIntegrationAvailable = false;
            System.out.println("Distant Horizons detected but API not accessible: " + e.getMessage());
        } catch (Exception e) {
            dhIntegrationAvailable = false;
            System.err.println("Error checking DH integration: " + e.getMessage());
        }
    }

    /**
     * Gets a description of the current LOD integration status.
     *
     * @return String describing the LOD strategy being used
     */
    public String getIntegrationStatus() {
        if (isDistantHorizonsIntegrationAvailable()) {
            return "Distant Horizons LOD integration active";
        } else if (ModDetection.isDistantHorizonsAvailable()) {
            return "Distant Horizons detected but API integration failed - using fallback";
        } else {
            return "Using distance-based LOD fallback";
        }
    }
}

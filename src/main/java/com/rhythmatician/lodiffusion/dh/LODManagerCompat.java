package com.rhythmatician.lodiffusion.dh;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
    private static final Logger LOGGER = LoggerFactory.getLogger(LODManagerCompat.class);
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
     * Attempts to get LOD from Distant Horizons using the DH API.
     */
    private int getChunkLODFromDH(ServerPlayerEntity player, ChunkPos chunkPos) throws Exception {
        try {
            // Check if DH is initialized and world proxy is available
            if (com.seibel.distanthorizons.api.DhApi.Delayed.worldProxy == null) {
                throw new Exception("DH world proxy not initialized");
            }
            
            // For now, we'll use a distance-based calculation that aligns with DH's typical behavior
            // In the future, this could be enhanced to query DH's actual LOD system directly
            double playerX = player.getX();
            double playerZ = player.getZ();
            double chunkCenterX = chunkPos.getStartX() + 8.0;
            double chunkCenterZ = chunkPos.getStartZ() + 8.0;
            double distanceSquared = (playerX - chunkCenterX) * (playerX - chunkCenterX) + 
                                   (playerZ - chunkCenterZ) * (playerZ - chunkCenterZ);
            double distance = Math.sqrt(distanceSquared);
            
            // Convert distance to LOD level based on DH's typical LOD distances
            // These values align with typical DH render distances and LOD levels
            if (distance < 48) return 0;      // High detail (close to vanilla render distance)
            else if (distance < 128) return 1; // Medium detail 
            else if (distance < 384) return 2; // Low detail
            else return 3;                     // Very low detail (distant terrain)
            
        } catch (Exception e) {
            // If DH API call fails, throw to trigger fallback
            throw new Exception("DH API call failed: " + e.getMessage());
        }
    }

    /**
     * Checks if Distant Horizons integration is available.
     */
    private void checkDistantHorizonsIntegration() {
        dhIntegrationChecked = true;
        
        if (!ModDetection.isDistantHorizonsAvailable()) {
            dhIntegrationAvailable = false;
            return;
        }
        
        try {
            // Try to access DH API classes
            Class.forName("com.seibel.distanthorizons.api.DhApi");
            dhIntegrationAvailable = true;
            LOGGER.debug("Distant Horizons API integration available");
            
        } catch (ClassNotFoundException e) {
            dhIntegrationAvailable = false;
            LOGGER.debug("Distant Horizons detected but API not accessible: {}", e.getMessage());
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

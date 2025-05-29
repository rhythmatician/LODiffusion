package com.rhythmatician.lodiffusion;

import net.fabricmc.loader.api.FabricLoader;

/**
 * Utility class for safely detecting if Distant Horizons mod is available.
 * Uses proper Fabric mod detection instead of dangerous reflection.
 */
public class ModDetection {
    
    private static final String DISTANT_HORIZONS_MOD_ID = "distanthorizons";
    
    /**
     * Checks if Distant Horizons mod is loaded using Fabric's safe mod detection.
     * 
     * @return true if Distant Horizons is available, false otherwise
     */
    public static boolean isDistantHorizonsAvailable() {
        return FabricLoader.getInstance().isModLoaded(DISTANT_HORIZONS_MOD_ID);
    }
    
    /**
     * Gets information about the available LOD strategy.
     * 
     * @return String describing the LOD strategy being used
     */
    public static String getLODStrategyInfo() {
        if (isDistantHorizonsAvailable()) {
            return "Distant Horizons detected - advanced LOD available";
        } else {
            return "Using fallback distance-based LOD calculation";
        }
    }
}
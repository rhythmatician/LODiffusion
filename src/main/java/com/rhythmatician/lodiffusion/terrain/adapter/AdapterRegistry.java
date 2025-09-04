package com.rhythmatician.lodiffusion.terrain.adapter;

import java.util.HashMap;
import java.util.Map;

import com.rhythmatician.lodiffusion.HelloTerrainMod;

/**
 * Registry for ONNX adapters. Manages adapter instances and provides
 * lookup by adapter name (from config.json).
 */
public final class AdapterRegistry {
    
    private static final Map<String, OnnxAdapter> ADAPTERS = new HashMap<>();
    
    static {
        // Register available adapters
        register(new Heightmap8x8Adapter());
        register(new Voxel8x8x8Adapter());
    }
    
    /**
     * Register an adapter by its name.
     */
    private static void register(OnnxAdapter adapter) {
        ADAPTERS.put(adapter.getAdapterName(), adapter);
        HelloTerrainMod.LOGGER.debug("[AdapterRegistry] Registered adapter: {}", adapter.getAdapterName());
    }
    
    /**
     * Get an adapter by name.
     * @param adapterName Name from config (e.g., "heightmap8x8")
     * @return Adapter instance, or null if not found
     */
    public static OnnxAdapter getAdapter(String adapterName) {
        OnnxAdapter adapter = ADAPTERS.get(adapterName);
        if (adapter == null) {
            HelloTerrainMod.LOGGER.warn("[AdapterRegistry] Unknown adapter: {}", adapterName);
        }
        return adapter;
    }
    
    /**
     * Get all registered adapter names.
     * @return Array of available adapter names
     */
    public static String[] getAvailableAdapters() {
        return ADAPTERS.keySet().toArray(new String[0]);
    }
    
    /**
     * Check if an adapter is registered.
     * @param adapterName Adapter name to check
     * @return true if registered, false otherwise
     */
    public static boolean hasAdapter(String adapterName) {
        return ADAPTERS.containsKey(adapterName);
    }
    
    // Prevent instantiation
    private AdapterRegistry() {}
}

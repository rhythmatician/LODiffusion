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
        // Register progressive LOD refinement adapters
        register(new ProgressiveLOD4to3Adapter());  // 1x1x1 → 2x2x2
        register(new ProgressiveLOD3to2Adapter());  // 2x2x2 → 4x4x4
        register(new ProgressiveLOD2to1Adapter());  // 4x4x4 → 8x8x8
        register(new ProgressiveLOD1to0Adapter());  // 8x8x8 → 16x16x16
        register(new Heightmap16x16Adapter());      // Heightmap enhancement (non-progressive)
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

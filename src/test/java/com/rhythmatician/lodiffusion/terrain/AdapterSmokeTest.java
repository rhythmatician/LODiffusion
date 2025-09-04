package com.rhythmatician.lodiffusion.terrain;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import com.rhythmatician.lodiffusion.terrain.adapter.AdapterRegistry;
import com.rhythmatician.lodiffusion.terrain.adapter.Heightmap16x16Adapter;
import com.rhythmatician.lodiffusion.terrain.adapter.OnnxAdapter;
import com.rhythmatician.lodiffusion.terrain.adapter.ProgressiveLOD1to0Adapter;

/**
 * Smoke tests for the ONNX adapter system.
 * Validates basic functionality without requiring actual models or chunks.
 */
@Tag("ci")
public class AdapterSmokeTest {
    
    @Test
    public void testAdapterRegistry() {
        // Test that adapters are registered
        String[] adapters = AdapterRegistry.getAvailableAdapters();
        assertTrue(adapters.length >= 2, "Should have at least 2 adapters registered");
        
        // Test specific adapters
        assertTrue(AdapterRegistry.hasAdapter("heightmap16x16"), "Should have heightmap16x16 adapter");
        assertTrue(AdapterRegistry.hasAdapter("progressive_lod1to0"), "Should have progressive LOD1→0 adapter");
        
        // Test unknown adapter
        assertFalse(AdapterRegistry.hasAdapter("unknown"), "Should not have unknown adapter");
    }
    
    @Test
    public void testHeightmapAdapter() {
        OnnxAdapter adapter = AdapterRegistry.getAdapter("heightmap16x16");
        assertNotNull(adapter, "Heightmap adapter should be available");
        assertTrue(adapter instanceof Heightmap16x16Adapter, "Should be correct adapter type");
        
        // Test properties
        assertEquals("heightmap16x16", adapter.getAdapterName());
        assertArrayEquals(new long[]{1, 1, 16, 16}, adapter.getExpectedInputShape());
        assertArrayEquals(new long[]{1, 1, 16, 16}, adapter.getExpectedOutputShape());
    }
    
    @Test
    public void testProgressiveLODAdapter() {
        OnnxAdapter adapter = AdapterRegistry.getAdapter("progressive_lod1to0");
        assertNotNull(adapter, "Progressive LOD1→0 adapter should be available");
        assertTrue(adapter instanceof ProgressiveLOD1to0Adapter, "Should be correct adapter type");
        
        // Test properties
        assertEquals("progressive_lod1to0", adapter.getAdapterName());
        assertArrayEquals(new long[]{1, 1, 8, 8, 8}, adapter.getExpectedInputShape());
        assertArrayEquals(new long[]{1, 1, 16, 16, 16}, adapter.getExpectedOutputShape());
    }
    
    @Test
    public void testAdapterCompatibility() {
        // Test that adapters can handle basic compatibility checks
        OnnxAdapter heightmapAdapter = AdapterRegistry.getAdapter("heightmap16x16");
        OnnxAdapter progressiveAdapter = AdapterRegistry.getAdapter("progressive_lod1to0");
        
        // These should not throw exceptions (even if they return false without a real model)
        assertDoesNotThrow(() -> heightmapAdapter.isCompatible(null));
        assertDoesNotThrow(() -> progressiveAdapter.isCompatible(null));
    }
}

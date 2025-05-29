package com.rhythmatician.lodiffusion.training;

import com.rhythmatician.lodiffusion.world.ChunkDataExtractor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.DisplayName;

import java.io.File;
import java.io.IOException;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Test integration of real NBT parsing into TerrainPatchDataset.
 * Validates that mock data creation is replaced with actual NBT extraction.
 */
@Tag("ci")
public class TerrainPatchDatasetRealNBTTest {

    private TerrainPatchDataset dataset;

    @BeforeEach
    void setUp() {
        dataset = new TerrainPatchDataset();
    }

    @Test
    @DisplayName("Dataset should use real NBT extraction when world data available")
    void testRealNBTIntegration() {
        // Skip test if no world data available
        if (!ChunkDataExtractor.isWorldDataAvailable()) {
            System.out.println("Skipping real NBT test - no world data available");
            return;
        }

        // Load patches using real NBT data
        int patchCount = dataset.loadFromWorldData();
        assertTrue(patchCount > 0, "Should load some patches from real world data");
        assertTrue(dataset.isLoaded(), "Dataset should be marked as loaded");

        // Verify patches contain real data (not mock patterns)
        List<TerrainPatch> patches = dataset.getAllPatches();
        assertFalse(patches.isEmpty(), "Should have loaded patches");

        // Test first patch for real data characteristics
        TerrainPatch firstPatch = patches.get(0);
        int[][] heightmap = firstPatch.getHeightmap();
        String[] biomes = firstPatch.getBiomes();

        // Verify patch structure
        assertEquals(8, heightmap.length, "Heightmap should be 8x8");
        assertEquals(8, heightmap[0].length, "Heightmap should be 8x8");
        assertEquals(64, biomes.length, "Should have 64 biome entries for 8x8 patch");

        // Verify heights are realistic (not the mock pattern of base 64 ± 10)
        boolean hasRealisticHeights = false;
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                int height = heightmap[x][z];
                assertTrue(height >= 0 && height <= 512,
                    "Height should be within valid range: " + height);

                // Real NBT data should not follow the mock pattern exactly
                if (height != 64) {
                    hasRealisticHeights = true;
                }
            }
        }
        assertTrue(hasRealisticHeights, "Should have realistic height variation from NBT data");

        // Verify biomes are proper identifiers (not just "plains", "forest", "hills")
        boolean hasProperBiomes = false;
        for (String biome : biomes) {
            assertNotNull(biome, "Biome should not be null");
            if (biome.startsWith("minecraft:")) {
                hasProperBiomes = true;
            }
        }
        assertTrue(hasProperBiomes, "Should have proper Minecraft biome identifiers");
    }

    @Test
    @DisplayName("Patch extraction should split 16x16 chunks into 2x2 grid of 8x8 patches")
    void testPatchExtractionFromRealChunk() throws IOException {
        // Skip test if no world data available
        if (!ChunkDataExtractor.isWorldDataAvailable()) {
            System.out.println("Skipping patch extraction test - no world data available");
            return;
        }

        File[] regionFiles = ChunkDataExtractor.getAvailableRegionFiles();
        if (regionFiles.length == 0) {
            System.out.println("No region files available for testing");
            return;
        }

        // Test direct chunk extraction
        File firstRegion = regionFiles[0];
        int[][] chunkHeightmap = ChunkDataExtractor.extractHeightmapFromChunk(firstRegion, 0, 0);
        String[] chunkBiomes = ChunkDataExtractor.extractBiomesFromChunk(firstRegion, 0, 0);

        if (chunkHeightmap != null && chunkBiomes != null) {
            // Verify chunk data structure
            assertEquals(16, chunkHeightmap.length, "Chunk heightmap should be 16x16");
            assertEquals(16, chunkHeightmap[0].length, "Chunk heightmap should be 16x16");
            assertEquals(256, chunkBiomes.length, "Chunk should have 256 biome entries");

            System.out.println("Successfully extracted real chunk data:");
            System.out.println("- Heightmap: 16x16 with heights " +
                chunkHeightmap[0][0] + " to " + chunkHeightmap[15][15]);
            System.out.println("- Biomes: " + chunkBiomes[0] + " (example)");
        }
    }

    @Test
    @DisplayName("Dataset should handle missing or corrupt chunks gracefully")
    void testErrorHandling() {
        // This test should always pass, even without world data
        if (!ChunkDataExtractor.isWorldDataAvailable()) {
            // Without world data, should throw IllegalStateException
            assertThrows(IllegalStateException.class, () -> {
                dataset.loadFromWorldData();
            });
        } else {
            // With world data, should handle errors gracefully
            assertDoesNotThrow(() -> {
                int patches = dataset.loadFromWorldData();
                // Should complete without throwing, even if some chunks fail
                assertTrue(patches >= 0, "Patch count should be non-negative");
            });
        }
    }

    @Test
    @DisplayName("Debug information should be comprehensive")
    void testDebuggingOutput() {
        // This test verifies our debugging output is helpful
        if (!ChunkDataExtractor.isWorldDataAvailable()) {
            System.out.println("World data summary (no data): " +
                ChunkDataExtractor.getWorldDataSummary());
            return;
        }

        System.out.println("World data summary: " + ChunkDataExtractor.getWorldDataSummary());

        int patchCount = dataset.loadFromWorldData();
        System.out.println("Dataset summary after loading: " + dataset.getDatasetSummary());
        System.out.println("Patches loaded: " + patchCount);

        if (patchCount > 0) {
            TerrainPatch sample = dataset.getPatch(0);
            System.out.println("Sample patch coordinates: [" +
                sample.getWorldX() + ", " + sample.getWorldZ() + "]");
            System.out.println("Sample heightmap range: " +
                getHeightmapRange(sample.getHeightmap()));
            System.out.println("Sample biome: " + sample.getBiomes()[0]);
        }
    }

    private String getHeightmapRange(int[][] heightmap) {
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;

        for (int x = 0; x < heightmap.length; x++) {
            for (int z = 0; z < heightmap[0].length; z++) {
                min = Math.min(min, heightmap[x][z]);
                max = Math.max(max, heightmap[x][z]);
            }
        }

        return min + " to " + max;
    }
}

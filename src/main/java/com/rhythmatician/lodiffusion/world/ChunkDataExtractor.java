package com.rhythmatician.lodiffusion.world;

import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Utility class for extracting chunk data from Minecraft world files.
 * This class provides methods to read real world data for testing and training.
 *
 * Future enhancement: Add NBT parsing to extract real heightmaps and biome data.
 */
public class ChunkDataExtractor {
    
    private static final String WORLD_PATH = "test-data";

    /**
     * Check if example world data is available.
     * @return true if world data exists and is readable
     */
    public static boolean isWorldDataAvailable() {
        Path worldPath = Paths.get(WORLD_PATH);
        Path regionPath = worldPath.resolve("region");

        if (!worldPath.toFile().exists() || !regionPath.toFile().exists()) {
            return false;
        }

        File[] regionFiles = regionPath.toFile().listFiles((dir, name) -> name.endsWith(".mca"));
        return regionFiles != null && regionFiles.length > 0;
    }

    /**
     * Get list of available region files for processing.
     * @return Array of region file paths, or empty array if none found
     */
    public static File[] getAvailableRegionFiles() {
        if (!isWorldDataAvailable()) {
            return new File[0];
        }

        Path regionPath = Paths.get(WORLD_PATH, "region");
        File[] regionFiles = regionPath.toFile().listFiles((dir, name) -> name.endsWith(".mca"));
        return regionFiles != null ? regionFiles : new File[0];
    }

    /**
     * Parse region coordinates from filename (e.g. "r.0.1.mca" -> [0, 1]).
     * @param regionFile Region file
     * @return Array containing [regionX, regionZ] coordinates
     * @throws IllegalArgumentException if filename format is invalid
     */
    public static int[] parseRegionCoordinates(File regionFile) {
        String filename = regionFile.getName();
        if (!filename.matches("r\\.-?\\d+\\.-?\\d+\\.mca")) {
            throw new IllegalArgumentException("Invalid region file format: " + filename);
        }

        // Remove "r." prefix and ".mca" suffix
        String coords = filename.substring(2, filename.length() - 4);
        String[] parts = coords.split("\\.");

        if (parts.length != 2) {
            throw new IllegalArgumentException("Invalid coordinate format: " + filename);
        }

        return new int[] { Integer.parseInt(parts[0]), Integer.parseInt(parts[1]) };
    }

    /**
     * Calculate chunk coordinates within a region.
     * Each region contains 32x32 chunks.
     * @param regionX Region X coordinate
     * @param regionZ Region Z coordinate
     * @param localChunkX Local chunk X within region (0-31)
     * @param localChunkZ Local chunk Z within region (0-31)
     * @return Array containing [worldChunkX, worldChunkZ]
     */
    public static int[] getWorldChunkCoordinates(int regionX, int regionZ, int localChunkX, int localChunkZ) {
        if (localChunkX < 0 || localChunkX >= 32 || localChunkZ < 0 || localChunkZ >= 32) {
            throw new IllegalArgumentException("Local chunk coordinates must be 0-31");
        }

        int worldChunkX = regionX * 32 + localChunkX;
        int worldChunkZ = regionZ * 32 + localChunkZ;

        return new int[] { worldChunkX, worldChunkZ };
    }

    /**
     * Placeholder for future NBT parsing implementation.
     * TODO: Implement using NBT library to extract real heightmap data.
     * @param regionFile Region file to parse
     * @param chunkX Chunk X coordinate within region
     * @param chunkZ Chunk Z coordinate within region
     * @return Heightmap data array, or null if not implemented
     */
    public static int[][] extractHeightmapFromChunk(File regionFile, int chunkX, int chunkZ) {
        // TODO: Implement NBT parsing
        // This would:
        // 1. Read the .mca file using NBT library
        // 2. Find the specific chunk at (chunkX, chunkZ)
        // 3. Extract the MOTION_BLOCKING heightmap
        // 4. Return as 16x16 int array

        throw new UnsupportedOperationException("NBT parsing not yet implemented - future enhancement");
    }

    /**
     * Placeholder for future biome data extraction.
     * TODO: Implement biome extraction from chunk NBT data.
     * @param regionFile Region file to parse
     * @param chunkX Chunk X coordinate within region
     * @param chunkZ Chunk Z coordinate within region
     * @return Biome data array, or null if not implemented
     */
    public static String[] extractBiomesFromChunk(File regionFile, int chunkX, int chunkZ) {
        // TODO: Implement biome extraction
        // This would:
        // 1. Read chunk NBT data
        // 2. Extract biome palette and indices
        // 3. Convert to biome identifier strings
        // 4. Return as String array for 16x16 positions

        throw new UnsupportedOperationException("Biome extraction not yet implemented - future enhancement");
    }

    /**
     * Get statistics about available world data.
     * @return Summary string describing available data
     */
    public static String getWorldDataSummary() {
        if (!isWorldDataAvailable()) {
            return "No world data available";
        }

        File[] regionFiles = getAvailableRegionFiles();
        StringBuilder summary = new StringBuilder();
        summary.append("Example world data available:\n");
        summary.append("- Region files: ").append(regionFiles.length).append("\n");
        summary.append("- Total chunks: ~").append(regionFiles.length * 1024).append("\n");
        summary.append("- Coverage: Real Minecraft terrain data\n");
        summary.append("- Use case: Integration testing and algorithm validation");

        return summary.toString();
    }
}

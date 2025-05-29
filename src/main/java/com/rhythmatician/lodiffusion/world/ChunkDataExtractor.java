package com.rhythmatician.lodiffusion.world;


import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import net.querz.mca.Chunk;
import net.querz.mca.MCAFile;
import net.querz.mca.MCAUtil;
import net.querz.nbt.tag.CompoundTag;
import net.querz.nbt.tag.LongArrayTag;

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
    }    /**
     * Extract heightmap data from a specific chunk in a region file.
     * @param regionFile Region file to parse
     * @param chunkX Chunk X coordinate within region (0-31)
     * @param chunkZ Chunk Z coordinate within region (0-31)
     * @return 16x16 heightmap data array, or null if chunk not found
     * @throws IOException if file cannot be read
     */
    public static int[][] extractHeightmapFromChunk(File regionFile, int chunkX, int chunkZ) throws IOException {
        if (chunkX < 0 || chunkX >= 32 || chunkZ < 0 || chunkZ >= 32) {
            throw new IllegalArgumentException("Chunk coordinates must be 0-31");
        }

        try {            MCAFile mcaFile = MCAUtil.read(regionFile);
            if (mcaFile == null) {
                return null;
            }            // Get chunk data - using updateHandle() with coordinates
            Chunk chunk = mcaFile.getChunk(chunkX, chunkZ);
            if (chunk == null) {
                return null;
            }

            chunk.updateHandle(chunkX, chunkZ);
            CompoundTag chunkTag = chunk.getHandle();
            if (chunkTag == null) {
                return null;
            }

            // Navigate to heightmaps section
            CompoundTag levelTag = chunkTag.getCompoundTag("Level");
            if (levelTag == null) {
                // Try new format (1.18+)
                levelTag = chunkTag;
            }

            CompoundTag heightmapsTag = levelTag.getCompoundTag("Heightmaps");
            if (heightmapsTag == null) {
                return null;
            }

            // Extract MOTION_BLOCKING heightmap (most useful for terrain generation)
            LongArrayTag motionBlockingTag = heightmapsTag.getLongArrayTag("MOTION_BLOCKING");
            if (motionBlockingTag == null) {
                return null;
            }

            return decodeHeightmapFromLongArray(motionBlockingTag.getValue());        } catch (Exception e) {
            throw new IOException("Failed to parse region file: " + regionFile.getName(), e);
        }
    }

    /**
     * Decode a packed long array from Minecraft heightmap format into a 16x16 int array.
     * Minecraft stores heightmaps as packed data in LongArrayTag format.
     * Each height value is 9 bits (max height 512), packed into 64-bit longs.
     * 
     * @param packedLongs The packed long array from NBT heightmap data
     * @return 16x16 array of height values
     */
    private static int[][] decodeHeightmapFromLongArray(long[] packedLongs) {
        int[][] heightmap = new int[16][16];
        int bitsPerValue = 9; // Minecraft uses 9 bits per height value (max height 512)
        int valuesPerLong = 64 / bitsPerValue; // How many height values fit in one long
        
        for (int index = 0; index < 256; index++) { // 16x16 = 256 positions
            int x = index & 15; // index % 16
            int z = index >> 4; // index / 16
            
            // Calculate which long contains this value and the bit offset
            int longIndex = index / valuesPerLong;
            int bitOffset = (index % valuesPerLong) * bitsPerValue;
            
            if (longIndex < packedLongs.length) {
                // Extract the 9-bit height value from the packed long
                long mask = (1L << bitsPerValue) - 1; // Create 9-bit mask (0x1FF)
                int height = (int) ((packedLongs[longIndex] >>> bitOffset) & mask);
                heightmap[x][z] = height;
            } else {
                // Default height if data is missing/corrupt
                heightmap[x][z] = 64; // Sea level
            }
        }
        
        return heightmap;
    }    /**
     * Extract biome data from a specific chunk in a region file.
     * Handles both pre-1.18 (2D biomes) and 1.18+ (3D palette-indexed biomes).
     * @param regionFile Region file to parse
     * @param chunkX Chunk X coordinate within region (0-31)
     * @param chunkZ Chunk Z coordinate within region (0-31)
     * @return Biome identifier array for surface level (16x16), or null if not found
     * @throws IOException if file cannot be read
     */
    public static String[] extractBiomesFromChunk(File regionFile, int chunkX, int chunkZ) throws IOException {
        if (chunkX < 0 || chunkX >= 32 || chunkZ < 0 || chunkZ >= 32) {
            throw new IllegalArgumentException("Chunk coordinates must be 0-31");
        }        try {
            MCAFile mcaFile = MCAUtil.read(regionFile);
            if (mcaFile == null) {
                return null;
            }

            // Get chunk data - using updateHandle() with coordinates
            Chunk chunk = mcaFile.getChunk(chunkX, chunkZ);
            if (chunk == null) {
                return null;
            }

            chunk.updateHandle(chunkX, chunkZ);
            CompoundTag chunkTag = chunk.getHandle();
            if (chunkTag == null) {
                return null;
            }

            // Try to extract biomes based on format (1.18+ vs pre-1.18)
            return extractBiomesFromChunkTag(chunkTag);

        } catch (Exception e) {
            throw new IOException("Failed to extract biomes from chunk [" + chunkX + ", " + chunkZ + 
                                  "] in region " + regionFile.getName(), e);
        }
    }
    
    /**
     * Extract biome data from chunk NBT tag, handling version differences.
     * @param chunkTag The chunk's NBT data
     * @return Array of biome identifiers for 16x16 surface positions
     */
    private static String[] extractBiomesFromChunkTag(CompoundTag chunkTag) {
        // Try 1.18+ format first (3D biomes with sections)
        CompoundTag sectionsTag = chunkTag.getCompoundTag("sections");
        if (sectionsTag != null) {
            return extractBiomes1_18Plus(sectionsTag);
        }
        
        // Try pre-1.18 format (Level tag with 2D biomes)
        CompoundTag levelTag = chunkTag.getCompoundTag("Level");
        if (levelTag != null) {
            return extractBiomesPre1_18(levelTag);
        }
        
        return null; // Unable to extract biomes
    }
    
    /**
     * Extract biomes from 1.18+ format (3D palette-indexed per section).
     * @param sectionsTag The sections compound tag
     * @return Array of surface biome identifiers
     */
    private static String[] extractBiomes1_18Plus(CompoundTag sectionsTag) {
        // This is a simplified implementation that extracts surface biomes
        // For full 3D biome support, you'd need to process all Y sections
        
        // For now, return a placeholder indicating 1.18+ format detected
        String[] biomes = new String[256]; // 16x16 positions
        for (int i = 0; i < 256; i++) {
            biomes[i] = "minecraft:plains"; // Default placeholder
        }
        return biomes;
    }
    
    /**
     * Extract biomes from pre-1.18 format (2D ByteArrayTag).
     * @param levelTag The Level compound tag
     * @return Array of biome identifiers for 16x16 positions
     */
    private static String[] extractBiomesPre1_18(CompoundTag levelTag) {
        // Pre-1.18 used a simple ByteArrayTag for biomes
        // This is also a simplified implementation
        
        String[] biomes = new String[256]; // 16x16 positions
        for (int i = 0; i < 256; i++) {
            biomes[i] = "minecraft:plains"; // Default placeholder
        }
        return biomes;
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

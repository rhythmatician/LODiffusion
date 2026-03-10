package com.rhythmatician.lodiffusion;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Test class for DiffusionChunkGenerator following TDD principles.
 * Tests chunk generation with diffusion model integration.
 */
public class DiffusionChunkGeneratorTest {

    private DiffusionChunkGenerator generator;
    private int[][] testHeightmap;
    private String[] testBiomes;

    @BeforeEach
    void setUp() {
        generator = new DiffusionChunkGenerator();
        
        // Create test heightmap (16x16)
        testHeightmap = new int[16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                testHeightmap[x][z] = 64; // Sea level default
            }
        }

        // Create test biomes (256 entries for 16x16)
        testBiomes = new String[256];
        for (int i = 0; i < 256; i++) {
            testBiomes[i] = "plains";
        }
    }

    @Test
    void testConstructor_DefaultLODQuery() {
        DiffusionChunkGenerator defaultGenerator = new DiffusionChunkGenerator();
        assertNotNull(defaultGenerator);
    }

    @Test
    void testConstructor_CustomLODQuery() {
        LODQuery customQuery = new DefaultLODQuery();
        DiffusionChunkGenerator customGenerator = new DiffusionChunkGenerator(customQuery);
        assertNotNull(customGenerator);
    }

    @Test
    void testBuildSurface_NoParameters() {
        // Test basic buildSurface method (currently a stub)
        assertDoesNotThrow(() -> generator.buildSurface());
    }

    @Test
    void testBuildSurface_WithChunkCoordinates() {
        // Test buildSurface with chunk coordinates (currently a stub)
        assertDoesNotThrow(() -> generator.buildSurface(0, 0));
    }

    @Test
    void testBuildSurface_WithHeightmapAndBiomes_ModifiesHeightmap() {
        // Apply surface generation
        generator.buildSurface(0, 0, testHeightmap, testBiomes);
        
        // Verify heightmap was modified
        boolean heightmapChanged = false;
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (testHeightmap[x][z] != 64) {
                    heightmapChanged = true;
                    break;
                }
            }
            if (heightmapChanged) break;
        }
        
        assertTrue(heightmapChanged, "Heightmap should be modified by buildSurface");
    }

    @Test
    void testBuildSurfaceWithLOD_LOD0_HighestDetail() {
        generator.buildSurfaceWithLOD(0, 0, testHeightmap, testBiomes, 0);
        
        // Verify heightmap was processed (LOD 0 includes full diffusion + refinement)
        boolean heightmapChanged = false;
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (testHeightmap[x][z] != 64) {
                    heightmapChanged = true;
                    break;
                }
            }
            if (heightmapChanged) break;
        }
        
        assertTrue(heightmapChanged, "LOD 0 should modify heightmap with full processing");
    }

    @Test
    void testBuildSurfaceWithLOD_LOD2_LowerDetail() {
        generator.buildSurfaceWithLOD(0, 0, testHeightmap, testBiomes, 2);
        
        // Verify heightmap was processed (LOD 2 includes reduced diffusion)
        boolean heightmapChanged = false;
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (testHeightmap[x][z] != 64) {
                    heightmapChanged = true;
                    break;
                }
            }
            if (heightmapChanged) break;
        }
        
        assertTrue(heightmapChanged, "LOD 2 should modify heightmap with reduced processing");
    }

    @Test
    void testBuildSurfaceWithLOD_HighLOD_MinimalProcessing() {
        // Use chunk coordinates that will produce non-zero variation 
        generator.buildSurfaceWithLOD(1, 0, testHeightmap, testBiomes, 999);
        
        // Check that minimal processing was applied
        int expectedHeight = 64 + 1; // original + variation of 1
        assertEquals(expectedHeight, testHeightmap[4][4], 
                    "Position [4][4] should be modified by minimal processing");
    }

    @Test
    void testBuildSurfaceWithLOD_ByteOverload_ReturnsHeightmap() {
        int[][] result = generator.buildSurfaceWithLOD(0, 0, (byte) 1);
        
        assertNotNull(result, "Should return a heightmap");
        assertEquals(16, result.length, "Should return 16x16 heightmap");
        assertEquals(16, result[0].length, "Should return 16x16 heightmap");
    }

    @Test
    void testGetChunkLOD() {
        // Test the getChunkLOD method
        int lod = generator.getChunkLOD(5, 10);
        assertTrue(lod >= 0, "LOD should be non-negative");
        assertTrue(lod <= 3, "LOD should be reasonable value");
    }

    @Test
    void testGetChunkLODRelativeToPlayer() {
        // Test LOD calculation relative to player position
        int lod = generator.getChunkLODRelativeToPlayer(10, 15, 5, 5);
        assertTrue(lod >= 0, "LOD should be non-negative");
        assertTrue(lod <= 3, "LOD should be reasonable value");
    }

    @Test
    void testBuildSurfaceWithSmartLOD() {
        // Test smart LOD surface building
        assertDoesNotThrow(() -> {
            generator.buildSurfaceWithSmartLOD(2, 3, testHeightmap, testBiomes);
        });
        
        // Verify heightmap was processed
        boolean heightmapChanged = false;
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (testHeightmap[x][z] != 64) {
                    heightmapChanged = true;
                    break;
                }
            }
            if (heightmapChanged) break;
        }
        
        assertTrue(heightmapChanged, "Smart LOD should modify heightmap");
    }

    @Test
    void testBuildSurfaceWithLODManager_CoordinateBased() {
        // Test coordinate-based LOD manager integration
        assertDoesNotThrow(() -> {
            generator.buildSurfaceWithLODManager(5, 10, 0, 0, testHeightmap, testBiomes);
        });
        
        // Verify method completed successfully
        assertNotNull(testHeightmap, "Heightmap should remain valid");
    }

    @Test
    void testIsAdvancedLODAvailable() {
        // Test advanced LOD availability check
        boolean isAvailable = generator.isAdvancedLODAvailable();
        // This should return false in test environment without Distant Horizons
        assertFalse(isAvailable, "Advanced LOD should not be available in test environment");
    }

    @Test
    void testGetLODStrategyInfo() {
        // Test LOD strategy info retrieval
        String strategyInfo = generator.getLODStrategyInfo();
        assertNotNull(strategyInfo, "LOD strategy info should not be null");
        assertFalse(strategyInfo.isEmpty(), "LOD strategy info should not be empty");
    }

    // ── Sparse octree tests ──────────────────────────────────────────────────

    @Test
    void testBuildSparseOctree_ReturnsPatch() {
        int patchSize = com.rhythmatician.lodiffusion.training.TerrainPatch.PATCH_SIZE;
        int[][] patch8 = new int[patchSize][patchSize];
        String[] biomes8 = new String[patchSize * patchSize];
        for (int i = 0; i < patchSize * patchSize; i++) biomes8[i] = "plains";
        for (int x = 0; x < patchSize; x++)
            for (int z = 0; z < patchSize; z++)
                patch8[x][z] = 64;

        com.rhythmatician.lodiffusion.training.TerrainPatch result =
                generator.buildSparseOctree(0, 0, patch8, biomes8, 0);

        assertNotNull(result, "buildSparseOctree should return a TerrainPatch");
    }

    @Test
    void testBuildSparseOctree_HasOccupiedVoxels() {
        int patchSize = com.rhythmatician.lodiffusion.training.TerrainPatch.PATCH_SIZE;
        int[][] patch8 = new int[patchSize][patchSize];
        String[] biomes8 = new String[patchSize * patchSize];
        for (int i = 0; i < patchSize * patchSize; i++) biomes8[i] = "forest";
        for (int x = 0; x < patchSize; x++)
            for (int z = 0; z < patchSize; z++)
                patch8[x][z] = 70 + x + z;

        com.rhythmatician.lodiffusion.training.TerrainPatch result =
                generator.buildSparseOctree(1, 2, patch8, biomes8, 1);

        assertFalse(result.getOccupiedVoxels().isEmpty(),
                "Sparse octree patch should have occupied voxels for terrain");
        assertNotNull(result.getOctreeStructure(),
                "Octree structure should be present");
    }

    @Test
    void testBuildSparseOctree_WrongSizeHeightmapThrows() {
        int[][] wrongSize = new int[16][16]; // must be 8×8
        String[] biomes8 = new String[64];
        for (int i = 0; i < 64; i++) biomes8[i] = "plains";

        assertThrows(IllegalArgumentException.class,
                () -> generator.buildSparseOctree(0, 0, wrongSize, biomes8, 0),
                "Should reject a non-8×8 heightmap");
    }

    @Test
    void testBuildSparseOctree_WrongBiomesLengthThrows() {
        int patchSize = com.rhythmatician.lodiffusion.training.TerrainPatch.PATCH_SIZE;
        int[][] patch8 = new int[patchSize][patchSize];
        String[] wrongBiomes = new String[10]; // must be 64
        for (int i = 0; i < 10; i++) wrongBiomes[i] = "plains";

        assertThrows(IllegalArgumentException.class,
                () -> generator.buildSparseOctree(0, 0, patch8, wrongBiomes, 0),
                "Should reject biomes array with wrong length");
    }

    @Test
    void testBuildSparseOctree_MultipleLodsSucceed() {
        int patchSize = com.rhythmatician.lodiffusion.training.TerrainPatch.PATCH_SIZE;
        // Use a varied heightmap so smoothing has a visible effect
        String[] biomes8 = new String[patchSize * patchSize];
        for (int i = 0; i < biomes8.length; i++) biomes8[i] = "plains";

        com.rhythmatician.lodiffusion.training.TerrainPatch p0 =
                generator.buildSparseOctree(0, 0, buildVariedPatch(patchSize), biomes8, 0);
        com.rhythmatician.lodiffusion.training.TerrainPatch p4 =
                generator.buildSparseOctree(0, 0, buildVariedPatch(patchSize), biomes8, 4);

        assertNotNull(p0, "LOD 0 result should not be null");
        assertNotNull(p4, "LOD 4 result should not be null");
        assertTrue(p0.getOccupiedVoxelCount() > 0, "LOD 0 patch should have occupied voxels");
        assertTrue(p4.getOccupiedVoxelCount() > 0, "LOD 4 patch should have occupied voxels");
        int maxVoxels = patchSize * com.rhythmatician.lodiffusion.training.TerrainPatch.OCTREE_SIZE * patchSize;
        assertTrue(p0.getOccupiedVoxelCount() <= maxVoxels, "LOD 0 count must not exceed 8×8×8");
        assertTrue(p4.getOccupiedVoxelCount() <= maxVoxels, "LOD 4 count must not exceed 8×8×8");
    }

    private static int[][] buildVariedPatch(int size) {
        int[][] patch = new int[size][size];
        for (int x = 0; x < size; x++)
            for (int z = 0; z < size; z++)
                patch[x][z] = 64 + x * 4;
        return patch;
    }
}

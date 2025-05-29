package com.rhythmatician.lodiffusion.dh;

import com.seibel.distanthorizons.api.DhApi;
import com.seibel.distanthorizons.api.IDhApi;
import com.seibel.distanthorizons.api.IDhApiWorldGenerator;
import com.seibel.distanthorizons.api.IDhApiWorldProxy;
import com.seibel.distanthorizons.api.IDhApiTerrainDataRepo;
import com.seibel.distanthorizons.api.datatypes.IDhApiVec3f;
import com.seibel.distanthorizons.api.datatypes.DhApiTerrainDataPoint;

/**
 * Handles compatibility with Distant Horizons.
 * All DH-specific code should be in this layer.
 */
public class DistantHorizonsCompat {

    /**
     * Registers the custom LOD generator with Distant Horizons.
     * API: DhApi.get().setWorldGenerator (DH API 2.0.0+)
     */
    public static void registerWorldGenerator() {
        IDhApi dhApi = DhApi.get();
        if (dhApi != null) {
            dhApi.setWorldGenerator(new DiffusionLodGenerator());
        } else {
            // Log an error or handle the case where DH API is not available
            // This should not happen if DH is a required dependency
            System.err.println("Distant Horizons API not found, LODiffusion will not be able to generate LODs for DH.");
        }
    }

    /**
     * Example of how to query LOD data if needed.
     * API: DhApi.get().getTerrainDataRepo (DH API 2.0.0+)
     * API: IDhApiTerrainDataRepo.getAllTerrainDataAtChunkPos (DH API 2.0.0+)
     */
    public static DhApiTerrainDataPoint[] getLodData(int chunkX, int chunkZ) {
        IDhApi dhApi = DhApi.get();
        if (dhApi != null) {
            IDhApiTerrainDataRepo repo = dhApi.getTerrainDataRepo();
            if (repo != null) {
                // This is an example, actual usage might differ
                return repo.getAllTerrainDataAtChunkPos(chunkX, chunkZ);
            }
        }
        return null;
    }
}

/**
 * Custom LOD generator for Distant Horizons.
 */
class DiffusionLodGenerator implements IDhApiWorldGenerator {

    /**
     * Called by Distant Horizons to generate LOD terrain for a specific chunk and granularity.
     * API: IDhApiWorldGenerator.generateLod (DH API 2.0.0+)
     * @param world Proxy to interact with the DH world.
     * @param ctx Generation context containing chunk coordinates and granularity.
     */
    @Override
    public void generateLod(IDhApiWorldProxy world, IDhApiWorldGenerator.GenerationContext ctx) {
        int chunkX = ctx.getChunkX();
        int chunkZ = ctx.getChunkZ();
        float granularity = ctx.getGranularity(); // Represents the size of each data point (e.g., 8.0f for 8x8x8 blocks)

        System.out.println("DH requesting LOD for chunk: (" + chunkX + ", " + chunkZ + ") with granularity: " + granularity);

        // TODO:
        // 1. Extract LOD granularity (already available in ctx.getGranularity())
        // 2. Create a low-res heightmap grid (e.g., 8x8 or 16x16 based on granularity)
        //    - The size of the grid will depend on how many data points fit into a chunk at the given granularity.
        //    - A chunk is 16xH_MAXx16 blocks. If granularity is 8, then a chunk is 2x(H_MAX/8)x2 data points.
        // 3. Call DiffusionModel.run(...) with prior context (if conditioning)
        //    - This will be the core logic for generating terrain data.
        // 4. Convert output to a DhApiTerrainDataPoint[] using DhApiTerrainDataPoint.create(x, y, z, density, blockId, ...)
        //    - x, y, z are world coordinates of the data point.
        //    - density is a value from 0 (air) to 1 (solid).
        //    - blockId is the Minecraft block ID (e.g., "minecraft:stone").

        // Example: Create a flat layer of stone for demonstration
        // The number of points depends on the granularity.
        // For a 16x16 block area (one chunk column), if granularity is G,
        // there are (16/G) points in X and Z.
        int pointsPerAxis = (int) (16 / granularity);
        if (pointsPerAxis <= 0) pointsPerAxis = 1; // Ensure at least one point

        DhApiTerrainDataPoint[] terrainData = new DhApiTerrainDataPoint[pointsPerAxis * pointsPerAxis];
        int index = 0;
        for (int dx = 0; dx < pointsPerAxis; dx++) {
            for (int dz = 0; dz < pointsPerAxis; dz++) {
                // Calculate world coordinates for this data point
                // The ctx.getMinCorner() provides the minimum world coordinates for this LOD generation task.
                // It's not necessarily aligned with chunk boundaries if granularity is large.
                float worldX = ctx.getMinCorner().getX() + dx * granularity;
                float worldZ = ctx.getMinCorner().getZ() + dz * granularity;
                float worldY = 64.0f; // Example: flat terrain at Y=64

                // Density: 1.0f for solid, 0.0f for air.
                // Block ID: e.g., "minecraft:stone"
                terrainData[index++] = DhApiTerrainDataPoint.create(worldX, worldY, worldZ, 1.0f, "minecraft:stone");
            }
        }
        
        // Provide the generated data to Distant Horizons
        world.setTerrainData(terrainData);
    }
}

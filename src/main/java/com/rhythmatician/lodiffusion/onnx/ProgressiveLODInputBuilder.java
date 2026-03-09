package com.rhythmatician.lodiffusion.onnx;

import java.util.HashMap;
import java.util.Map;

import com.rhythmatician.lodiffusion.world.noise.NoiseTap;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import net.minecraft.world.Heightmap;

/**
 * Builds DJL input tensors from NoiseTap cache data for progressive LOD models.
 * Handles the conversion from native API shapes to model-specific input tensors
 * with proper normalization according to the model configuration.
 * 
 * Supports all four models in the progressive pipeline:
 * 0. Init (Bootstrap): Noise → LOD4 (1×1×1)
 * 1. LOD4 → LOD3: 1×1×1 → 2×2×2
 * 2. LOD3 → LOD2: 2×2×2 → 4×4×4
 * 3. LOD2 → LOD1: 4×4×4 → 8×8×8
 *
 * LOD0 is NOT generated — vanilla Minecraft handles full resolution.
 */
public class ProgressiveLODInputBuilder {
    
    private final NDManager manager;
    private final ModelConfig config;
    
    public ProgressiveLODInputBuilder(NDManager manager, ModelConfig config) {
        this.manager = manager;
        this.config = config;
        config.validate(); // Ensure config is valid
    }
    
    /**
     * Build complete input tensor map for a progressive LOD model.
     *
     * @param cache NoiseTap cache with raw data at native API resolutions
     * @param parentPrev Previous LOD level binary occupancy (derived from block_logits argmax), or zeros for first model
     * @param lodLevel LOD level (0=init, 1=lod4→lod3, 2=lod3→lod2, 3=lod2→lod1)
     * @return Map of tensor names to NDArrays ready for model inference
     */
    public Map<String, NDArray> buildInputs(
            NoiseTap.Cache cache,
            NDArray parentPrev, // Previous LOD output or zeros
            int lodLevel
    ) {

        Map<String, NDArray> inputs = new HashMap<>();

        // Parent input only for progressive models (not init model)
        if (lodLevel > 0) {
            inputs.put("x_parent_prev", parentPrev != null ? parentPrev : createZeroParentTensor(lodLevel));
        }

        // Core inputs (always present)
        inputs.put("x_height_planes", createHeightPlanesTensor(cache));
        inputs.put("x_biome_quart", createBiomeQuartTensor(cache));
        inputs.put("x_router6", createRouter6Tensor(cache));
        inputs.put("x_chunk_pos", createChunkPosTensor(cache));
        inputs.put("x_lod", createLodTensor(lodLevel));

        return inputs;
    }
    
    /**
     * Create zero parent tensor for the appropriate input resolution.
     */
    private NDArray createZeroParentTensor(int lodLevel) {
        // For Init model (lodLevel 0), input is always [1,1,1,1,1] zeros
        if (lodLevel == 0) {
            return manager.zeros(new Shape(1, 1, 1, 1, 1));
        }

        // For other models, input resolution depends on previous stage output
        int inputRes = switch (lodLevel) {
            case 1 -> 1;  // LOD4→LOD3: takes 1³ from Init
            case 2 -> 2;  // LOD3→LOD2: takes 2³ from LOD4→LOD3
            case 3 -> 4;  // LOD2→LOD1: takes 4³ from LOD3→LOD2
            default -> throw new IllegalArgumentException("Invalid LOD level (0-3): " + lodLevel);
        };

        return manager.zeros(new Shape(1, 1, inputRes, inputRes, inputRes));
    }
    
    /**
     * Create height planes tensor: [1,5,1,16,16]
     * Channels: surface, ocean_floor, slope_x, slope_z, curvature
     */
    private NDArray createHeightPlanesTensor(NoiseTap.Cache cache) {
        float[][][][][] heightPlanes = new float[1][5][1][16][16];

        // Channel 0: Surface height (normalized)
        short[][] surface = cache.getHeightmap(Heightmap.Type.WORLD_SURFACE_WG);
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                heightPlanes[0][0][0][x][z] = config.normalization().heights().normalize(surface[x][z]);
            }
        }
        
        // Channel 1: Ocean floor height (normalized)
        short[][] oceanFloor = cache.getHeightmap(Heightmap.Type.OCEAN_FLOOR_WG);
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                heightPlanes[0][1][0][x][z] = config.normalization().heights().normalize(oceanFloor[x][z]);
            }
        }
        
        // Channel 2: Slope X (computed derivative)
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                float slopeX = computeSlopeX(surface, x, z);
                heightPlanes[0][2][0][x][z] = slopeX;
            }
        }
        
        // Channel 3: Slope Z (computed derivative)
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                float slopeZ = computeSlopeZ(surface, x, z);
                heightPlanes[0][3][0][x][z] = slopeZ;
            }
        }

        // Channel 4: Curvature (computed Laplacian)
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                float curvature = computeCurvature(surface, x, z);
                heightPlanes[0][4][0][x][z] = curvature;
            }
        }

        // Flatten and create NDArray with correct shape
        float[] flatData = flattenArray5D(heightPlanes);
        return manager.create(flatData, new Shape(1, 5, 1, 16, 16));
    }
    
    /**
     * Create biome quart tensor: [1,6,4,4,4]
     * Channels: temp, precip_onehot[3], isCold, downfall
     */
    private NDArray createBiomeQuartTensor(NoiseTap.Cache cache) {
        float[][][][][] biomeQuart = new float[1][6][4][4][4];
        
        int[][][] biomes = cache.biomes4();
        
        for (int qx = 0; qx < 4; qx++) {
            for (int qz = 0; qz < 4; qz++) {
                for (int qy = 0; qy < 4; qy++) {
                    int biomeId = biomes[qx][qz][qy];
                    
                    // For now, use simplified biome encoding
                    // TODO: Implement proper biome registry lookup for real temperature/precipitation
                    
                    // Channel 0: Temperature (normalized to [0,1])
                    biomeQuart[0][0][qx][qz][qy] = (biomeId % 10) / 10.0f;

                    // Channels 1-3: Precipitation one-hot [none, rain, snow]
                    int precipType = (biomeId / 10) % 3;
                    biomeQuart[0][1][qx][qz][qy] = precipType == 0 ? 1.0f : 0.0f; // none
                    biomeQuart[0][2][qx][qz][qy] = precipType == 1 ? 1.0f : 0.0f; // rain
                    biomeQuart[0][3][qx][qz][qy] = precipType == 2 ? 1.0f : 0.0f; // snow

                    // Channel 4: Is cold flag
                    biomeQuart[0][4][qx][qz][qy] = (biomeId % 2 == 0) ? 1.0f : 0.0f;

                    // Channel 5: Downfall (simplified)
                    biomeQuart[0][5][qx][qz][qy] = ((biomeId / 5) % 10) / 10.0f;
                }
            }
        }
        
        // Flatten and create NDArray with correct shape
        float[] flatData = flattenArray5D(biomeQuart);
        return manager.create(flatData, new Shape(1, 6, 4, 4, 4));
    }
    
    /**
     * Create router6 tensor: [1,6,1,16,16]
     * Channels: temperature, vegetation, continents, erosion, depth, ridges
     */
    private NDArray createRouter6Tensor(NoiseTap.Cache cache) {
        float[][][][][] router6 = new float[1][6][1][16][16];
        
        NoiseTap.RouterField[] fields = {
            NoiseTap.RouterField.TEMPERATURE,
            NoiseTap.RouterField.VEGETATION,
            NoiseTap.RouterField.CONTINENTS,
            NoiseTap.RouterField.EROSION,
            NoiseTap.RouterField.DEPTH,
            NoiseTap.RouterField.RIDGES
        };
        
        for (int channel = 0; channel < 6; channel++) {
            float[][][] fieldData = cache.getRouterField(fields[channel]);
            
            for (int x = 0; x < 16; x++) {
                for (int z = 0; z < 16; z++) {
                    // Use Y=0 slice for planar representation (16×16 @ one Y)
                    float rawValue = fieldData[x][z][0];
                    float normalized = config.normalization().router6().normalize(rawValue, channel);
                    router6[0][channel][0][x][z] = normalized;
                }
            }
        }
        
        // Flatten and create NDArray with correct shape
        float[] flatData = flattenArray5D(router6);
        return manager.create(flatData, new Shape(1, 6, 1, 16, 16));
    }
    
    /**
     * Create chunk position tensor: [1,2]
     */
    private NDArray createChunkPosTensor(NoiseTap.Cache cache) {
        float[][] chunkPos = new float[1][2];
        chunkPos[0][0] = (float) cache.chunkX();
        chunkPos[0][1] = (float) cache.chunkZ();
        return manager.create(chunkPos);
    }

    /**
     * Create LOD level tensor: [1,1]
     */
    private NDArray createLodTensor(int lodLevel) {
        long[][] lod = new long[1][1];
        lod[0][0] = lodLevel;
        return manager.create(lod);
    }

    // Helper methods for feature computation

    private float computeSlopeX(short[][] surface, int x, int z) {
        int x1 = Math.max(0, x - 1);
        int x2 = Math.min(15, x + 1);
        return (surface[x2][z] - surface[x1][z]) / (2.0f * (x2 - x1));
    }
    
    private float computeSlopeZ(short[][] surface, int x, int z) {
        int z1 = Math.max(0, z - 1);
        int z2 = Math.min(15, z + 1);
        return (surface[x][z2] - surface[x][z1]) / (2.0f * (z2 - z1));
    }

    private float computeCurvature(short[][] surface, int x, int z) {
        // Simplified discrete Laplacian
        int center = surface[x][z];
        int sum = 0;
        int count = 0;
        
        for (int dx = -1; dx <= 1; dx++) {
            for (int dz = -1; dz <= 1; dz++) {
                if (dx == 0 && dz == 0) continue;
                int nx = x + dx;
                int nz = z + dz;
                if (nx >= 0 && nx < 16 && nz >= 0 && nz < 16) {
                    sum += surface[nx][nz];
                    count++;
                }
            }
        }
        
        return count > 0 ? (sum / (float)count - center) : 0.0f;
    }

    /**
     * Flatten a 5D array to 1D array for DJL NDArray creation.
     * Order: [batch][channel][depth][height][width]
     */
    private float[] flattenArray5D(float[][][][][] array) {
        int batch = array.length;
        int channels = array[0].length;
        int depth = array[0][0].length;
        int height = array[0][0][0].length;
        int width = array[0][0][0][0].length;
        
        float[] flattened = new float[batch * channels * depth * height * width];
        int idx = 0;
        
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < channels; c++) {
                for (int d = 0; d < depth; d++) {
                    for (int h = 0; h < height; h++) {
                        for (int w = 0; w < width; w++) {
                            flattened[idx++] = array[b][c][d][h][w];
                        }
                    }
                }
            }
        }
        
        return flattened;
    }
}

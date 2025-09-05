package com.rhythmatician.lodiffusion;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.Logger;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

/**
 * ONNX-based terrain generator for LODiffusion.
 * Implements the LODiffusion v1 contract for 8x8x8 -> 16x16x16 terrain generation.
 * 
 * Contract:
 * - Input: x_parent [1,1,8,8,8], x_biome [1,256,8,8,1], x_height [1,1,8,8,1], x_lod [1,1]
 * - Output: block_logits [1,1104,16,16,16], air_mask [1,1,16,16,16]
 * - Opset 17, static shapes, no dynamic ops
 * 
 * Note: Now includes actual DJL integration for ONNX inference.
 * Falls back to stub implementation if model loading fails.
 */
public class OnnxTerrainGenerator implements AutoCloseable {
    
    private static final Logger LOGGER = Logger.getLogger(OnnxTerrainGenerator.class.getName());
    private static volatile OnnxTerrainGenerator instance;
    
    // Progressive model paths (updated for new onnx_export models)
    private static final String PROGRESSIVE_MODEL_BASE = "onnx_export/flexible_unet3d_";
    private static final String DEFAULT_MODEL_PATH = "artifacts/chunk_16x16/model.onnx"; // Legacy fallback
    
    // Progressive LOD model paths
    private static final String MODEL_LOD4TO3 = PROGRESSIVE_MODEL_BASE + "lod4to3.onnx";
    private static final String MODEL_LOD3TO2 = PROGRESSIVE_MODEL_BASE + "lod3to2.onnx"; 
    private static final String MODEL_LOD2TO1 = PROGRESSIVE_MODEL_BASE + "lod2to1.onnx";
    private static final String MODEL_LOD1TO0 = PROGRESSIVE_MODEL_BASE + "lod1to0.onnx";
    
    // LODiffusion v1 contract constants
    public static final int MAX_BLOCK_TYPES = 1104;
    public static final int INPUT_SIZE = 8;
    public static final int OUTPUT_SIZE = 16;
    
    private boolean available = false;
    private String modelPath;
    
    // Progressive model management
    private boolean useProgressiveModels = true;
    private volatile Model modelLod4to3;
    private volatile Model modelLod3to2;
    private volatile Model modelLod2to1;
    private volatile Model modelLod1to0;
    
    // Thread-safe model loading with lazy initialization
    private volatile Model model; // Legacy model for fallback
    private final ThreadLocal<Predictor<TerrainInput, TerrainGenerationResult>> predictorTL = 
        ThreadLocal.withInitial(() -> {
            try {
                return getModel().newPredictor(new TerrainTranslator());
            } catch (Exception e) {
                LOGGER.warning("Failed to create predictor: " + e.getMessage());
                return null;
            }
        });
    
    /**
     * Input data for progressive terrain generation.
     */
    public static class ProgressiveTerrainInput {
        public final float[][][][][] parentVoxel;    // Progressive parent voxels [1,1,X,X,X]
        public final float[][][][] biomePatch;       // Biome data [1,256,16,16] 
        public final float[][][][][] heightmapPatch; // Height data [1,1,16,16,1]
        public final float[][][][][] riverPatch;     // River feature data [1,1,16,16,1]
        
        public ProgressiveTerrainInput(float[][][][][] parentVoxel, float[][][][] biomePatch, 
                                     float[][][][][] heightmapPatch, float[][][][][] riverPatch) {
            this.parentVoxel = parentVoxel;
            this.biomePatch = biomePatch;
            this.heightmapPatch = heightmapPatch;
            this.riverPatch = riverPatch;
        }
    }

    /**
     * Input data for terrain generation following LODiffusion v1 contract.
     */
    public static class TerrainInput {
        public final float[][][] parentHeightmap; // [8][8][8] - parent heightmap
        public final float[][][] biomeData;       // [256][8][8] - one-hot biome data
        public final float timestep;             // Diffusion timestep (0.0 for inference)
        public final float[] chunkPos;           // [2] - chunk position
        
        public TerrainInput(float[][][] parentHeightmap, float[][][] biomeData, float timestep, float[] chunkPos) {
            this.parentHeightmap = parentHeightmap;
            this.biomeData = biomeData;
            this.timestep = timestep;
            this.chunkPos = chunkPos;
        }
    }
    
    /**
     * Result of terrain generation following LODiffusion v1 contract.
     */
    public static class TerrainGenerationResult {
        public final float[][][][] blockLogits; // [1104][16][16][16] - logits for each block type
        public final float[][][][] airMask;     // [1][16][16][16] - air/solid mask
        
        public TerrainGenerationResult(float[][][][] blockLogits, float[][][][] airMask) {
            this.blockLogits = blockLogits;
            this.airMask = airMask;
        }
    }
    
    /**
     * Input data for terrain generation using the new 16x16 model contract.
     */
    public static class TerrainInputNew {
        public final float[][][][][] heightTensor;   // [1, 1, 16, 16, 1] - height data
        public final float[][][][][] biomeTensor;    // [1, 256, 16, 16, 1] - biome data
        public final float[][][] parentHeightmap;   // [8][8][8] - parent heightmap for LOD context
        public final float timestep;               // Diffusion timestep
        public final float[] chunkPos;             // [2] - chunk position
        
        public TerrainInputNew(float[][][][][] heightTensor, float[][][][][] biomeTensor, 
                              float[][][] parentHeightmap, float timestep, float[] chunkPos) {
            this.heightTensor = heightTensor;
            this.biomeTensor = biomeTensor;
            this.parentHeightmap = parentHeightmap;
            this.timestep = timestep;
            this.chunkPos = chunkPos;
        }
    }
    
    /**
     * DJL Translator for Progressive LOD models.
     * Converts Java data to NDArrays for new progressive models with updated input names.
     */
    private static class ProgressiveTerrainTranslator implements Translator<ProgressiveTerrainInput, TerrainGenerationResult> {
        
        @Override
        public NDList processInput(TranslatorContext ctx, ProgressiveTerrainInput input) {
            NDManager manager = ctx.getNDManager();
            
            // parent_voxel: [1, 1, X, X, X] - already in correct 5D format
            int[] parentShape = {
                input.parentVoxel.length,
                input.parentVoxel[0].length,
                input.parentVoxel[0][0].length,
                input.parentVoxel[0][0][0].length,
                input.parentVoxel[0][0][0][0].length
            };
            float[] parentFlat = flatten5D(input.parentVoxel);
            // Convert int[] to long[] for DJL Shape constructor
            long[] parentShapeLong = new long[parentShape.length];
            for (int i = 0; i < parentShape.length; i++) {
                parentShapeLong[i] = parentShape[i];
            }
            NDArray parentArray = manager.create(parentFlat).reshape(new Shape(parentShapeLong));
            
            // biome_patch: [1, 256, 16, 16] - already in correct 4D format  
            int[] biomeShape = {
                input.biomePatch.length,
                input.biomePatch[0].length,
                input.biomePatch[0][0].length,
                input.biomePatch[0][0][0].length
            };
            float[] biomeFlat = flatten4D(input.biomePatch);
            // Convert int[] to long[] for DJL Shape constructor
            long[] biomeShapeLong = new long[biomeShape.length];
            for (int i = 0; i < biomeShape.length; i++) {
                biomeShapeLong[i] = biomeShape[i];
            }
            NDArray biomeArray = manager.create(biomeFlat).reshape(new Shape(biomeShapeLong));
            
            // heightmap_patch: [1, 1, 16, 16, 1] - already in correct 5D format
            int[] heightShape = {
                input.heightmapPatch.length,
                input.heightmapPatch[0].length,
                input.heightmapPatch[0][0].length,
                input.heightmapPatch[0][0][0].length,
                input.heightmapPatch[0][0][0][0].length
            };
            float[] heightFlat = flatten5D(input.heightmapPatch);
            // Convert int[] to long[] for DJL Shape constructor
            long[] heightShapeLong = new long[heightShape.length];
            for (int i = 0; i < heightShape.length; i++) {
                heightShapeLong[i] = heightShape[i];
            }
            NDArray heightArray = manager.create(heightFlat).reshape(new Shape(heightShapeLong));
            
            // river_patch: [1, 1, 16, 16, 1] - already in correct 5D format
            float[] riverFlat = flatten5D(input.riverPatch);
            NDArray riverArray = manager.create(riverFlat).reshape(new Shape(heightShapeLong)); // Same shape as height
            
            // Return inputs in order expected by progressive models
            return new NDList(parentArray, biomeArray, heightArray, riverArray);
        }
        
        @Override
        public TerrainGenerationResult processOutput(TranslatorContext ctx, NDList list) {
            // Progressive models output: air_mask_logits, block_type_logits
            NDArray airMaskLogits = list.get(0);
            NDArray blockTypeLogits = list.get(1);
            
            // Convert to Java arrays with correct BCHWD order for terrain generation
            float[][][][] airMask = convertNDArrayTo4D(airMaskLogits);
            float[][][][] blockLogits = convertNDArrayTo4D(blockTypeLogits);
            
            return new TerrainGenerationResult(blockLogits, airMask);
        }
        
        @Override
        public Batchifier getBatchifier() {
            return null; // No batching for terrain generation
        }
        
        // Helper methods for array flattening
        private float[] flatten5D(float[][][][][] array) {
            int size = array.length * array[0].length * array[0][0].length * 
                      array[0][0][0].length * array[0][0][0][0].length;
            float[] result = new float[size];
            int idx = 0;
            for (int i = 0; i < array.length; i++) {
                for (int j = 0; j < array[0].length; j++) {
                    for (int k = 0; k < array[0][0].length; k++) {
                        for (int l = 0; l < array[0][0][0].length; l++) {
                            for (int m = 0; m < array[0][0][0][0].length; m++) {
                                result[idx++] = array[i][j][k][l][m];
                            }
                        }
                    }
                }
            }
            return result;
        }
        
        private float[] flatten4D(float[][][][] array) {
            int size = array.length * array[0].length * array[0][0].length * array[0][0][0].length;
            float[] result = new float[size];
            int idx = 0;
            for (int i = 0; i < array.length; i++) {
                for (int j = 0; j < array[0].length; j++) {
                    for (int k = 0; k < array[0][0].length; k++) {
                        for (int l = 0; l < array[0][0][0].length; l++) {
                            result[idx++] = array[i][j][k][l];
                        }
                    }
                }
            }
            return result;
        }
        
        // Helper method to convert NDArray to 4D Java array
        private float[][][][] convertNDArrayTo4D(NDArray ndArray) {
            long[] shape = ndArray.getShape().getShape();
            if (shape.length != 4) {
                throw new IllegalArgumentException("Expected 4D array, got " + shape.length + "D");
            }
            
            int dim0 = (int) shape[0];
            int dim1 = (int) shape[1]; 
            int dim2 = (int) shape[2];
            int dim3 = (int) shape[3];
            
            float[] flat = ndArray.toFloatArray();
            float[][][][] result = new float[dim0][dim1][dim2][dim3];
            
            int idx = 0;
            for (int i = 0; i < dim0; i++) {
                for (int j = 0; j < dim1; j++) {
                    for (int k = 0; k < dim2; k++) {
                        for (int l = 0; l < dim3; l++) {
                            result[i][j][k][l] = flat[idx++];
                        }
                    }
                }
            }
            return result;
        }
    }

    /**
     * DJL Translator for LODiffusion v1 contract.
     * Converts Java data to NDArrays and back.
     */
    private static class TerrainTranslator implements Translator<TerrainInput, TerrainGenerationResult> {
        
        /**
         * Prepare input arrays for ONNX model.
         * Input order MUST match the trained model's signature:
         * 1. x_parent: [1, 8, 8, 8, 1] - Parent chunk heightmap
         * 2. x_biomes: [1, 256, 8, 8, 1] - One-hot biome encoding  
         * 3. x_height: [1, 1, 8, 8, 1] - Height channel from parent
         * 4. x_lod: [1, 1] - LOD level (always 1 for 8->16 upsampling)
         */
        @Override
        public NDList processInput(TranslatorContext ctx, TerrainInput input) {
            NDManager manager = ctx.getNDManager();
            
            // Flatten 3D arrays to 1D for DJL and create with explicit shapes
            
            // x_parent: [1, 1, 8, 8, 8] - flatten [8][8][8] to [512] then reshape
            float[] parentFlat = new float[8 * 8 * 8];
            int idx = 0;
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    for (int z = 0; z < 8; z++) {
                        parentFlat[idx++] = input.parentHeightmap[x][y][z];
                    }
                }
            }
            NDArray parentArray = manager.create(parentFlat).reshape(new Shape(1, 1, 8, 8, 8));
            
            // x_biome: [1, 256, 8, 8, 1] - flatten [256][8][8] to [16384] then reshape
            float[] biomeFlat = new float[256 * 8 * 8];
            idx = 0;
            for (int b = 0; b < 256; b++) {
                for (int x = 0; x < 8; x++) {
                    for (int z = 0; z < 8; z++) {
                        biomeFlat[idx++] = input.biomeData[b][x][z];
                    }
                }
            }
            NDArray biomeArray = manager.create(biomeFlat).reshape(new Shape(1, 256, 8, 8, 1));
            
            // x_height: [1, 1, 8, 8, 1] - extract height from parent heightmap (use top layer)
            float[] heightFlat = new float[8 * 8];
            int TOP_LAYER_INDEX = input.parentHeightmap[0].length - 1;
            idx = 0;
            for (int x = 0; x < 8; x++) {
                for (int z = 0; z < 8; z++) {
                    heightFlat[idx++] = input.parentHeightmap[x][TOP_LAYER_INDEX][z]; // Use top layer as height
                }
            }
            NDArray heightArray = manager.create(heightFlat).reshape(new Shape(1, 1, 8, 8, 1));
            
            // x_lod: [1, 1] - LOD level (always 1 for 8->16 upsampling)
            NDArray lodArray = manager.create(new float[]{1.0f}).reshape(new Shape(1, 1));
            
            return new NDList(parentArray, biomeArray, heightArray, lodArray);
        }
        
        @Override
        public TerrainGenerationResult processOutput(TranslatorContext ctx, NDList list) {
            // Extract outputs following LODiffusion v1 contract
            // block_logits: [1, 1104, 16, 16, 16]
            // air_mask: [1, 1, 16, 16, 16]
            
            NDArray blockLogitsND = list.get(0);  // block_logits
            NDArray airMaskND = list.get(1);      // air_mask
            
            // Convert NDArrays to flat arrays first
            float[] blockLogitsFlat = blockLogitsND.toFloatArray();
            float[] airMaskFlat = airMaskND.toFloatArray();
            
            // Reshape to expected format [1104][16][16][16] and [1][16][16][16]
            float[][][][] blockLogits = new float[1104][16][16][16];
            float[][][][] airMask = new float[1][16][16][16];
            
            // Convert block logits from flat array [1*1104*16*16*16] to [1104][16][16][16]
            int idx = 0;
            for (int b = 0; b < 1104; b++) {
                for (int x = 0; x < 16; x++) {
                    for (int y = 0; y < 16; y++) {
                        for (int z = 0; z < 16; z++) {
                            blockLogits[b][x][y][z] = blockLogitsFlat[idx++];
                        }
                    }
                }
            }
            
            // Convert air mask from flat array [1*1*16*16*16] to [1][16][16][16]
            idx = 0;
            for (int x = 0; x < 16; x++) {
                for (int y = 0; y < 16; y++) {
                    for (int z = 0; z < 16; z++) {
                        airMask[0][x][y][z] = airMaskFlat[idx++];
                    }
                }
            }
            
            return new TerrainGenerationResult(blockLogits, airMask);
        }
        
        @Override
        public Batchifier getBatchifier() {
            return null; // No batching needed for single predictions
        }
    }
    
    /**
     * Default constructor using default model path.
     */
    /**
     * Get the singleton instance of OnnxTerrainGenerator.
     * Creates a new instance if one doesn't exist.
     * @return the OnnxTerrainGenerator instance
     */
    public static OnnxTerrainGenerator getInstance() {
        if (instance == null) {
            synchronized (OnnxTerrainGenerator.class) {
                if (instance == null) {
                    try {
                        instance = new OnnxTerrainGenerator();
                    } catch (IOException e) {
                        LOGGER.warning("Failed to create OnnxTerrainGenerator instance: " + e.getMessage());
                        return null;
                    }
                }
            }
        }
        return instance;
    }

    public OnnxTerrainGenerator() throws IOException {
        this(DEFAULT_MODEL_PATH);
    }
    
    /**
     * Constructor with custom model path.
     */
    public OnnxTerrainGenerator(String modelPath) throws IOException {
        this.modelPath = modelPath;
        
        // Set the singleton instance
        if (instance == null) {
            instance = this;
        }
        
        // Try to load progressive models first
        try {
            loadProgressiveModels();
        } catch (Exception e) {
            LOGGER.warning("Failed to load progressive models, falling back to legacy: " + e.getMessage());
            useProgressiveModels = false;
        }
        
        // Load legacy model as fallback (only if progressive models aren't available)
        if (!available) {
            loadModel(modelPath);
        }
    }

    /**
     * Get the loaded model, initializing it if necessary (thread-safe).
     */
    private Model getModel() throws IOException {
        Model m = model;
        if (m != null) return m;
        
        synchronized (this) {
            if (model == null) {
                loadModel(modelPath);
            }
            return model;
        }
    }

    /**
     * Load the ONNX model using DJL.
     */
    private void loadModel(String modelPath) throws IOException {
        try {
            // Try multiple path resolutions - model is now included in source code
            Path[] pathsToTry = {
                Paths.get(modelPath),                                    // Relative to current dir
                Paths.get("../" + modelPath),                           // Relative to project root (from run/)
                Paths.get("../../" + modelPath)                         // Alternative relative path
            };
            
            Path validPath = null;
            for (Path path : pathsToTry) {
                if (path.toFile().exists()) {
                    validPath = path;
                    LOGGER.fine("Found model at: " + path.toAbsolutePath());
                    break;
                }
            }
            
            if (validPath == null) {
                LOGGER.warning("Model file not found at any of these locations:");
                for (Path path : pathsToTry) {
                    LOGGER.warning("  - " + path.toAbsolutePath());
                }
                LOGGER.warning("Falling back to stub implementation");
                available = false;
                return;
            }
            
            // Validate it's a reasonable size (should be > 1MB for a real model)
            long fileSize = validPath.toFile().length();
            if (fileSize < 1024 * 1024) {
                LOGGER.warning("Model file seems small (" + fileSize + " bytes). Expected > 1MB for ONNX model - falling back to stub");
                available = false;
                return;
            }
            
            // Load ONNX model with DJL
            model = Model.newInstance("lodiffusion-terrain-generator");
            model.load(validPath);
            
            available = true;
            LOGGER.info("✅ ONNX terrain generator loaded successfully!");
            LOGGER.info("   Model: " + validPath.toAbsolutePath() + " (" + fileSize + " bytes)");
            LOGGER.info("   Contract: LODiffusion v1 (8x8x8 -> 16x16x16)");
            LOGGER.info("   Runtime: DJL with ONNX Runtime backend");
            
        } catch (Exception e) {
            available = false;
            LOGGER.warning("Failed to load ONNX model with DJL: " + e.getMessage() + " - falling back to stub implementation");
            LOGGER.fine("Exception details: " + e.getClass().getSimpleName() + " at " + (e.getStackTrace().length > 0 ? e.getStackTrace()[0] : "unknown location"));
            // Don't throw - fall back to stub implementation
        }
    }
    
    /**
     * Load all progressive LOD models.
     */
    private void loadProgressiveModels() throws IOException {
        if (!useProgressiveModels) {
            LOGGER.info("Progressive models disabled, using legacy model only");
            return;
        }
        
        LOGGER.info("Loading progressive LOD models...");
        
        try {
            modelLod4to3 = loadSingleProgressiveModel(MODEL_LOD4TO3, "LOD4→3");
            modelLod3to2 = loadSingleProgressiveModel(MODEL_LOD3TO2, "LOD3→2");
            modelLod2to1 = loadSingleProgressiveModel(MODEL_LOD2TO1, "LOD2→1");
            modelLod1to0 = loadSingleProgressiveModel(MODEL_LOD1TO0, "LOD1→0");
            
            if (modelLod4to3 != null && modelLod3to2 != null && 
                modelLod2to1 != null && modelLod1to0 != null) {
                LOGGER.info("✅ All progressive LOD models loaded successfully!");
                LOGGER.info("   Progressive refinement: LOD4→LOD3→LOD2→LOD1→LOD0");
                LOGGER.info("   Input contract: parent_voxel, biome_patch, heightmap_patch, river_patch");
                LOGGER.info("   Output contract: air_mask_logits, block_type_logits");
                available = true; // Mark as available since progressive models loaded
            } else {
                LOGGER.warning("Failed to load some progressive models, falling back to legacy");
                useProgressiveModels = false;
            }
            
        } catch (Exception e) {
            LOGGER.warning("Failed to load progressive models: " + e.getMessage());
            useProgressiveModels = false;
        }
    }
    
    /**
     * Load a single progressive model.
     */
    private Model loadSingleProgressiveModel(String modelPath, String lodStage) throws IOException {
        Path[] pathsToTry = {
            Paths.get(modelPath),
            Paths.get("../" + modelPath),
            Paths.get("../../" + modelPath)
        };
        
        for (Path path : pathsToTry) {
            if (path.toFile().exists()) {
                long fileSize = path.toFile().length();
                if (fileSize < 1024 * 1024) {
                    LOGGER.warning(lodStage + " model too small (" + fileSize + " bytes)");
                    continue;
                }
                
                try {
                    Model progressiveModel = Model.newInstance("lodiffusion-" + lodStage.toLowerCase());
                    progressiveModel.load(path);
                    LOGGER.info("   " + lodStage + ": " + path.getFileName() + " (" + fileSize + " bytes)");
                    return progressiveModel;
                } catch (Exception e) {
                    LOGGER.warning("Failed to load " + lodStage + " model: " + e.getMessage());
                    // Continue to try next path
                }
            }
        }
        
        LOGGER.warning(lodStage + " model not found at: " + modelPath);
        return null;
    }
    
    /**
     * Check if the generator is available for use.
     */
    public boolean isAvailable() {
        return available;
    }
    
    /**
     * Generate terrain using progressive LOD refinement (LOD4→LOD3→LOD2→LOD1→LOD0).
     * This is the main entry point for progressive terrain generation.
     * 
     * @param biomeIds 16x16 array of biome IDs
     * @param heightValues 16x16 array of base height values 
     * @param chunkX Chunk X coordinate
     * @param chunkZ Chunk Z coordinate
     * @return Complete 16x16x16 terrain with progressive refinement
     */
    public int[][][] generateProgressiveTerrain(int[][] biomeIds, int[][] heightValues, float chunkX, float chunkZ) {
        if (!useProgressiveModels || !areProgressiveModelsLoaded()) {
            LOGGER.warning("Progressive models not available, falling back to legacy generation");
            return generateTerrainFromHeights(heightValues, biomeIds, 64, chunkX, chunkZ);
        }
        
        try {
            LOGGER.info("🚀 Starting 4-stage progressive LOD terrain generation for chunk (" + chunkX + ", " + chunkZ + ")");
            LOGGER.info("📊 Progressive refinement chain: LOD4(1³)→LOD3(2³)→LOD2(4³)→LOD1(8³)→LOD0(16³)");
            
            // Prepare input data
            float[][][][] biomePatch = createBiomePatch(biomeIds);          // [1,256,16,16]
            float[][][][][] heightmapPatch = createHeightmapPatch(heightValues); // [1,1,16,16,1]
            float[][][][][] riverPatch = createRiverPatch();                // [1,1,16,16,1] (stub for now)
            
            // Stage 1: LOD4→LOD3 (1x1x1 → 2x2x2)
            LOGGER.info("🔄 Stage 1/4: Refining from LOD4 (1³) to LOD3 (2³) for chunk (" + chunkX + ", " + chunkZ + ")");
            float[][][][][] parentVoxel4 = createInitialParentVoxel(); // [1,1,1,1,1]
            ProgressiveTerrainInput input4to3 = new ProgressiveTerrainInput(
                parentVoxel4, biomePatch, heightmapPatch, riverPatch);
            TerrainGenerationResult result3 = generateLodStage(modelLod4to3, input4to3, "LOD4→3");
            
            // Stage 2: LOD3→LOD2 (2x2x2 → 4x4x4)
            LOGGER.info("🔄 Stage 2/4: Refining from LOD3 (2³) to LOD2 (4³) for chunk (" + chunkX + ", " + chunkZ + ")");
            float[][][][][] parentVoxel3 = convertResultToParentVoxel(result3, 2, 2, 2);
            ProgressiveTerrainInput input3to2 = new ProgressiveTerrainInput(
                parentVoxel3, biomePatch, heightmapPatch, riverPatch);
            TerrainGenerationResult result2 = generateLodStage(modelLod3to2, input3to2, "LOD3→2");
            
            // Stage 3: LOD2→LOD1 (4x4x4 → 8x8x8)
            LOGGER.info("🔄 Stage 3/4: Refining from LOD2 (4³) to LOD1 (8³) for chunk (" + chunkX + ", " + chunkZ + ")");
            float[][][][][] parentVoxel2 = convertResultToParentVoxel(result2, 4, 4, 4);
            ProgressiveTerrainInput input2to1 = new ProgressiveTerrainInput(
                parentVoxel2, biomePatch, heightmapPatch, riverPatch);
            TerrainGenerationResult result1 = generateLodStage(modelLod2to1, input2to1, "LOD2→1");
            
            // Stage 4: LOD1→LOD0 (8x8x8 → 16x16x16)
            LOGGER.info("🔄 Stage 4/4: Refining from LOD1 (8³) to LOD0 (16³) for chunk (" + chunkX + ", " + chunkZ + ")");
            float[][][][][] parentVoxel1 = convertResultToParentVoxel(result1, 8, 8, 8);
            ProgressiveTerrainInput input1to0 = new ProgressiveTerrainInput(
                parentVoxel1, biomePatch, heightmapPatch, riverPatch);
            TerrainGenerationResult result0 = generateLodStage(modelLod1to0, input1to0, "LOD1→0");
            
            // Convert final result to block IDs
            int[][][] blockPredictions = extractBlockPredictions(result0.blockLogits);
            int[][][] finalTerrain = applyAirMask(blockPredictions, result0.airMask);
            
            LOGGER.info("🎉 Progressive LOD terrain generation completed successfully for chunk (" + chunkX + ", " + chunkZ + ")");
            return finalTerrain;
            
        } catch (Exception e) {
            LOGGER.warning("Progressive terrain generation failed, falling back to legacy: " + e.getMessage());
            return generateTerrainFromHeights(heightValues, biomeIds, 64, chunkX, chunkZ);
        }
    }
    
    /**
     * Generate a single LOD stage using the appropriate model.
     */
    private TerrainGenerationResult generateLodStage(Model model, ProgressiveTerrainInput input, String stage) {
        try {
            LOGGER.info("🔄 Starting " + stage + " refinement inference...");
            
            Predictor<ProgressiveTerrainInput, TerrainGenerationResult> predictor = 
                model.newPredictor(new ProgressiveTerrainTranslator());
            
            TerrainGenerationResult result = predictor.predict(input);
            predictor.close();
            
            LOGGER.info("✅ " + stage + " refinement completed - output shapes: " + 
                       result.blockLogits.length + "x" + result.blockLogits[0].length + "x" +
                       result.blockLogits[0][0].length + "x" + result.blockLogits[0][0][0].length);
            
            return result;
            
        } catch (Exception e) {
            LOGGER.warning("❌ " + stage + " inference failed: " + e.getMessage());
            throw new RuntimeException("Progressive LOD stage " + stage + " failed", e);
        }
    }
    
    /**
     * Check if all progressive models are loaded.
     */
    private boolean areProgressiveModelsLoaded() {
        return modelLod4to3 != null && modelLod3to2 != null && 
               modelLod2to1 != null && modelLod1to0 != null;
    }

    /**
     * Generate terrain using the new model that accepts 16x16 inputs directly.
     * Uses the chunk_16x16 model contract with proper tensor shapes.
     * 
     * @param heightTensor Height data tensor [1, 1, 16, 16, 1]
     * @param biomeTensor Biome data tensor [1, 256, 16, 16, 1]
     * @param parentHeightmap Parent heightmap [8][8][8] for LOD context
     * @param timestep Diffusion timestep
     * @param chunkPos Chunk position [x, z]
     * @return Terrain generation result with block predictions and air mask
     */
    public TerrainGenerationResult generateTerrainNew(
            float[][][][][] heightTensor,
            float[][][][][] biomeTensor,
            float[][][] parentHeightmap,
            float timestep,
            float[] chunkPos) {
        
        LOGGER.fine("Generating terrain with new model - height tensor: [" + heightTensor.length + 
                   "][" + heightTensor[0].length + "][" + heightTensor[0][0].length + "][" + heightTensor[0][0][0].length + "][" + heightTensor[0][0][0][0].length + "]");
        LOGGER.fine("Biome tensor: [" + biomeTensor.length + "][" + biomeTensor[0].length + "][" + biomeTensor[0][0].length + "][" + biomeTensor[0][0][0].length + "][" + biomeTensor[0][0][0][0].length + "]");
        
        if (available && predictorTL.get() != null) {
            try {
                LOGGER.fine("🧠 Running ONNX inference with new 16x16 model...");
                
                // For now, use the old generateTerrain method as a bridge
                // TODO: Create proper new model translator
                float[][][] biomeData = convertBiomeTensorTo8x8(biomeTensor);
                TerrainGenerationResult result = generateTerrain(parentHeightmap, biomeData, timestep, chunkPos);
                
                LOGGER.fine("✅ New ONNX inference completed successfully");
                return result;
                
            } catch (Exception e) {
                LOGGER.warning("� DJL ONNX inference failed, falling back to stub: " + e.getMessage());
                // Fall through to stub implementation  
            }
        }
        
        // Generate stub terrain as fallback
        // Convert 5D tensors back to 3D for stub generation
        float[][][] biomeData = convertBiomeTensorTo8x8(biomeTensor);
        return generateStubTerrain(parentHeightmap, biomeData, timestep, chunkPos);
    }
    
    /**
     * Generate terrain using the ONNX model.
     * 
     * Uses actual DJL-based ONNX inference if model is loaded,
     * otherwise falls back to stub data with correct LODiffusion v1 contract shapes.
     * 
     * @param parentHeightmap Parent heightmap data [8][8][8] (binary: 0=air, 1=solid)
     * @param biomeData One-hot biome data [256][8][8] (float32, one-hot encoded)
     * @param timestep Diffusion timestep (currently unused, set to 0.0 for 8->16)
     * @param chunkPos Chunk position (currently unused)
     * @return Terrain generation result with correct contract shapes
     */
    public TerrainGenerationResult generateTerrain(
            float[][][] parentHeightmap,
            float[][][] biomeData, 
            float timestep,
            float[] chunkPos) {
        
        // Log the input for debugging
        LOGGER.fine("Generating terrain with parent shape: [" + parentHeightmap.length + 
                   "][" + parentHeightmap[0].length + "][" + parentHeightmap[0][0].length + "]");
        LOGGER.fine("Biome data shape: [" + biomeData.length + "][" + biomeData[0].length + "][" + biomeData[0][0].length + "]");
        
        if (available) {
            // Use actual ONNX inference with thread-safe predictor
            try {
                LOGGER.fine("🧠 Running ONNX inference with DJL...");
                
                Predictor<TerrainInput, TerrainGenerationResult> predictor = predictorTL.get();
                if (predictor == null) {
                    throw new RuntimeException("Failed to get thread-local predictor");
                }
                
                TerrainInput input = new TerrainInput(parentHeightmap, biomeData, timestep, chunkPos);
                TerrainGenerationResult result = predictor.predict(input);
                
                LOGGER.fine("✅ ONNX inference completed successfully");
                LOGGER.fine("Generated terrain with shapes - block_logits: [" + result.blockLogits.length + 
                           "][" + result.blockLogits[0].length + "][" + result.blockLogits[0][0].length + "][" + result.blockLogits[0][0][0].length + 
                           "], air_mask: [" + result.airMask.length + "][" + result.airMask[0].length + "][" + result.airMask[0][0].length + "][" + result.airMask[0][0][0].length + "]");
                
                return result;
                
            } catch (Exception e) {
                LOGGER.warning("� DJL ONNX inference failed, falling back to stub: " + e.getMessage());
                LOGGER.fine("Exception details: " + e.getClass().getSimpleName() + " at " + (e.getStackTrace().length > 0 ? e.getStackTrace()[0] : "unknown location"));
                // Fall through to stub implementation
            }
        }
        
        // Generate stub terrain as fallback
        return generateStubTerrain(parentHeightmap, biomeData, timestep, chunkPos);
    }
    
    /**
     * Generate stub terrain data that follows the LODiffusion v1 contract.
     * Used as fallback when ONNX model is not available.
     */
    private TerrainGenerationResult generateStubTerrain(
            float[][][] parentHeightmap,
            float[][][] biomeData, 
            float timestep,
            float[] chunkPos) {
        
        // Create block logits [1104][16][16][16] 
        float[][][][] blockLogits = new float[1104][16][16][16];
        
        // Create air mask [1][16][16][16]
        float[][][][] airMask = new float[1][16][16][16];
        
        // Generate terrain that resembles the parent at higher resolution
        for (int x = 0; x < 16; x++) {
            for (int y = 0; y < 16; y++) {
                for (int z = 0; z < 16; z++) {
                    // Map 16x16x16 coordinates back to 8x8x8 parent coordinates
                    int parentX = Math.min(7, x / 2);
                    int parentY = Math.min(7, y / 2);
                    int parentZ = Math.min(7, z / 2);
                    
                    // Check if parent has solid block
                    boolean parentSolid = parentHeightmap[parentX][parentY][parentZ] > 0.5f;
                    
                    if (parentSolid) {
                        // Solid area - set air mask and block logits
                        airMask[0][x][y][z] = 1.0f; // Solid
                        
                        // Determine block type based on height and biome
                        if (y < 4) {
                            // Stone at bottom
                            blockLogits[1][x][y][z] = 5.0f; // Stone (block ID 1)
                        } else if (y < 6) {
                            // Dirt in middle
                            blockLogits[3][x][y][z] = 5.0f; // Dirt (block ID 3)
                        } else {
                            // Grass on top
                            blockLogits[2][x][y][z] = 5.0f; // Grass (block ID 2)
                        }
                        
                        // Set low probability for all other blocks
                        for (int b = 0; b < 1104; b++) {
                            if (b != 1 && b != 2 && b != 3) {
                                blockLogits[b][x][y][z] = -5.0f;
                            }
                        }
                    } else {
                        // Air area
                        airMask[0][x][y][z] = 0.0f; // Air
                        
                        // Air block (block ID 0)
                        blockLogits[0][x][y][z] = 5.0f;
                        for (int b = 1; b < 1104; b++) {
                            blockLogits[b][x][y][z] = -5.0f;
                        }
                    }
                }
            }
        }
        
        LOGGER.fine("Generated stub terrain with shapes - block_logits: [" + blockLogits.length + 
                   "][" + blockLogits[0].length + "][" + blockLogits[0][0].length + "][" + blockLogits[0][0][0].length + 
                   "], air_mask: [" + airMask.length + "][" + airMask[0].length + "][" + airMask[0][0].length + "][" + airMask[0][0][0].length + "]");
        
        return new TerrainGenerationResult(blockLogits, airMask);
    }
    
    /**
     * Get the path to the loaded model file.
     */
    public String getModelPath() {
        return modelPath;
    }
    
    // ================================================================================================
    // DATA PREPROCESSING HELPER METHODS
    // ================================================================================================
    
    /**
     * Create a parent heightmap from Minecraft chunk data.
     * Converts block states to binary heightmap format expected by the model.
     * 
     * @param chunkBlocks 3D array of block IDs [16][384][16] (y-axis extended for full height)
     * @param startX Starting X coordinate in the chunk
     * @param startY Starting Y coordinate in the chunk  
     * @param startZ Starting Z coordinate in the chunk
     * @return Parent heightmap [8][8][8] with binary values (0=air, 1=solid)
     */
    public static float[][][] createParentHeightmap(int[][][] chunkBlocks, int startX, int startY, int startZ) {
        float[][][] heightmap = new float[8][8][8];
        
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                for (int z = 0; z < 8; z++) {
                    int chunkX = Math.min(15, startX + x);
                    int chunkY = Math.min(383, startY + y);
                    int chunkZ = Math.min(15, startZ + z);
                    
                    // Convert block ID to binary (0 = air, non-zero = solid)
                    heightmap[x][y][z] = (chunkBlocks[chunkX][chunkY][chunkZ] == 0) ? 0.0f : 1.0f;
                }
            }
        }
        
        return heightmap;
    }
    
    /**
     * Create a simplified parent heightmap from height values only.
     * Useful for terrain generation where only surface height is known.
     * 
     * @param heightValues 2D array of height values [8][8]
     * @param baseY Base Y coordinate to start from
     * @return Parent heightmap [8][8][8] with solid blocks up to height
     */
    public static float[][][] createParentHeightmapFromHeights(int[][] heightValues, int baseY) {
        if (heightValues == null || heightValues.length != 8) {
            throw new IllegalArgumentException("heightValues must be 8x8 array");
        }
        for (int i = 0; i < 8; i++) {
            if (heightValues[i] == null || heightValues[i].length != 8) {
                throw new IllegalArgumentException("heightValues must be 8x8 array");
            }
        }
        
        float[][][] heightmap = new float[8][8][8];
        
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                int surfaceHeight = heightValues[x][z];
                
                for (int y = 0; y < 8; y++) {
                    int worldY = baseY + y;
                    heightmap[x][y][z] = (worldY <= surfaceHeight) ? 1.0f : 0.0f;
                }
            }
        }
        
        return heightmap;
    }

    /**
     * Creates a heightmap tensor for the new model which accepts 16x16 input directly.
     * @param heights 16x16 array of height values 
     * @return Tensor in format [1, 1, 16, 16, 1]
     */
    public static float[][][][][] createHeightmapTensor(float[][] heights) {
        validateInput(heights, 16, 16, "heights");
        
        float[][][][][] heightTensor = new float[1][1][16][16][1];
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                heightTensor[0][0][x][z][0] = heights[x][z];
            }
        }
        
        return heightTensor;
    }
    
    /**
     * Create one-hot encoded biome data from biome IDs.
     * Converts Minecraft biome IDs to the one-hot format expected by the model.
     * 
     * @param biomeIds 2D array of biome IDs [8][8]
     * @return One-hot biome data [256][8][8] (supports up to 256 biomes)
     */
    public static float[][][] createBiomeData(int[][] biomeIds) {
        float[][][] biomeData = new float[256][8][8];
        
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                int biomeId = Math.max(0, Math.min(255, biomeIds[x][z])); // Clamp to [0, 255]
                
                // Set one-hot encoding
                for (int b = 0; b < 256; b++) {
                    biomeData[b][x][z] = (b == biomeId) ? 1.0f : 0.0f;
                }
            }
        }
        
        return biomeData;
    }
    
    /**
     * Create uniform biome data for a single biome type.
     * Useful for testing or when generating terrain with a single biome.
     * 
     * @param biomeId The biome ID to use (0-255)
     * @return One-hot biome data [256][8][8] with the specified biome
     */
    public static float[][][] createUniformBiomeData(int biomeId) {
        int[][] uniformBiomes = new int[8][8];
        int clampedBiomeId = Math.max(0, Math.min(255, biomeId));
        
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                uniformBiomes[x][z] = clampedBiomeId;
            }
        }
        
        return createBiomeData(uniformBiomes);
    }

    /**
     * Creates biome tensor for the new model which accepts 16x16 input directly.
     * @param biomeIds 16x16 array of biome IDs
     * @return Tensor in format [1, 256, 16, 16, 1]
     */
    public static float[][][][][] createBiomeTensor(int[][] biomeIds) {
        validateInput(biomeIds, 16, 16, "biomeIds");
        
        float[][][][][] biomeTensor = new float[1][256][16][16][1];
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                int biomeId = Math.max(0, Math.min(255, biomeIds[x][z])); // Clamp to [0, 255]
                
                // Set one-hot encoding
                for (int b = 0; b < 256; b++) {
                    biomeTensor[0][b][x][z][0] = (b == biomeId) ? 1.0f : 0.0f;
                }
            }
        }
        
        return biomeTensor;
    }
    
    /**
     * Extract block predictions from model output.
     * Converts the raw block logits to the most likely block ID for each position.
     * 
     * @param blockLogits Raw model output [1104][16][16][16]
     * @return Block IDs [16][16][16] with the most likely block at each position
     */
    public static int[][][] extractBlockPredictions(float[][][][] blockLogits) {
        int[][][] blocks = new int[16][16][16];
        
        for (int x = 0; x < 16; x++) {
            for (int y = 0; y < 16; y++) {
                for (int z = 0; z < 16; z++) {
                    int bestBlock = 0;
                    float bestLogit = blockLogits[0][x][y][z];
                    
                    // Find block type with highest logit
                    for (int b = 1; b < 1104; b++) {
                        if (blockLogits[b][x][y][z] > bestLogit) {
                            bestLogit = blockLogits[b][x][y][z];
                            bestBlock = b;
                        }
                    }
                    
                    blocks[x][y][z] = bestBlock;
                }
            }
        }
        
        return blocks;
    }
    
    /**
     * Apply air mask to block predictions.
     * Sets blocks to air (ID 0) where the air mask indicates air should be present.
     * 
     * @param blocks Block predictions [16][16][16]
     * @param airMask Air mask [1][16][16][16] (values > 0.5 indicate solid, <= 0.5 indicate air)
     * @return Filtered block predictions with air mask applied
     */
    public static int[][][] applyAirMask(int[][][] blocks, float[][][][] airMask) {
        int[][][] maskedBlocks = new int[16][16][16];
        
        for (int x = 0; x < 16; x++) {
            for (int y = 0; y < 16; y++) {
                for (int z = 0; z < 16; z++) {
                    if (airMask[0][x][y][z] > 0.5f) {
                        // Solid area - keep the predicted block
                        maskedBlocks[x][y][z] = blocks[x][y][z];
                    } else {
                        // Air area - force to air block
                        maskedBlocks[x][y][z] = 0; // Air block ID
                    }
                }
            }
        }
        
        return maskedBlocks;
    }
    
    /**
     * Generate complete terrain from parent data.
     * Convenience method that handles all preprocessing and postprocessing.
     * 
     * @param parentBlocks Parent block data [8][8][8] as block IDs
     * @param biomeIds Biome IDs [8][8]
     * @param chunkX Chunk X coordinate
     * @param chunkZ Chunk Z coordinate
     * @return Final block predictions [16][16][16] ready for placement
     */
    public int[][][] generateCompleteTerrainFromBlocks(int[][][] parentBlocks, int[][] biomeIds, float chunkX, float chunkZ) {
        // Convert parent blocks to heightmap
        float[][][] parentHeightmap = new float[8][8][8];
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                for (int z = 0; z < 8; z++) {
                    parentHeightmap[x][y][z] = (parentBlocks[x][y][z] == 0) ? 0.0f : 1.0f;
                }
            }
        }
        
        // Convert biomes to one-hot encoding
        float[][][] biomeData = createBiomeData(biomeIds);
        
        // Generate terrain using the model
        TerrainGenerationResult result = generateTerrain(
            parentHeightmap, 
            biomeData, 
            0.0f, // Timestep for inference
            new float[]{chunkX, chunkZ}
        );
        
        // Extract final block predictions
        int[][][] blockPredictions = extractBlockPredictions(result.blockLogits);
        return applyAirMask(blockPredictions, result.airMask);
    }
    
    /**
     * Generate terrain from 16x16 height and biome data using the new model.
     * This is the main entry point for terrain generation from Minecraft.
     * Uses the updated model that accepts 16x16 input directly.
     * 
     * @param heightValues Surface heights [16][16]
     * @param biomeIds Biome IDs [16][16]
     * @param baseY Base Y coordinate for height calculation
     * @param chunkX Chunk X coordinate
     * @param chunkZ Chunk Z coordinate
     * @return Final block predictions [16][16][16] ready for placement
     */
    public int[][][] generateTerrainFromHeights(int[][] heightValues, int[][] biomeIds, int baseY, float chunkX, float chunkZ) {
        // Convert int heights to float and normalize
        float[][] normalizedHeights = new float[16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                normalizedHeights[x][z] = (heightValues[x][z] - baseY) / 384.0f; // Normalize to [-1, 1] range
            }
        }

        // Create input tensors for new model  
        float[][][][][] heightTensor = createHeightmapTensor(normalizedHeights);
        float[][][][][] biomeTensor = createBiomeTensor(biomeIds);
        
        // Create parent LOD heightmap (8x8x8) from center of 16x16 input
        int[][] centerHeights = new int[8][8];
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                centerHeights[x][z] = heightValues[x + 4][z + 4]; // Center 8x8
            }
        }
        float[][][] parentHeightmap = createParentHeightmapFromHeights(centerHeights, baseY);
        
        // Generate terrain using the new model contract
        TerrainGenerationResult result = generateTerrainNew(
            heightTensor, 
            biomeTensor, 
            parentHeightmap,
            0.0f, // Timestep for inference  
            new float[]{chunkX, chunkZ}
        );
        
        // Extract final block predictions
        int[][][] blockPredictions = extractBlockPredictions(result.blockLogits);
        return applyAirMask(blockPredictions, result.airMask);
    }
    
    /**
     * Validate input data shapes and ranges.
     * Throws IllegalArgumentException if data doesn't meet LODiffusion v1 contract.
     * 
     * @param parentHeightmap Parent heightmap to validate
     * @param biomeData Biome data to validate
     * @param chunkPos Chunk position to validate
     */
    public static void validateInputData(float[][][] parentHeightmap, float[][][] biomeData, float[] chunkPos) {
        // Validate parent heightmap shape
        if (parentHeightmap.length != 8 || parentHeightmap[0].length != 8 || parentHeightmap[0][0].length != 8) {
            throw new IllegalArgumentException("Parent heightmap must be [8][8][8], got [" + 
                parentHeightmap.length + "][" + parentHeightmap[0].length + "][" + parentHeightmap[0][0].length + "]");
        }
        
        // Validate biome data shape
        if (biomeData.length != 256 || biomeData[0].length != 8 || biomeData[0][0].length != 8) {
            throw new IllegalArgumentException("Biome data must be [256][8][8], got [" + 
                biomeData.length + "][" + biomeData[0].length + "][" + biomeData[0][0].length + "]");
        }
        
        // Validate chunk position
        if (chunkPos.length < 2) {
            throw new IllegalArgumentException("Chunk position must have at least 2 elements, got " + chunkPos.length);
        }
        
        // Validate parent heightmap values
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                for (int z = 0; z < 8; z++) {
                    float value = parentHeightmap[x][y][z];
                    if (value < 0.0f || value > 1.0f) {
                        throw new IllegalArgumentException("Parent heightmap values must be in [0,1] range, found " + value + " at [" + x + "][" + y + "][" + z + "]");
                    }
                }
            }
        }
        
        // Validate biome data is one-hot encoded
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                float sum = 0.0f;
                int activeCount = 0;
                
                for (int b = 0; b < 256; b++) {
                    float value = biomeData[b][x][z];
                    if (value < 0.0f || value > 1.0f) {
                        throw new IllegalArgumentException("Biome data values must be in [0,1] range, found " + value + " at [" + b + "][" + x + "][" + z + "]");
                    }
                    sum += value;
                    if (value > 0.5f) activeCount++;
                }
                
                if (Math.abs(sum - 1.0f) > 0.001f) {
                    throw new IllegalArgumentException("Biome data must be one-hot encoded (sum=1.0), found sum=" + sum + " at position [" + x + "][" + z + "]");
                }
                
                if (activeCount != 1) {
                    throw new IllegalArgumentException("Biome data must have exactly one active value per position, found " + activeCount + " at [" + x + "][" + z + "]");
                }
            }
        }
    }
    
    @Override
    public void close() {
        try {
            // Close all thread-local predictors
            predictorTL.remove();
            LOGGER.fine("Thread-local predictors cleared");
            
            // Close progressive models
            if (modelLod4to3 != null) {
                modelLod4to3.close();
                modelLod4to3 = null;
            }
            if (modelLod3to2 != null) {
                modelLod3to2.close();
                modelLod3to2 = null;
            }
            if (modelLod2to1 != null) {
                modelLod2to1.close();
                modelLod2to1 = null;
            }
            if (modelLod1to0 != null) {
                modelLod1to0.close();
                modelLod1to0 = null;
            }
            
            // Close legacy model
            if (model != null) {
                model.close();
                model = null;
                LOGGER.fine("DJL model closed");
            }
        } catch (Exception e) {
            LOGGER.warning("Error closing DJL resources: " + e.getMessage());
        }
        available = false;
        LOGGER.fine("ONNX terrain generator closed");
    }

    /**
     * Validates input arrays for correct dimensions.
     * @param array The 2D array to validate
     * @param expectedWidth Expected width
     * @param expectedHeight Expected height  
     * @param name Name for error messages
     */
    private static void validateInput(float[][] array, int expectedWidth, int expectedHeight, String name) {
        if (array == null || array.length != expectedWidth) {
            throw new IllegalArgumentException(name + " must be " + expectedWidth + "x" + expectedHeight + " array");
        }
        for (int i = 0; i < expectedWidth; i++) {
            if (array[i] == null || array[i].length != expectedHeight) {
                throw new IllegalArgumentException(name + " must be " + expectedWidth + "x" + expectedHeight + " array");
            }
        }
    }

    /**
     * Validates input arrays for correct dimensions.
     * @param array The 2D array to validate
     * @param expectedWidth Expected width
     * @param expectedHeight Expected height  
     * @param name Name for error messages
     */
    private static void validateInput(int[][] array, int expectedWidth, int expectedHeight, String name) {
        if (array == null || array.length != expectedWidth) {
            throw new IllegalArgumentException(name + " must be " + expectedWidth + "x" + expectedHeight + " array");
        }
        for (int i = 0; i < expectedWidth; i++) {
            if (array[i] == null || array[i].length != expectedHeight) {
                throw new IllegalArgumentException(name + " must be " + expectedWidth + "x" + expectedHeight + " array");
            }
        }
    }

    /**
     * Convert 16x16 biome tensor to 8x8 biome data for compatibility.
     * Downsamples by taking center 8x8 region.
     * @param biomeTensor [1, 256, 16, 16, 1] biome tensor
     * @return [256, 8, 8] biome data
     */
    private static float[][][] convertBiomeTensorTo8x8(float[][][][][] biomeTensor) {
        float[][][] biomeData = new float[256][8][8];
        
        for (int b = 0; b < 256; b++) {
            for (int x = 0; x < 8; x++) {
                for (int z = 0; z < 8; z++) {
                    // Take center 8x8 from 16x16
                    biomeData[b][x][z] = biomeTensor[0][b][x + 4][z + 4][0];
                }
            }
        }
        
        return biomeData;
    }
    
    // ================================================================================================
    // PROGRESSIVE LOD HELPER METHODS
    // ================================================================================================
    
    /**
     * Create biome patch in progressive format [1, 256, 16, 16].
     */
    private static float[][][][] createBiomePatch(int[][] biomeIds) {
        float[][][][] biomePatch = new float[1][256][16][16];
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                int biomeId = Math.max(0, Math.min(255, biomeIds[x][z]));
                biomePatch[0][biomeId][x][z] = 1.0f; // One-hot encoding
            }
        }
        
        return biomePatch;
    }
    
    /**
     * Create heightmap patch in progressive format [1, 1, 16, 16, 1].
     */
    private static float[][][][][] createHeightmapPatch(int[][] heightValues) {
        float[][][][][] heightmapPatch = new float[1][1][16][16][1];
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                // Normalize height to [-1, 1] range
                heightmapPatch[0][0][x][z][0] = (heightValues[x][z] - 64) / 64.0f;
            }
        }
        
        return heightmapPatch;
    }
    
    /**
     * Create river patch in progressive format [1, 1, 16, 16, 1].
     * Currently returns zeros - can be enhanced with actual river data.
     */
    private static float[][][][][] createRiverPatch() {
        return new float[1][1][16][16][1]; // All zeros for now
    }
    
    /**
     * Create initial parent voxel for LOD4 stage [1, 1, 1, 1, 1].
     * Represents the entire subchunk as a single voxel.
     */
    private static float[][][][][] createInitialParentVoxel() {
        float[][][][][] parentVoxel = new float[1][1][1][1][1];
        parentVoxel[0][0][0][0][0] = 0.5f; // Neutral starting value
        return parentVoxel;
    }
    
    /**
     * Convert terrain generation result to parent voxel for next LOD stage.
     * Uses air mask to determine solid/air classification.
     */
    private static float[][][][][] convertResultToParentVoxel(TerrainGenerationResult result, int x, int y, int z) {
        float[][][][][] parentVoxel = new float[1][1][x][y][z];
        
        // Extract air mask and convert to parent voxel representation
        float[][][][] airMask = result.airMask;
        
        for (int i = 0; i < x; i++) {
            for (int j = 0; j < y; j++) {
                for (int k = 0; k < z; k++) {
                    // Use air mask to determine if voxel is solid (1.0) or air (0.0)
                    parentVoxel[0][0][i][j][k] = (airMask[0][i][j][k] > 0.5f) ? 1.0f : 0.0f;
                }
            }
        }
        
        return parentVoxel;
    }
}

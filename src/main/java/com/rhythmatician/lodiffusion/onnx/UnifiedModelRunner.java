package com.rhythmatician.lodiffusion.onnx;

import java.io.IOException;
import java.nio.file.Path;
import java.util.logging.Logger;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;

/**
 * Loads and runs the single unified VoxelTree ONNX model.
 *
 * <p>Implements the <b>lodiffusion.v1</b> contract:
 * <pre>
 *   Inputs:
 *     x_parent  [1, 1, 8, 8, 8]           float32  binary parent occupancy
 *     x_biome   [1, 256, 16, 16, 1]       float32  one-hot biome
 *     x_height  [1, 1, 16, 16, 1]         float32  normalised heightmap [0,1]
 *     x_lod     [1, 1]                    float32  LOD token (1–4)
 *
 *   Outputs:
 *     block_logits  [1, 1104, 16, 16, 16]  float32  raw block-type logits
 *     air_mask      [1, 1, 16, 16, 16]     float32  air probability logits
 * </pre>
 */
public final class UnifiedModelRunner implements AutoCloseable {

    private static final Logger LOGGER = Logger.getLogger(UnifiedModelRunner.class.getName());

    private final NDManager manager;
    private final ZooModel<NDList, NDList> model;
    private final ModelConfig config;
    private final BlockVocabulary vocabulary;

    private UnifiedModelRunner(NDManager manager,
                               ZooModel<NDList, NDList> model,
                               ModelConfig config,
                               BlockVocabulary vocabulary) {
        this.manager = manager;
        this.model = model;
        this.config = config;
        this.vocabulary = vocabulary;
    }

    // ------------------------------------------------------------------
    // Factory
    // ------------------------------------------------------------------

    /**
     * Load the unified ONNX model and its sidecar config.
     *
     * @param onnxPath   path to {@code model.onnx}
     * @param configPath path to {@code model_config.json}
     */
    public static UnifiedModelRunner load(Path onnxPath, Path configPath) throws IOException {
        ModelConfig cfg = ConfigLoader.load(configPath);

        NDManager manager = NDManager.newBaseManager();
        Criteria<NDList, NDList> criteria = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelPath(onnxPath.getParent())  // DJL looks for *.onnx in dir
                .optModelName(stripExtension(onnxPath.getFileName().toString()))
                .optTranslator(new NoopTranslator())
                .build();

        ZooModel<NDList, NDList> zooModel;
        try {
            zooModel = criteria.loadModel();
        } catch (Exception e) {
            manager.close();
            throw new IOException("Failed to load ONNX model from " + onnxPath, e);
        }

        BlockVocabulary vocab = BlockVocabulary.fromConfig(cfg);

        LOGGER.info("UnifiedModelRunner loaded: " + onnxPath
                + "  vocab=" + vocab.size()
                + "  contract=" + cfg.contract());

        return new UnifiedModelRunner(manager, zooModel, cfg, vocab);
    }

    // ------------------------------------------------------------------
    // Inference
    // ------------------------------------------------------------------

    /**
     * Result record returned by {@link #generate}.
     *
     * @param blockLogits raw logits [1, N, 16, 16, 16]
     * @param airMask     air/solid logits [1, 1, 16, 16, 16]
     * @param elapsedMs   wall-clock inference time
     */
    public record InferenceResult(
        float[][][][][] blockLogits,
        float[][][][][] airMask,
        long elapsedMs
    ) {}

    /**
     * Run inference for a single LOD expansion.
     *
     * @param parentOccupancy [8][8][8] binary occupancy of the parent chunk (0=air, 1=solid)
     * @param biomeOneHot     [256][16][16] one-hot biome encoding per column
     * @param heightmap       [16][16] normalised height values in [0, 1]
     * @param lodLevel        LOD token (1–4)
     */
    public InferenceResult generate(float[][][] parentOccupancy,
                                    float[][][] biomeOneHot,
                                    float[][] heightmap,
                                    int lodLevel) throws TranslateException {
        long t0 = System.currentTimeMillis();

        try (NDManager sub = manager.newSubManager()) {
            // x_parent [1, 1, 8, 8, 8]
            NDArray xParent = sub.create(flatten3D(parentOccupancy, 8, 8, 8),
                    new Shape(1, 1, 8, 8, 8));
            xParent.setName("x_parent");

            // x_biome [1, 256, 16, 16, 1]
            NDArray xBiome = sub.create(flatten3D(biomeOneHot, 256, 16, 16),
                    new Shape(1, 256, 16, 16, 1));
            xBiome.setName("x_biome");

            // x_height [1, 1, 16, 16, 1]
            NDArray xHeight = sub.create(flatten2D(heightmap, 16, 16),
                    new Shape(1, 1, 16, 16, 1));
            xHeight.setName("x_height");

            // x_lod [1, 1]
            NDArray xLod = sub.create(new float[]{(float) lodLevel}, new Shape(1, 1));
            xLod.setName("x_lod");

            NDList inputs = new NDList(xParent, xBiome, xHeight, xLod);

            NDList outputs;
            try (var predictor = model.newPredictor(new NoopTranslator())) {
                outputs = predictor.predict(inputs);
            }

            // Identify outputs by channel count
            NDArray blockLogitsArr = null;
            NDArray airMaskArr = null;
            for (NDArray t : outputs) {
                long[] shape = t.getShape().getShape();
                if (shape.length == 5 && shape[0] == 1) {
                    if (shape[1] > 100) blockLogitsArr = t;
                    else if (shape[1] == 1) airMaskArr = t;
                }
            }
            if (blockLogitsArr == null) throw new IllegalStateException("block_logits not found in output");
            if (airMaskArr == null) throw new IllegalStateException("air_mask not found in output");

            float[][][][][] blockLogits = extract5D(blockLogitsArr);
            float[][][][][] airMask = extract5D(airMaskArr);

            long elapsed = System.currentTimeMillis() - t0;
            return new InferenceResult(blockLogits, airMask, elapsed);
        }
    }

    /**
     * Run inference for the v2 anchor-channel LOD contract.
     *
     * <p>Requires a model loaded from a {@code lodiffusion.v2} sidecar config.
     *
     * @param parentOccupancy [8][8][8] binary occupancy of the parent chunk
     * @param heightPlanes    [5][256] row-major height-plane features (surface, ocean_floor,
     *                        slope_x, slope_z, curvature) — from
     *                        {@link com.rhythmatician.lodiffusion.voxy.AnchorSampler}
     * @param router6         [6][256] row-major CORE router values normalised to [0,1]
     * @param biomeIdx        [16][16] integer biome index per column
     * @param yIndex          y-slab index in [0, 23]
     * @param lodLevel        LOD token (1–4)
     */
    public InferenceResult generateV2(float[][][] parentOccupancy,
                                      float[][] heightPlanes,
                                      float[][] router6,
                                      int[][] biomeIdx,
                                      int yIndex,
                                      int lodLevel) throws TranslateException {
        long t0 = System.currentTimeMillis();

        try (NDManager sub = manager.newSubManager()) {
            // x_parent [1,1,8,8,8]
            NDArray xParent = sub.create(flatten3D(parentOccupancy, 8, 8, 8),
                    new Shape(1, 1, 8, 8, 8));
            xParent.setName("x_parent");

            // x_height_planes [1,5,16,16]
            NDArray xHeightPlanes = sub.create(flatten2DChannels(heightPlanes, 5, 256),
                    new Shape(1, 5, 16, 16));
            xHeightPlanes.setName("x_height_planes");

            // x_router6 [1,6,16,16]
            NDArray xRouter6 = sub.create(flatten2DChannels(router6, 6, 256),
                    new Shape(1, 6, 16, 16));
            xRouter6.setName("x_router6");

            // x_biome [1,16,16] int64
            long[] biomeFlat = new long[256];
            for (int x = 0; x < 16; x++)
                for (int z = 0; z < 16; z++)
                    biomeFlat[x * 16 + z] = biomeIdx[x][z];
            NDArray xBiome = sub.create(biomeFlat, new Shape(1, 16, 16));
            xBiome.setName("x_biome");

            // x_y_index [1] int64
            NDArray xYIndex = sub.create(new long[]{yIndex}, new Shape(1));
            xYIndex.setName("x_y_index");

            // x_lod [1] int64
            NDArray xLod = sub.create(new long[]{lodLevel}, new Shape(1));
            xLod.setName("x_lod");

            NDList inputs = new NDList(xParent, xHeightPlanes, xRouter6, xBiome, xYIndex, xLod);

            NDList outputs;
            try (var predictor = model.newPredictor(new NoopTranslator())) {
                outputs = predictor.predict(inputs);
            }

            NDArray blockLogitsArr = null;
            NDArray airMaskArr = null;
            for (NDArray t : outputs) {
                long[] shape = t.getShape().getShape();
                if (shape.length == 5 && shape[0] == 1) {
                    if (shape[1] > 100) blockLogitsArr = t;
                    else if (shape[1] == 1) airMaskArr = t;
                }
            }
            if (blockLogitsArr == null)
                throw new IllegalStateException("block_logits not found in v2 output");
            if (airMaskArr == null)
                throw new IllegalStateException("air_mask not found in v2 output");

            float[][][][][] blockLogits = extract5D(blockLogitsArr);
            float[][][][][] airMask    = extract5D(airMaskArr);

            long elapsed = System.currentTimeMillis() - t0;
            return new InferenceResult(blockLogits, airMask, elapsed);
        }
    }

    /** Whether the loaded model uses the v2 anchor-channel contract. */
    public boolean isV2() {
        return !config.isV1Contract();
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------
    public ModelConfig config() { return config; }
    public BlockVocabulary vocabulary() { return vocabulary; }
    public boolean isAvailable() { return model != null; }

    // ------------------------------------------------------------------
    // Lifecycle
    // ------------------------------------------------------------------

    @Override
    public void close() {
        if (model != null) model.close();
        if (manager != null) manager.close();
    }

    // ------------------------------------------------------------------
    // Array helpers
    // ------------------------------------------------------------------

    private static float[] flatten3D(float[][][] arr, int d0, int d1, int d2) {
        float[] flat = new float[d0 * d1 * d2];
        int idx = 0;
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                for (int k = 0; k < d2; k++)
                    flat[idx++] = arr[i][j][k];
        return flat;
    }

    private static float[] flatten2D(float[][] arr, int d0, int d1) {
        float[] flat = new float[d0 * d1];
        int idx = 0;
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                flat[idx++] = arr[i][j];
        return flat;
    }

    /**
     * Flatten a [channels][spatialSize] array to a single float[].
     * Used to feed x_height_planes and x_router6 tensors to the ONNX session.
     */
    private static float[] flatten2DChannels(float[][] arr, int channels, int spatial) {
        float[] flat = new float[channels * spatial];
        for (int c = 0; c < channels; c++)
            System.arraycopy(arr[c], 0, flat, c * spatial, spatial);
        return flat;
    }

    private static float[][][][][] extract5D(NDArray t) {
        long[] s = t.getShape().getShape();
        int b = (int) s[0], c = (int) s[1], d = (int) s[2], h = (int) s[3], w = (int) s[4];
        float[] flat = t.toFloatArray();
        float[][][][][] out = new float[b][c][d][h][w];
        int idx = 0;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int di = 0; di < d; di++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++)
                            out[bi][ci][di][hi][wi] = flat[idx++];
        return out;
    }

    private static String stripExtension(String filename) {
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }
}

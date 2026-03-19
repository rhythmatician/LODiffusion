package com.rhythmatician.lodiffusion.onnx;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;

/**
 * Loads and runs the single {@code sparse_octree.onnx} model that replaces the
 * three-model octree pipeline (init / refine / leaf) for Stage 2 block
 * selection.
 *
 * <h3>Contract: {@code lodiffusion.v6.sparse_octree}</h3>
 *
 * <pre>
 *   sparse_octree.onnx
 *     Input:
 *       noise_3d   float32[1, C, Dy, Dz, Dx]  (noise channels × spatial cells)
 *                  Legacy models:  [1, 13, 4, 2, 4]   (13 intermediate channels, 4×2×4)
 *                  v7+ models:     [1, 15, 4, 4, 4]   (15 NoiseRouter fields, 4×4×4 quarts)
 *       biome_ids  int32[1, ...]   (discrete biome palette indices, optional)
 *
 *     Outputs (teacher-forced, all nodes expanded at every level):
 *       split_L4   float32[1,    1]           split logits at level 4 (root)
 *       label_L4   float32[1,    1, C]        block-class logits
 *       split_L3   float32[1,    8]
 *       label_L3   float32[1,    8, C]
 *       split_L2   float32[1,   64]
 *       label_L2   float32[1,   64, C]
 *       split_L1   float32[1,  512]
 *       label_L1   float32[1,  512, C]
 *       split_L0   float32[1, 4096]
 *       label_L0   float32[1, 4096, C]
 * </pre>
 *
 * <p>The 4096 L0 nodes correspond to individual blocks in the 16³ subchunk.
 * Node ordering follows breadth-first octant traversal:
 * <pre>
 *   n = a3*512 + a2*64 + a1*8 + a0    where each a is an octant index (0-7)
 *   bx = (a3&amp;1)*8 | (a2&amp;1)*4 | (a1&amp;1)*2 | (a0&amp;1)
 *   bz = ((a3&gt;&gt;1)&amp;1)*8 | ((a2&gt;&gt;1)&amp;1)*4 | ((a1&gt;&gt;1)&amp;1)*2 | ((a0&gt;&gt;1)&amp;1)
 *   by = ((a3&gt;&gt;2)&amp;1)*8 | ((a2&gt;&gt;2)&amp;1)*4 | ((a1&gt;&gt;2)&amp;1)*2 | ((a0&gt;&gt;2)&amp;1)
 * </pre>
 *
 * <p>At inference time, greedy top-down pruning is applied: a node is only
 * expanded to its 8 children if {@code sigmoid(split_logit) > splitThreshold}.
 * Otherwise the argmax of its label logits fills the entire sub-region.
 *
 * @see com.rhythmatician.lodiffusion.world.noise.NoiseRouterSampler
 */
public final class SparseOctreeModelRunner implements AutoCloseable {

    private static final Logger LOGGER = LoggerFactory.getLogger(SparseOctreeModelRunner.class);

    /** ONNX model stem (no extension). */
    private static final String STEM = "sparse_octree";
    /** Config sidecar filename. */
    private static final String CONFIG = "sparse_octree_config.json";
    /** Runtime config key for split expansion probability threshold. */
    static final String SPLIT_THRESHOLD_CONFIG_KEY = "sparseRootSplitThreshold";
    /** Default split expansion threshold used when runtime config does not override it. */
    static final float DEFAULT_SPLIT_THRESHOLD = 0.43f;

    /** Octree depth. L4 = root, L0 = leaf (individual blocks). */
    private static final int LEVELS = 5;
    /** Edge of the generated subchunk, in blocks. */
    static final int SUBCHUNK = 16;
    /** Total leaf nodes in the fully-expanded octree (16³). */
    static final int LEAF_NODES = 4096;

    /**
     * Split threshold: sigmoid(split_logit) &gt; this → expand to children.
     * Set once at load time from runtime config key {@code "sparseRootSplitThreshold"}
     * (default 0.43). The same value is written to {@code sparse_octree_config.json}
     * by {@code export_sparse_octree.py} so training and runtime stay in sync.
     */
    private final float splitThreshold;

    /** Whether the model expects a 2D noise input (noise_2d). */
    private final boolean hasNoise2d;

    /** Shape of the noise_2d input tensor, if present. */
    private final long[] noise2dShape;

    /** Whether the model expects biome IDs input (biome_ids). */
    private final boolean hasBiomeIds;

    /** Shape of the biome_ids input tensor, if present. */
    private final long[] biomeIdsShape;

    /** Whether the model expects a heightmap_surface input. */
    private final boolean hasHeightmapSurface;

    /** Shape of the heightmap_surface input tensor, if present. */
    private final long[] heightmapSurfaceShape;

    /** Whether the model expects a heightmap_ocean_floor input. */
    private final boolean hasHeightmapOceanFloor;

    /** Shape of the heightmap_ocean_floor input tensor, if present. */
    private final long[] heightmapOceanFloorShape;

    /**
     * Shape of the noise_3d input tensor as declared in the model config sidecar.
     * Legacy (v6): {@code [1, 13, 4, 2, 4]}.  New (v7+): {@code [1, 15, 4, 2, 4]}.
     * Used by {@link #runInferenceWithBiome} to reshape the flat noise array.
     * Falls back to {@code [1, 13, 4, 2, 4]} if the config doesn't declare a shape.
     */
    private final long[] noise3dShape;

    /** Ordered ONNX/model-config input tensor names for name-based mapping. */
    private final List<String> inputOrder;

    // ── Node counts per level ──────────────────────────────────────────
    /** Number of nodes in the fully-expanded octree at each level 4-0. */
    private static final int[] LEVEL_NODES = { 1, 8, 64, 512, 4096 };
    // level index in this array: 0=L4(root), 1=L3, 2=L2, 3=L1, 4=L0(leaf)

    private final NDManager manager;
    private final ZooModel<NDList, NDList> model;
    private final BlockVocabulary vocabulary;
    private final int numClasses;

    /** Guards the one-shot diagnostic log (fires on first inference only). */
    private final AtomicBoolean debugOnce = new AtomicBoolean(false);

    private SparseOctreeModelRunner(NDManager manager,
                                  ZooModel<NDList, NDList> model,
                                  BlockVocabulary vocabulary,
                                  int numClasses,
                                  float splitThreshold,
                                  boolean hasNoise2d,
                                  long[] noise2dShape,
                                  boolean hasBiomeIds,
                                  long[] biomeIdsShape,
                                  boolean hasHeightmapSurface,
                                  long[] heightmapSurfaceShape,
                                  boolean hasHeightmapOceanFloor,
                                  long[] heightmapOceanFloorShape,
                                  long[] noise3dShape,
                                  List<String> inputOrder) {
        this.manager                  = manager;
        this.model                    = model;
        this.vocabulary               = vocabulary;
        this.numClasses               = numClasses;
        this.splitThreshold           = splitThreshold;
        this.hasNoise2d               = hasNoise2d;
        this.noise2dShape             = noise2dShape;
        this.hasBiomeIds              = hasBiomeIds;
        this.biomeIdsShape            = biomeIdsShape;
        this.hasHeightmapSurface      = hasHeightmapSurface;
        this.heightmapSurfaceShape    = heightmapSurfaceShape;
        this.hasHeightmapOceanFloor   = hasHeightmapOceanFloor;
        this.heightmapOceanFloorShape = heightmapOceanFloorShape;
        this.noise3dShape             = noise3dShape;
        this.inputOrder               = List.copyOf(inputOrder);
    }

    // ------------------------------------------------------------------
    // Factory
    // ------------------------------------------------------------------

    /**
     * Try to load {@code sparse_octree.onnx} from {@code modelDir}.
     *
     * @param modelDir directory containing model files
     * @return loaded runner, or {@code null} if the file is absent
     * @throws IOException if the file is present but cannot be loaded
     */
    public static SparseOctreeModelRunner tryLoad(Path modelDir) throws IOException {
        Path onnxPath = modelDir.resolve(STEM + ".onnx");
        if (!Files.exists(onnxPath)) {
            LOGGER.info("[SparseOctree] {} not found — sparse-root model unavailable", onnxPath);
            return null;
        }

        InferenceDeviceSelector.Provider provider = InferenceDeviceSelector.selectProvider();
        LOGGER.info("[SparseOctree] Loading {} (provider={})", onnxPath, provider);

        ClassLoader prevCl = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(
                SparseOctreeModelRunner.class.getClassLoader());
        try {
            NDManager manager = NDManager.newBaseManager();
            try {
                ZooModel<NDList, NDList> zm = buildAndLoad(modelDir, provider);

                // Load vocab from config sidecar, fall back to a default
                BlockVocabulary vocab = null;
                int numClassesFromConfig = 256;
                boolean hasNoise2d = false;
                long[] noise2dShape = null;
                boolean hasBiomeIds = false;
                long[] biomeIdsShape = null;
                boolean hasHeightmapSurface = false;
                long[] heightmapSurfaceShape = null;
                boolean hasHeightmapOceanFloor = false;
                long[] heightmapOceanFloorShape = null;
                long[] noise3dShape = new long[]{1, 13, 4, 2, 4}; // legacy default
                List<String> inputOrder = List.of("noise_3d");
                Path configPath = modelDir.resolve(CONFIG);
                if (Files.exists(configPath)) {
                    ModelConfig cfg = ConfigLoader.load(configPath);
                    int nc = cfg.effectiveBlockVocabSize();
                    if (cfg.blockMapping() != null && !cfg.blockMapping().isEmpty()) {
                        try {
                            vocab = BlockVocabulary.fromConfig(cfg);
                            if (nc <= 0) nc = vocab.size();
                        } catch (ExceptionInInitializerError | NoClassDefFoundError e) {
                            // Minecraft block registry not bootstrapped (unit-test env) — skip vocab.
                            // In prod, registries are always initialized before this path is reached.
                            LOGGER.warn("[SparseOctree] Block registry unavailable (test env?) — "
                                    + "vocab skipped: {}", e.getClass().getSimpleName());
                        }
                    }
                    if (nc <= 0) nc = 256;
                    numClassesFromConfig = nc;

                    hasNoise2d = cfg.hasInput("noise_2d");
                    if (hasNoise2d) {
                        int[] shape = cfg.getInputShape("noise_2d");
                        noise2dShape = new long[shape.length];
                        for (int i = 0; i < shape.length; i++) noise2dShape[i] = shape[i];
                    }

                    hasBiomeIds = cfg.hasInput("biome_ids");
                    if (hasBiomeIds) {
                        int[] shape = cfg.getInputShape("biome_ids");
                        biomeIdsShape = new long[shape.length];
                        for (int i = 0; i < shape.length; i++) biomeIdsShape[i] = shape[i];
                    }

                    hasHeightmapSurface = cfg.hasInput("heightmap_surface");
                    if (hasHeightmapSurface) {
                        int[] shape = cfg.getInputShape("heightmap_surface");
                        heightmapSurfaceShape = new long[shape.length];
                        for (int i = 0; i < shape.length; i++) heightmapSurfaceShape[i] = shape[i];
                    }

                    hasHeightmapOceanFloor = cfg.hasInput("heightmap_ocean_floor");
                    if (hasHeightmapOceanFloor) {
                        int[] shape = cfg.getInputShape("heightmap_ocean_floor");
                        heightmapOceanFloorShape = new long[shape.length];
                        for (int i = 0; i < shape.length; i++) heightmapOceanFloorShape[i] = shape[i];
                    }

                    // Discover noise_3d shape from config sidecar.
                    // Legacy (v6) models declare [1, 13, 4, 2, 4].
                    // New (v7+) models declare [1, 15, 4, 2, 4] (15 NoiseRouter fields, quart res).
                    if (cfg.hasInput("noise_3d")) {
                        int[] shape = cfg.getInputShape("noise_3d");
                        if (shape != null && shape.length > 0) {
                            noise3dShape = new long[shape.length];
                            for (int i = 0; i < shape.length; i++) noise3dShape[i] = shape[i];
                        }
                    }

                    inputOrder = resolveInputOrder(cfg);

                    LOGGER.info("[SparseOctree] Loaded — vocab={} numClasses={} provider={}",
                            vocab != null ? vocab.size() : "none", nc, provider);
                } else {
                    LOGGER.warn("[SparseOctree] {} not found — using raw block indices (no vocab)", configPath);
                }
                // Split threshold priority: runtime config > sidecar > default.
                float splitThreshold;
                if (com.rhythmatician.lodiffusion.Config.hasKey(SPLIT_THRESHOLD_CONFIG_KEY)) {
                    // Explicit runtime override takes precedence.
                    splitThreshold = (float) com.rhythmatician.lodiffusion.Config
                            .getDouble(SPLIT_THRESHOLD_CONFIG_KEY, DEFAULT_SPLIT_THRESHOLD);
                    LOGGER.info("[SparseOctree] splitThreshold={} (runtime config override)", splitThreshold);
                } else if (Files.exists(configPath)) {
                    ModelConfig sidecar = ConfigLoader.load(configPath);
                    if (sidecar.splitThreshold() != null) {
                        splitThreshold = sidecar.splitThreshold().floatValue();
                        LOGGER.info("[SparseOctree] splitThreshold={} (from model sidecar)", splitThreshold);
                    } else {
                        splitThreshold = DEFAULT_SPLIT_THRESHOLD;
                        LOGGER.info("[SparseOctree] splitThreshold={} (default fallback)", splitThreshold);
                    }
                } else {
                    splitThreshold = DEFAULT_SPLIT_THRESHOLD;
                    LOGGER.info("[SparseOctree] splitThreshold={} (default fallback)", splitThreshold);
                }
                LOGGER.info("[SparseOctree] noise_3d shape: {}",
                        java.util.Arrays.toString(noise3dShape));
                return new SparseOctreeModelRunner(manager, zm, vocab, numClassesFromConfig,
                        splitThreshold, hasNoise2d, noise2dShape,
                        hasBiomeIds, biomeIdsShape,
                        hasHeightmapSurface, heightmapSurfaceShape,
                        hasHeightmapOceanFloor, heightmapOceanFloorShape,
                        noise3dShape, inputOrder);

            } catch (Exception e) {
                manager.close();
                if (e instanceof IOException) throw (IOException) e;
                throw new IOException("Failed to load sparse_octree from " + modelDir, e);
            }
        } finally {
            Thread.currentThread().setContextClassLoader(prevCl);
        }
    }

    @SuppressWarnings("null")
    private static ZooModel<NDList, NDList> buildAndLoad(Path dir,
                                                          InferenceDeviceSelector.Provider provider)
            throws Exception {
        // Try the requested provider first (e.g. DirectML)
        if (!provider.djlOptionValue().isEmpty()) {
            ZooModel<NDList, NDList> attempted = null;
            try {
                attempted = buildCriteria(dir, provider.djlOptionValue())
                        .build().loadModel();
                LOGGER.info("[SparseOctree] Loaded with provider {}", provider);
                return attempted;
            } catch (Exception ex) {
                if (attempted != null) {
                    try { attempted.close(); } catch (Exception ignore) {}
                }
                LOGGER.warn("[SparseOctree] Provider {} unavailable ({}); falling back to CPU",
                        provider, ex.getMessage());
            }
        }
        // CPU fallback (or initial attempt when provider == CPU)
        ZooModel<NDList, NDList> model = buildCriteria(dir, null).build().loadModel();
        if (!provider.djlOptionValue().isEmpty()) {
            LOGGER.info("[SparseOctree] Loaded with CPU fallback");
        } else {
            LOGGER.info("[SparseOctree] Loaded (CPU)");
        }
        return model;
    }

    private static Criteria.Builder<NDList, NDList> buildCriteria(Path dir,
                                                                   String ortDevice) {
        Criteria.Builder<NDList, NDList> builder = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelPath(dir)
                .optModelName(STEM)
                .optEngine("OnnxRuntime")
                .optTranslator(new NoopTranslator())
                .optOption("interOpNumThreads", "1")
                .optOption("intraOpNumThreads", "4");
        if (ortDevice != null && !ortDevice.isEmpty()) {
            builder.optOption("ortDevice", ortDevice);
        }
        return builder;
    }

    // ------------------------------------------------------------------
    // Inference
    // ------------------------------------------------------------------

    static List<String> resolveInputOrder(ModelConfig cfg) {
        List<String> order = new ArrayList<>();
        java.util.Set<String> knownInputs = java.util.Set.of(
                "noise_2d", "noise_3d", "biome_ids",
                "heightmap_surface", "heightmap_ocean_floor");
        if (cfg.inputs() != null) {
            for (String name : cfg.inputs().keySet()) {
                if (knownInputs.contains(name)) {
                    order.add(name);
                }
            }
        }
        if (cfg.optionalInputs() != null) {
            for (String name : cfg.optionalInputs().keySet()) {
                if (!order.contains(name) && knownInputs.contains(name)) {
                    order.add(name);
                }
            }
        }
        if (!order.contains("noise_3d")) {
            order.add("noise_3d");
        }
        return order;
    }

    /**
     * Run inference for a single 16³ subchunk.
     *
     * @param noiseFlat flat noise input in channel-outermost order.
     *        Shape must match {@link #noise3dShape()}.  Standard v7+ input:
     *        {@code float[15 * 4 * 2 * 4 = 480]} from
     *        {@link com.rhythmatician.lodiffusion.world.noise.NoiseRouterSampler}.
     * @param biomeIds optional biome IDs at cell resolution matching noiseFlat.
     *        If {@code null} or empty, the model runs without biome conditioning.
     * @return {@code int[16][16][16]} block IDs in {@code [y][z][x]} order
     *         (matching the Voxy native storage format), or {@code null} if
     *         inference failed
     */
    public int[][][] runInference(float[] noiseFlat, int... unused) {
        return runInferenceWithBiome(noiseFlat, null, null, null);
    }

    /**
     * Run inference with explicit biome IDs and heightmaps.
     *
     * @param noiseFlat        flat noise input (length must match the product of
     *                         {@link #noise3dShape()} dimensions)
     * @param biomeIds         biome IDs array, or {@code null} for zero-fill
     * @param heightmapSurface 16×16 surface heightmap (block Y), or {@code null}
     * @param heightmapOceanFloor 16×16 ocean floor heightmap (block Y), or {@code null}
     * @return {@code int[16][16][16]} block IDs, or {@code null} if inference failed
     */
    public int[][][] runInferenceWithBiome(float[] noiseFlat, int[][][] biomeIds,
                                           float[][] heightmapSurface,
                                           float[][] heightmapOceanFloor) {
        ClassLoader prevCl = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(getClass().getClassLoader());
        try (NDManager sub = manager.newSubManager()) {
            // Build input tensor using the model's declared shape
            NDArray noise3d = sub.create(noiseFlat,
                    new Shape(noise3dShape));
            NDArray noise2d = (hasNoise2d && noise2dShape != null)
                    ? sub.zeros(new Shape(noise2dShape))
                    : null;
            NDArray biome = hasBiomeIds ? buildBiomeTensor(sub, biomeIds) : null;
            NDArray hmSurface = hasHeightmapSurface
                    ? buildHeightmapTensor(sub, heightmapSurface, heightmapSurfaceShape)
                    : null;
            NDArray hmOcean = hasHeightmapOceanFloor
                    ? buildHeightmapTensor(sub, heightmapOceanFloor, heightmapOceanFloorShape)
                    : null;

            NDList inputs = new NDList();
            for (String name : inputOrder) {
                switch (name) {
                    case "noise_2d" -> {
                        if (noise2d != null) inputs.add(noise2d);
                    }
                    case "noise_3d" -> inputs.add(noise3d);
                    case "biome_ids" -> {
                        if (biome != null) inputs.add(biome);
                    }
                    case "heightmap_surface" -> {
                        if (hmSurface != null) inputs.add(hmSurface);
                    }
                    case "heightmap_ocean_floor" -> {
                        if (hmOcean != null) inputs.add(hmOcean);
                    }
                    default -> {
                        // ignored
                    }
                }
            }
            if (inputs.isEmpty()) {
                inputs.add(noise3d);
                if (noise2d != null) inputs.add(0, noise2d);
                if (biome != null) inputs.add(biome);
                if (hmSurface != null) inputs.add(hmSurface);
                if (hmOcean != null) inputs.add(hmOcean);
            }

            long t0 = System.currentTimeMillis();
            int[][][] blocks;
            try (var predictor = model.newPredictor()) {
                NDList outputs = predictor.predict(inputs);
                long elapsedMs = System.currentTimeMillis() - t0;
                LOGGER.debug("[SparseOctree] Inference complete in {}ms", elapsedMs);
                blocks = decodeOutputs(outputs, sub);
            }
            return blocks;

        } catch (TranslateException e) {
            LOGGER.warn("[SparseOctree] Inference failed: {}", e.getMessage());
            return null;
        } finally {
            Thread.currentThread().setContextClassLoader(prevCl);
        }
    }

    private NDArray buildBiomeTensor(NDManager sub, int[][][] biomeIds) {
        // Read spatial dimensions from the model config sidecar.
        // v7+ models declare [1, 4, 2, 4]; legacy v6 declared [1, 4, 2, 4].
        final int d0, d1, d2;
        if (biomeIdsShape != null && biomeIdsShape.length == 4) {
            d0 = (int) biomeIdsShape[1];
            d1 = (int) biomeIdsShape[2];
            d2 = (int) biomeIdsShape[3];
        } else {
            // Default to v7 quart-resolution 4×2×4 (vanilla cellHeight=8)
            d0 = 4;
            d1 = 2;
            d2 = 4;
        }

        // ONNX model expects biome_ids as int64 (torch.long), so use long[].
        long[] flattened = new long[d0 * d1 * d2];
        if (biomeIds == null) {
            LOGGER.debug("[SparseOctree] biome_ids expected but null was provided; using zeros fallback");
            return sub.create(flattened, new Shape(1, d0, d1, d2));
        }
        if (biomeIds.length != d0) {
            LOGGER.warn("[SparseOctree] biome_ids dim0 size {} != {}; using zeros fallback", biomeIds.length, d0);
            return sub.create(flattened, new Shape(1, d0, d1, d2));
        }

        int idx = 0;
        for (int i = 0; i < d0; i++) {
            if (biomeIds[i] == null || biomeIds[i].length != d1) {
                LOGGER.warn("[SparseOctree] biome_ids dim1 mismatch at [{}]; using zeros fallback", i);
                return sub.create(new long[d0 * d1 * d2], new Shape(1, d0, d1, d2));
            }
            for (int j = 0; j < d1; j++) {
                if (biomeIds[i][j] == null || biomeIds[i][j].length != d2) {
                    LOGGER.warn("[SparseOctree] biome_ids dim2 mismatch at [{},{}]; using zeros fallback", i, j);
                    return sub.create(new long[d0 * d1 * d2], new Shape(1, d0, d1, d2));
                }
                for (int k = 0; k < d2; k++) {
                    int biome = biomeIds[i][j][k];
                    if (biome < 0) {
                        LOGGER.warn("[SparseOctree] biome_ids contains negative value {}; using zeros fallback", biome);
                        return sub.create(new long[d0 * d1 * d2], new Shape(1, d0, d1, d2));
                    }
                    flattened[idx++] = biome;
                }
            }
        }
        return sub.create(flattened, new Shape(1, d0, d1, d2));
    }

    /**
     * Build a heightmap tensor from a {@code float[16][16]} array.
     *
     * @param sub      NDManager for tensor allocation
     * @param heightmap 16×16 block-Y heightmap, or {@code null} for zero-fill
     * @param shape    declared shape from model config (e.g. [1, 16, 16])
     * @return NDArray with the expected shape
     */
    private NDArray buildHeightmapTensor(NDManager sub, float[][] heightmap, long[] shape) {
        long[] effectiveShape = (shape != null && shape.length >= 2) ? shape : new long[]{1, 16, 16};
        int rows = (int) effectiveShape[effectiveShape.length - 2];
        int cols = (int) effectiveShape[effectiveShape.length - 1];
        float[] flat = new float[rows * cols];
        if (heightmap != null) {
            int idx = 0;
            for (int r = 0; r < Math.min(rows, heightmap.length); r++) {
                if (heightmap[r] != null) {
                    for (int c = 0; c < Math.min(cols, heightmap[r].length); c++) {
                        flat[idx + c] = heightmap[r][c];
                    }
                }
                idx += cols;
            }
        }
        return sub.create(flat, new Shape(effectiveShape));
    }

    // ------------------------------------------------------------------
    // Output decoding
    // ------------------------------------------------------------------

    /**
     * Decode the 10-output sparse-root ONNX result into a 16³ block grid.
     *
     * <p>Performs greedy top-down traversal:
     * <ol>
     *   <li>Start at the single L4 root node.</li>
     *   <li>At each node, evaluate {@link #shouldExpandNode(float, float)}.</li>
     *   <li>If <em>split</em>: enqueue all 8 children at the next finer level.</li>
     *   <li>If <em>leaf</em>: fill the node's sub-region with
     *       {@code argmax(label_logits)}.</li>
     * </ol>
     *
     * <p>The output tensors are identified by shape rather than by name to
     * remain robust to different ONNX export orderings.
     */
    private int[][][] decodeOutputs(NDList outputs, NDManager sub) {
        // Locate split and label tensors by level node count
        float[][] splitByLevel = new float[LEVELS][];   // index 0 = L4 (1 node)
        float[][] labelByLevel = new float[LEVELS][];   // flat [N*C]
        int[]    cByLevel      = new int[LEVELS];        // actual C from tensor shape

        for (NDArray t : outputs) {
            long[] shape = t.getShape().getShape();
            if (shape.length == 2) {
                // Split tensor: [1, N]
                int n = (int) shape[1];
                int lvlIdx = levelIndexFromNodeCount(n);
                if (lvlIdx >= 0 && splitByLevel[lvlIdx] == null) {
                    splitByLevel[lvlIdx] = t.toFloatArray();
                }
            } else if (shape.length == 3) {
                // Label tensor: [1, N, C]  — capture C so argmax stride is correct
                int n = (int) shape[1];
                int c = (int) shape[2];
                int lvlIdx = levelIndexFromNodeCount(n);
                if (lvlIdx >= 0 && labelByLevel[lvlIdx] == null) {
                    labelByLevel[lvlIdx] = t.toFloatArray();
                    cByLevel[lvlIdx] = c;
                }
            }
        }

        // One-shot diagnostics: log tensor shapes + root signal on first inference
        if (debugOnce.compareAndSet(false, true)) {
            StringBuilder sb = new StringBuilder("[SparseOctree] First-inference diagnostics:");
            sb.append("\n  Output tensors (").append(outputs.size()).append(" total):");
            for (NDArray t : outputs) {
                sb.append(" ").append(java.util.Arrays.toString(t.getShape().getShape()));
            }
            sb.append("\n  cByLevel: ").append(java.util.Arrays.toString(cByLevel));
            // Root split
            if (splitByLevel[0] != null && splitByLevel[0].length > 0) {
                float rootLogit = splitByLevel[0][0];
                float rootSigm = sigmoid(rootLogit);
                sb.append("\n  Root split logit=").append(rootLogit)
                  .append(" sigmoid=").append(String.format("%.4f", rootSigm))
                  .append(" threshold=").append(splitThreshold)
                  .append(" => ").append(shouldExpandNode(rootLogit, splitThreshold) ? "SPLIT" : "LEAF");
            }
            // Root label argmax
            if (labelByLevel[0] != null && cByLevel[0] > 0) {
                int rootBlockId = argmaxLabel(labelByLevel[0], 0, cByLevel[0]);
                sb.append("\n  Root label argmax=").append(rootBlockId);
                // Show top-5 logit values at root
                float[] rootLabel = labelByLevel[0];
                sb.append(" top logits:");
                for (int i = 0; i < Math.min(5, cByLevel[0]); i++) {
                    sb.append(" [").append(i).append("]").append(String.format("%.3f", rootLabel[i]));
                }
            }
            // L0 label class 0 logit vs class 1 at node 0
            if (labelByLevel[4] != null && cByLevel[4] > 0) {
                int l0BlockId = argmaxLabel(labelByLevel[4], 0, cByLevel[4]);
                sb.append("\n  L0[0] argmax=").append(l0BlockId)
                  .append(" c=").append(cByLevel[4])
                  .append(" array_len=").append(labelByLevel[4].length)
                  .append(" expected=").append((long) LEVEL_NODES[4] * cByLevel[4]);
            }
            LOGGER.info("{}", sb);
        }

        // ── Model-level defer policy (Layer 2 safety) ───────────────
        // Validates raw tensors before decode.  On rejection, return null
        // so the caller can fall back to the sampler-based generation path.
        ModelOutputValidator.ValidationResult preCheck =
                ModelOutputValidator.validatePreDecode(splitByLevel, labelByLevel, cByLevel, LEVELS);
        if (preCheck != ModelOutputValidator.ValidationResult.ACCEPT) {
            LOGGER.warn("[SparseOctree] Pre-decode validation failed: {} — deferring to fallback",
                    preCheck);
            return null;
        }

        for (int i = 0; i < LEVELS; i++) {
            if (splitByLevel[i] == null || labelByLevel[i] == null) {
                LOGGER.warn("[SparseOctree] Missing output tensor for level index {} "
                        + "(nodeCount={}); using dense L0 fallback", i, LEVEL_NODES[i]);
                // Fall back to filling the whole block with argmax of L0 label
            }
        }

        int[][][] grid = new int[SUBCHUNK][SUBCHUNK][SUBCHUNK];
        float thresh = this.splitThreshold;

        // Greedy top-down traversal.  Each active node is represented as an
        // integer triple (level_index, node_index_in_level, packed_block_origin)
        // where packed_block_origin = by*256 + bz*16 + bx (all in [0,15]).
        Deque<int[]> pending = new ArrayDeque<>(4096);
        // Root: level_index=0, node_index=0, origin=(0,0,0)
        pending.push(new int[] { 0, 0, 0 });

        while (!pending.isEmpty()) {
            int[] node = pending.pop();
            int lvlIdx   = node[0];
            int nodeIdx  = node[1];
            int origin   = node[2];  // packed by,bz,bx

            int by0 = (origin >> 8) & 0xF;
            int bz0 = (origin >> 4) & 0xF;
            int bx0 = origin & 0xF;

            // Size of this node's sub-region in blocks: 16 >> lvlIdx
            // lvlIdx 0=L4→16, 1=L3→8, 2=L2→4, 3=L1→2, 4=L0→1
            int size = SUBCHUNK >> lvlIdx;

            boolean isSplit = false;
            float[] splitArr = splitByLevel[lvlIdx];
            if (splitArr != null && lvlIdx < LEVELS - 1) {
                isSplit = shouldExpandNode(splitArr[nodeIdx], thresh);
            }

            if (isSplit && lvlIdx < LEVELS - 1) {
                // Expand: enqueue 8 children at the next finer level
                int childSize = size >> 1;
                int childLvlIdx = lvlIdx + 1;
                int childBase = nodeIdx * 8;
                for (int oct = 0; oct < 8; oct++) {
                    int dx = oct & 1;
                    int dz = (oct >> 1) & 1;
                    int dy = (oct >> 2) & 1;
                    int childOrigin = ((by0 + dy * childSize) << 8)
                                    | ((bz0 + dz * childSize) << 4)
                                    | (bx0 + dx * childSize);
                    pending.push(new int[] { childLvlIdx, childBase + oct, childOrigin });
                }
            } else {
                // Leaf: fill sub-region with argmax of label logits.
                // Use the actual C captured from the tensor shape — NOT numClasses
                // (which is the config vocab size and may differ from the model's
                // trained output width, causing incorrect stride and all-air output).
                int c = (cByLevel[lvlIdx] > 0) ? cByLevel[lvlIdx] : numClasses;
                int blockId = argmaxLabel(labelByLevel[lvlIdx], nodeIdx, c);
                fillRegion(grid, by0, bz0, bx0, size, blockId);
            }
        }

        // ── Post-decode validation ──────────────────────────────────
        ModelOutputValidator.ValidationResult postCheck =
                ModelOutputValidator.validatePostDecode(grid);
        if (postCheck != ModelOutputValidator.ValidationResult.ACCEPT) {
            LOGGER.warn("[SparseOctree] Post-decode validation failed: {} — deferring to fallback",
                    postCheck);
            return null;
        }

        return grid;
    }

    /**
     * Compute the argmax block ID for a single node's label logits.
     *
     * @param labelFlat flat label array {@code [N * C]} row-major
     * @param nodeIdx   index of this node within the level (0-based)
     * @param c         number of classes
     * @return argmax class index, clamped to [0, c-1]
     */
    private static int argmaxLabel(float[] labelFlat, int nodeIdx, int c) {
        if (labelFlat == null || c <= 0) return 0;
        int offset = nodeIdx * c;
        if (offset >= labelFlat.length) return 0;

        int best = 0;
        float bestVal = labelFlat[offset];
        for (int cl = 1; cl < c && offset + cl < labelFlat.length; cl++) {
            float v = labelFlat[offset + cl];
            if (v > bestVal) {
                bestVal = v;
                best = cl;
            }
        }
        return best;
    }

    /**
     * Fill a cubic sub-region of the block grid with a single block ID.
     *
     * @param grid  {@code int[16][16][16]} in [y][z][x] order
     * @param by0   Y origin
     * @param bz0   Z origin
     * @param bx0   X origin
     * @param size  edge length of the cube to fill
     * @param blockId block ID to write
     */
    private static void fillRegion(int[][][] grid,
                                   int by0, int bz0, int bx0,
                                   int size, int blockId) {
        int yEnd = Math.min(by0 + size, SUBCHUNK);
        int zEnd = Math.min(bz0 + size, SUBCHUNK);
        int xEnd = Math.min(bx0 + size, SUBCHUNK);
        for (int by = by0; by < yEnd; by++)
            for (int bz = bz0; bz < zEnd; bz++)
                for (int bx = bx0; bx < xEnd; bx++)
                    grid[by][bz][bx] = blockId;
    }

    /**
     * Map a node count to a level index (0=L4/root, 4=L0/leaf).
     * Returns -1 for unrecognised counts.
     */
    private static int levelIndexFromNodeCount(int n) {
        for (int i = 0; i < LEVELS; i++) {
            if (LEVEL_NODES[i] == n) return i;
        }
        return -1;
    }


    static boolean shouldExpandNode(float splitLogit, float splitThreshold) {
        return sigmoid(splitLogit) > splitThreshold;
    }

    static float sigmoid(float value) {
        return 1.0f / (1.0f + (float) Math.exp(-value));
    }

    static float logitForThreshold(float threshold) {
        return (float) Math.log(threshold / (1.0f - threshold));
    }

    // ------------------------------------------------------------------
    // Batch inference
    // ------------------------------------------------------------------

    /** Maximum batch size for a single ONNX call. */
    public static final int MAX_BATCH_SIZE = 8;

    /**
     * Tracks whether the model supports dynamic batch dimension.
     * Starts as {@code true} (optimistic); flips to {@code false} on first
     * failure, after which all batch calls degrade to sequential.
     */
    private volatile boolean batchSupported = true;

    public int[][][][] runBatchInference(float[][] noiseBatch, int[][][][] biomeBatch) {
        return runBatchInference(noiseBatch, biomeBatch, null, null);
    }

    /**
     * Run inference for multiple sections in a single ONNX call.
     *
     * <p>If the model supports dynamic batch dimension, the noise arrays are
     * stacked into a single {@code [N, C, Dy, Dz, Dx]} tensor and dispatched
     * once.  Output tensors are then sliced per sample and decoded
     * independently.
     *
     * <p>If batched inference fails (model compiled with static batch=1),
     * this method transparently falls back to sequential
     * {@link #runInferenceWithBiome} calls and marks batch mode as
     * unsupported for future calls.
     *
     * @param noiseBatch array of flat noise inputs (each of length
     *                   {@link #noise3dFlatLength()}).  Length ∈ [1, MAX_BATCH_SIZE].
     * @param biomeBatch optional parallel array of biome IDs (same length as
     *                   {@code noiseBatch}), or {@code null} to zero-fill all
     * @param hmSurfaceBatch optional parallel array of surface heightmaps
     * @param hmOceanBatch   optional parallel array of ocean floor heightmaps
     * @return array of {@code int[16][16][16]} block grids, same length as
     *         {@code noiseBatch}.  Individual entries may be {@code null} on
     *         per-section decode failure.
     */
    public int[][][][] runBatchInference(float[][] noiseBatch, int[][][][] biomeBatch,
                                         float[][][] hmSurfaceBatch,
                                         float[][][] hmOceanBatch) {
        if (noiseBatch == null || noiseBatch.length == 0) {
            return new int[0][][][];
        }
        int n = noiseBatch.length;
        if (n == 1) {
            // Single-sample fast path (no batch overhead)
            int[][][] result = runInferenceWithBiome(noiseBatch[0],
                    biomeBatch != null ? biomeBatch[0] : null,
                    hmSurfaceBatch != null ? hmSurfaceBatch[0] : null,
                    hmOceanBatch != null ? hmOceanBatch[0] : null);
            return new int[][][][] { result };
        }

        // Try true batched inference if still supported
        if (batchSupported) {
            int[][][][] batched = tryBatchedInference(noiseBatch, biomeBatch,
                    hmSurfaceBatch, hmOceanBatch);
            if (batched != null) return batched;
            // First failure disables batching for all future calls
            batchSupported = false;
            LOGGER.warn("[SparseOctree] Batched inference failed; "
                    + "falling back to sequential (model has static batch=1?)");
        }

        // Sequential fallback
        return runSequentialFallback(noiseBatch, biomeBatch, hmSurfaceBatch, hmOceanBatch);
    }

    /**
     * Attempt true batched ONNX inference.
     * @return decoded grids, or {@code null} if the model rejects the batched input
     */
    private int[][][][] tryBatchedInference(float[][] noiseBatch, int[][][][] biomeBatch,
                                            float[][][] hmSurfaceBatch,
                                            float[][][] hmOceanBatch) {
        int n = noiseBatch.length;
        int flatLen = noise3dFlatLength();

        ClassLoader prevCl = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(getClass().getClassLoader());
        try (NDManager sub = manager.newSubManager()) {
            // Stack noise: [N, C, Dy, Dz, Dx]
            float[] stacked = new float[n * flatLen];
            for (int i = 0; i < n; i++) {
                System.arraycopy(noiseBatch[i], 0, stacked, i * flatLen, flatLen);
            }
            long[] batchShape = noise3dShape.clone();
            batchShape[0] = n;  // replace batch dim
            NDArray noise3d = sub.create(stacked, new Shape(batchShape));

            NDArray noise2d = (hasNoise2d && noise2dShape != null)
                    ? sub.zeros(new Shape(prependBatch(noise2dShape, n)))
                    : null;
            NDArray biome = hasBiomeIds
                    ? buildBatchedBiomeTensor(sub, biomeBatch, n)
                    : null;
            NDArray hmSurface = hasHeightmapSurface
                    ? buildBatchedHeightmapTensor(sub, hmSurfaceBatch, n, heightmapSurfaceShape)
                    : null;
            NDArray hmOcean = hasHeightmapOceanFloor
                    ? buildBatchedHeightmapTensor(sub, hmOceanBatch, n, heightmapOceanFloorShape)
                    : null;

            NDList inputs = new NDList();
            for (String name : inputOrder) {
                switch (name) {
                    case "noise_2d" -> { if (noise2d != null) inputs.add(noise2d); }
                    case "noise_3d" -> inputs.add(noise3d);
                    case "biome_ids" -> { if (biome != null) inputs.add(biome); }
                    case "heightmap_surface" -> { if (hmSurface != null) inputs.add(hmSurface); }
                    case "heightmap_ocean_floor" -> { if (hmOcean != null) inputs.add(hmOcean); }
                    default -> { /* ignored */ }
                }
            }
            if (inputs.isEmpty()) {
                inputs.add(noise3d);
                if (noise2d != null) inputs.add(0, noise2d);
                if (biome != null) inputs.add(biome);
                if (hmSurface != null) inputs.add(hmSurface);
                if (hmOcean != null) inputs.add(hmOcean);
            }

            long t0 = System.currentTimeMillis();
            try (var predictor = model.newPredictor()) {
                NDList outputs = predictor.predict(inputs);
                long elapsed = System.currentTimeMillis() - t0;
                LOGGER.debug("[SparseOctree] Batch inference ({} sections) in {}ms", n, elapsed);
                return decodeBatchOutputs(outputs, sub, n);
            }
        } catch (TranslateException e) {
            LOGGER.debug("[SparseOctree] Batch inference rejected: {}", e.getMessage());
            return null;
        } finally {
            Thread.currentThread().setContextClassLoader(prevCl);
        }
    }

    /**
     * Replace the batch dimension (index 0) of a shape array.
     */
    private static long[] prependBatch(long[] shape, int n) {
        long[] out = shape.clone();
        out[0] = n;
        return out;
    }

    /**
     * Build a batched heightmap tensor by stacking individual heightmaps.
     */
    private NDArray buildBatchedHeightmapTensor(NDManager sub, float[][][] hmBatch,
                                                int n, long[] shape) {
        NDList tensors = new NDList(n);
        for (int i = 0; i < n; i++) {
            tensors.add(buildHeightmapTensor(sub,
                    hmBatch != null && i < hmBatch.length ? hmBatch[i] : null, shape));
        }
        return NDArrays.concat(tensors, 0);
    }

    /**
     * Build a batched biome tensor by stacking individual biome IDs.
     */
    private NDArray buildBatchedBiomeTensor(NDManager sub, int[][][][] biomeBatch, int n) {
        NDList biomes = new NDList(n);
        for (int i = 0; i < n; i++) {
            biomes.add(buildBiomeTensor(sub,
                    biomeBatch != null && i < biomeBatch.length ? biomeBatch[i] : null));
        }
        return NDArrays.concat(biomes, 0);
    }

    /**
     * Decode batched outputs: each tensor has shape {@code [N, ...]} instead
     * of {@code [1, ...]}.  We slice per sample and delegate to
     * {@link #decodeOutputs}.
     */
    private int[][][][] decodeBatchOutputs(NDList outputs, NDManager sub, int batchSize) {
        int[][][][] results = new int[batchSize][][][];
        for (int i = 0; i < batchSize; i++) {
            // Slice each output tensor along dim 0 for this sample
            NDList sampleOutputs = new NDList(outputs.size());
            for (NDArray t : outputs) {
                // t.get(i) selects index i along the first dimension
                NDArray sliced = t.get(i).expandDims(0);  // restore [1, ...] shape
                sampleOutputs.add(sliced);
            }
            results[i] = decodeOutputs(sampleOutputs, sub);
        }
        return results;
    }

    /**
     * Sequential fallback: process one section at a time.
     */
    private int[][][][] runSequentialFallback(float[][] noiseBatch, int[][][][] biomeBatch,
                                              float[][][] hmSurfaceBatch,
                                              float[][][] hmOceanBatch) {
        int n = noiseBatch.length;
        int[][][][] results = new int[n][][][];
        for (int i = 0; i < n; i++) {
            results[i] = runInferenceWithBiome(noiseBatch[i],
                    biomeBatch != null && i < biomeBatch.length ? biomeBatch[i] : null,
                    hmSurfaceBatch != null && i < hmSurfaceBatch.length ? hmSurfaceBatch[i] : null,
                    hmOceanBatch != null && i < hmOceanBatch.length ? hmOceanBatch[i] : null);
        }
        return results;
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /** Block vocabulary used by this model. */
    public BlockVocabulary vocabulary() {
        return vocabulary;
    }

    /**
     * Whether the loaded model config declares a {@code biome_ids} input.
     * When {@code true}, callers should supply biome IDs via
     * {@link #runInferenceWithBiome} for best accuracy; when {@code false}
     * the model ignores biome conditioning entirely.
     */
    public boolean acceptsBiomeIds() {
        return hasBiomeIds;
    }

    /**
     * Whether the loaded model config declares heightmap inputs
     * ({@code heightmap_surface} and/or {@code heightmap_ocean_floor}).
     */
    public boolean acceptsHeightmaps() {
        return hasHeightmapSurface || hasHeightmapOceanFloor;
    }

    /** Number of block classes the model was trained with. */
    public int numClasses() {
        return numClasses;
    }

    /**
     * The expected shape of the noise_3d input tensor, as declared in
     * the model config sidecar.
     *
     * <p>Legacy (v6) models: {@code [1, 13, 4, 2, 4]} (416 floats).<br>
     * New (v7+) models: {@code [1, 15, 4, 2, 4]} (480 floats).
     *
     * @return defensive copy of the shape array
     */
    public long[] noise3dShape() {
        return noise3dShape.clone();
    }

    /**
     * Expected flat length of the noise_3d input array
     * (product of all shape dimensions).
     */
    public int noise3dFlatLength() {
        long product = 1;
        for (long d : noise3dShape) product *= d;
        return (int) product;
    }

    // ------------------------------------------------------------------
    // AutoCloseable
    // ------------------------------------------------------------------

    @Override
    public void close() {
        try { model.close(); } catch (Exception ignored) {}
        try { manager.close(); } catch (Exception ignored) {}
    }
}

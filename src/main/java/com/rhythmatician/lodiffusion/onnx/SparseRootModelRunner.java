package com.rhythmatician.lodiffusion.onnx;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Deque;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;

/**
 * Loads and runs the single {@code sparse_root.onnx} model that replaces the
 * three-model octree pipeline (init / refine / leaf) for Stage 2 block
 * selection.
 *
 * <h3>Contract: {@code lodiffusion.v6.sparse_root}</h3>
 *
 * <pre>
 *   sparse_root.onnx
 *     Input:
 *       noise_3d   float32[1, 13, 4, 2, 4]   (13 noise channels, 4×2×4 cells)
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
 * @see com.rhythmatician.lodiffusion.voxy.WorldNoiseAccess#sampleNoise3DForSection
 */
public final class SparseRootModelRunner implements AutoCloseable {

    private static final Logger LOGGER = LoggerFactory.getLogger(SparseRootModelRunner.class);

    /** ONNX model stem (no extension). */
    private static final String STEM = "sparse_root";
    /** Config sidecar filename. */
    private static final String CONFIG = "sparse_root_config.json";

    /** Octree depth. L4 = root, L0 = leaf (individual blocks). */
    private static final int LEVELS = 5;
    /** Edge of the generated subchunk, in blocks. */
    static final int SUBCHUNK = 16;
    /** Total leaf nodes in the fully-expanded octree (16³). */
    static final int LEAF_NODES = 4096;

    /**
     * Split threshold: sigmoid(split_logit) &gt; this → expand to children.
     * Set once at load time from runtime config key {@code "sparseRootSplitThreshold"}
     * (default 0.6).  The same value is written to {@code sparse_root_config.json}
     * by {@code export_sparse_root.py} so training and runtime stay in sync.
     */
    private final float splitThreshold;

    /** Whether the model expects a 2D noise input (noise_2d). */
    private final boolean hasNoise2d;

    /** Shape of the noise_2d input tensor, if present. */
    private final long[] noise2dShape;

    // ── Node counts per level ──────────────────────────────────────────
    /** Number of nodes in the fully-expanded octree at each level 4-0. */
    private static final int[] LEVEL_NODES = { 1, 8, 64, 512, 4096 };
    // level index in this array: 0=L4(root), 1=L3, 2=L2, 3=L1, 4=L0(leaf)

    private final NDManager manager;
    private final ZooModel<NDList, NDList> model;
    private final BlockVocabulary vocabulary;
    private final int numClasses;

    private SparseRootModelRunner(NDManager manager,
                                  ZooModel<NDList, NDList> model,
                                  BlockVocabulary vocabulary,
                                  int numClasses,
                                  float splitThreshold,
                                  boolean hasNoise2d,
                                  long[] noise2dShape) {
        this.manager        = manager;
        this.model          = model;
        this.vocabulary     = vocabulary;
        this.numClasses     = numClasses;
        this.splitThreshold = splitThreshold;
        this.hasNoise2d     = hasNoise2d;
        this.noise2dShape   = noise2dShape;
    }

    // ------------------------------------------------------------------
    // Factory
    // ------------------------------------------------------------------

    /**
     * Try to load {@code sparse_root.onnx} from {@code modelDir}.
     *
     * @param modelDir directory containing model files
     * @return loaded runner, or {@code null} if the file is absent
     * @throws IOException if the file is present but cannot be loaded
     */
    public static SparseRootModelRunner tryLoad(Path modelDir) throws IOException {
        Path onnxPath = modelDir.resolve(STEM + ".onnx");
        if (!Files.exists(onnxPath)) {
            LOGGER.info("[SparseRoot] {} not found — sparse-root model unavailable", onnxPath);
            return null;
        }

        InferenceDeviceSelector.Provider provider = InferenceDeviceSelector.selectProvider();
        LOGGER.info("[SparseRoot] Loading {} (provider={})", onnxPath, provider);

        ClassLoader prevCl = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(
                SparseRootModelRunner.class.getClassLoader());
        try {
            NDManager manager = NDManager.newBaseManager();
            try {
                ZooModel<NDList, NDList> zm = buildAndLoad(modelDir, provider);

                // Load vocab from config sidecar, fall back to a default
                BlockVocabulary vocab = null;
                int numClassesFromConfig = 256;
                boolean hasNoise2d = false;
                long[] noise2dShape = null;
                Path configPath = modelDir.resolve(CONFIG);
                if (Files.exists(configPath)) {
                    ModelConfig cfg = ConfigLoader.load(configPath);
                    int nc = cfg.effectiveBlockVocabSize();
                    if (cfg.blockMapping() != null && !cfg.blockMapping().isEmpty()) {
                        vocab = BlockVocabulary.fromConfig(cfg);
                        if (nc <= 0) nc = vocab.size();
                    }
                    if (nc <= 0) nc = 256;
                    numClassesFromConfig = nc;

                    hasNoise2d = cfg.hasInput("noise_2d");
                    if (hasNoise2d) {
                        int[] shape = cfg.getInputShape("noise_2d");
                        noise2dShape = new long[shape.length];
                        for (int i = 0; i < shape.length; i++) noise2dShape[i] = shape[i];
                    }

                    LOGGER.info("[SparseRoot] Loaded — vocab={} numClasses={} provider={}",
                            vocab != null ? vocab.size() : "none", nc, provider);
                } else {
                    LOGGER.warn("[SparseRoot] {} not found — using raw block indices (no vocab)", configPath);
                }
                float splitThreshold = (float) com.rhythmatician.lodiffusion.Config
                        .getDouble("sparseRootSplitThreshold", 0.6);
                LOGGER.debug("[SparseRoot] splitThreshold={}", splitThreshold);
                return new SparseRootModelRunner(manager, zm, vocab, numClassesFromConfig,
                        splitThreshold, hasNoise2d, noise2dShape);

            } catch (Exception e) {
                manager.close();
                if (e instanceof IOException) throw (IOException) e;
                throw new IOException("Failed to load sparse_root from " + modelDir, e);
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
            try {
                ZooModel<NDList, NDList> model = buildCriteria(dir, provider.djlOptionValue())
                        .build().loadModel();
                LOGGER.info("[SparseRoot] Loaded with provider {}", provider);
                return model;
            } catch (Exception ex) {
                LOGGER.warn("[SparseRoot] Provider {} unavailable ({}); falling back to CPU",
                        provider, ex.getMessage());
            }
        }
        // CPU fallback (or initial attempt when provider == CPU)
        ZooModel<NDList, NDList> model = buildCriteria(dir, null).build().loadModel();
        if (!provider.djlOptionValue().isEmpty()) {
            LOGGER.info("[SparseRoot] Loaded with CPU fallback");
        } else {
            LOGGER.info("[SparseRoot] Loaded (CPU)");
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

    /**
     * Run inference for a single 16³ subchunk.
     *
     * @param noiseFlat flat {@code float[13 * 4 * 2 * 4 = 416]} noise input
     *        in channel-outermost order, as returned by
     *        {@link com.rhythmatician.lodiffusion.voxy.WorldNoiseAccess#sampleNoise3DForSection}
     * @return {@code int[16][16][16]} block IDs in {@code [y][z][x]} order
     *         (matching the Voxy native storage format), or {@code null} if
     *         inference failed
     */
    public int[][][] runInference(float[] noiseFlat) {
        ClassLoader prevCl = Thread.currentThread().getContextClassLoader();
        Thread.currentThread().setContextClassLoader(getClass().getClassLoader());
        try (NDManager sub = manager.newSubManager()) {
            // Build input tensor: [1, 13, 4, 2, 4]
            NDArray noise3d = sub.create(noiseFlat,
                    new Shape(1, 13, 4, 2, 4));

            NDList inputs;
            if (hasNoise2d && noise2dShape != null) {
                inputs = new NDList(sub.zeros(new Shape(noise2dShape)), noise3d);
            } else {
                inputs = new NDList(noise3d);
            }

            long t0 = System.currentTimeMillis();
            int[][][] blocks;
            try (var predictor = model.newPredictor()) {
                NDList outputs = predictor.predict(inputs);
                long elapsedMs = System.currentTimeMillis() - t0;
                LOGGER.debug("[SparseRoot] Inference complete in {}ms", elapsedMs);
                blocks = decodeOutputs(outputs, sub);
            }
            return blocks;

        } catch (TranslateException e) {
            LOGGER.warn("[SparseRoot] Inference failed: {}", e.getMessage());
            return null;
        } finally {
            Thread.currentThread().setContextClassLoader(prevCl);
        }
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
     *   <li>At each node, check {@code sigmoid(split_logit) > splitThreshold}.</li>
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
                // Label tensor: [1, N, C]
                int n = (int) shape[1];
                int lvlIdx = levelIndexFromNodeCount(n);
                if (lvlIdx >= 0 && labelByLevel[lvlIdx] == null) {
                    labelByLevel[lvlIdx] = t.toFloatArray();
                }
            }
        }

        // Validate that we got all 5 levels
        for (int i = 0; i < LEVELS; i++) {
            if (splitByLevel[i] == null || labelByLevel[i] == null) {
                LOGGER.warn("[SparseRoot] Missing output tensor for level index {} "
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
                float logit = splitArr[nodeIdx];
                float sigm = 1.0f / (1.0f + (float) Math.exp(-logit));
                isSplit = sigm > thresh;
            }

            if (isSplit && lvlIdx < LEVELS - 1) {
                // Expand: enqueue 8 children at the next finer level
                int childSize = size >> 1;
                int childLvlIdx = lvlIdx + 1;
                int childBase = nodeIdx * 8;
                for (int oct = 0; oct < 8; oct++) {
                    int dx = (oct & 1);
                    int dz = (oct >> 1) & 1;
                    int dy = (oct >> 2) & 1;
                    int childOrigin = ((by0 + dy * childSize) << 8)
                                    | ((bz0 + dz * childSize) << 4)
                                    | (bx0 + dx * childSize);
                    pending.push(new int[] { childLvlIdx, childBase + oct, childOrigin });
                }
            } else {
                // Leaf: fill sub-region with argmax of label logits
                int blockId = argmaxLabel(labelByLevel[lvlIdx], nodeIdx, numClasses);
                fillRegion(grid, by0, bz0, bx0, size, blockId);
            }
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

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /** Block vocabulary used by this model. */
    public BlockVocabulary vocabulary() {
        return vocabulary;
    }

    /** Number of block classes the model was trained with. */
    public int numClasses() {
        return numClasses;
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

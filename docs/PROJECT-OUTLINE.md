## 🔭 **LODiffusion — Minecraft Mod (Fabric 1.21.4)**

### 🎯 **Mission**

Render plausible terrain for far chunks via a progressive LOD pipeline driven by **VoxelTree** models, keeping strong parity with vanilla and tight compatibility with **Distant Horizons** (DH). Priorities: correctness → stability → speed.

---

## 🗺️ **Project Roadmap**

### **PHASE 0 — Developer Infrastructure (Complete ✅)**

* Fabric mod scaffolding (Java 17, Gradle 8.x), CI (Actions) + JaCoCo + Lint
* TDD setup (JUnit 5 + Mockito), deterministic test fixtures and seeds

---

### **PHASE 1 — Core LOD Engine & Runtime Contracts (🆕 In-progress)**

**Goal:** Replace the old “single diffusion pass” with a **5-stage refinement ladder** and shared input contract with VoxelTree.

**What’s new**

* **Five models**: `Init(→LOD4)`, `LOD4→3`, `3→2`, `2→1`, `1→0`
* **Shared cached inputs** (from worldgen), **identical** across all five models:

  * `x_height_planes` **\[1,5,1,16,16]** (surface, ocean\_floor, slope\_x, slope\_z, curvature)
  * `x_biome_quart` **\[1,6,4,4,4]** (quart lattice features)
  * `x_router6` **\[1,6,1,16,16]** (Router-6 at one Y slice)
  * *(opt)* `x_barrier` **\[1,1,1,16,16]**, *(opt)* `x_aquifer3` **\[1,3,1,16,16]**, *(opt)* `x_cave_prior4` **\[1,1,4,4,4]**
  * Scalars: `x_chunk_pos` **\[1,2]**, `x_lod` **\[1,1]**
* **Per-stage parent prior** only:

  * `x_parent_prev`: `[1,1,D,D,D]` with **D ∈ {1,2,4,8}\`** (zeros for Init; previous output otherwise)
* **Outputs (per stage)**: `block_logits [1,N,D,D,D]`, `air_mask [1,1,D,D,D]` where D={1,2,4,8,16}

**Rules**

* **No upsampling in the mod** (ever). LODiffusion passes the cached native-grid tensors as-is; each ONNX model performs any resizing/broadcast internally with static ops.
* **Vanilla `carve()` at LOD0 only.** Distant terrain skips carve; near terrain (LOD0) calls vanilla carve to finalize caves/aquifers/structures.

**Deliverables**

* `TerrainPipeline` (progress controller), `ModelOrchestrator` (five-model runner), `TensorPacker` (strict shapes), `FeatureCache` (LRU + optional disk sidecar)

---

### **PHASE 2 — World Integration & Noise Capture (🆕 In-progress)**

**Goal:** Gather the *same* signals vanilla has at generation time, cache at source granularity (no upsampling).

**Components**

* **NoiseTap** (runtime sampler):

  * Heightmaps: 16×16 (WG types); derive slope/curv
  * Biomes: 4×4×4 quart lattice (compact features)
  * NoiseRouter slices: 16×16 @ one Y (Router-6 + optional barrier/aquifer)
  * *(Optional)* Coarse 3D cave prior: 4×4×4 (or 8×8×8)
  * Chunk coords `(x,z)`; world height limits; sea level (scalar)
* **FeatureCache**:

  * In-memory LRU keyed by `ChunkPos`
  * Optional sidecar: `lod_cache/<dim>/<region>/c.<x>.<z>.nf.bin` (or `.npz`)
  * Strict immutability (except `x_parent_prev`)

**Testing**

* Parity tests: cached fields vs direct API reads (epsilon match)
* Determinism: same seed + coords → same cached tensors

---

### **PHASE 3 — DJL Inference & Model Lifecycle (🆕 Planned)**

**Goal:** Robust, fast, memory-safe inference for five models.

**Tasks**

* **ONNX loader (DJL ONNX Runtime)**: shared `ModelZoo`, lazy load per model
* **TensorPacker**: map `FeatureCache` → exact ONNX input names/shapes (no resize)
* **Refinement loop**:

  1. `Init` (D=1) → `x_parent_prev(1³)`
  2. `1→2` → `2→4` → `4→8` → `8→16` (propagate `x_parent_prev`)
  3. Write 16³ to chunk; call **vanilla `carve()`**
* **Perf controls**: per-stage timers, pool NDArrays, cap memory/threads

**Acceptance**

* All 5 models pass numeric parity with VoxelTree’s `test_vectors.npz`
* Total per-chunk inference time < 100ms on target CPU

---

### **PHASE 4 — DH Integration & LOD Policy (🆕 Planned)**

**Goal:** Only generate as much as needed for current DH LOD.

**Features**

* `LODManagerCompat`: query DH LOD for a chunk
* **Work policy**:

  * LOD4/3/2: prepare `x_parent_prev` progressively
  * LOD1→0 promotion: run final model (16³) + **vanilla carve()**
* **Edge blending**: use `air_mask` for smooth borders; respect DH tile boundaries
* **Switches**: vanilla vs AI, per-LOD enable/disable, overlay debug

---

### **PHASE 5 — UI, Debug, and Metrics (🆕 Planned)**

* Toggles: model packs on/off, optional channels on/off
* Visual overlays: `air_mask`, seam highlighters, Router-6 inspector
* Counters: cache hit/miss, sampling ms, inference ms per stage

---

### **PHASE 6 — Packaging & Distribution (🆕 Planned)**

* Bundle: `model.onnx` ×5, `model_config.json` ×5, `test_vectors.npz`, model hash
* Settings: JSON/TOML for toggles + paths
* Releases: Modrinth/CurseForge artifacts; version gate on Fabric/Yarn

---

## 🧱 **Module Breakdown (Updated)**

| Module                    | Purpose                                                                 | Status                      |
| ------------------------- | ----------------------------------------------------------------------- | --------------------------- |
| `NoiseTap`                | Capture vanilla signals at source granularity (16×16, 4×4×4, etc.)      | 🆕 WIP                      |
| `FeatureCache`            | Per-chunk cache (LRU + sidecar), immutable payloads                     | 🆕 WIP                      |
| `TensorPacker`            | Convert `FeatureCache` to ONNX inputs (exact shapes), no resize         | 🆕 WIP                      |
| `ModelOrchestrator`       | Load/run 5 models in sequence, manage `x_parent_prev`                   | 🆕 WIP                      |
| `TerrainPipeline`         | LOD policy: when to run which model; final write to chunk + **carve()** | 🆕 WIP                      |
| `DistantHorizonsCompat`   | DH LOD queries + safe guards                                            | ✅                           |
| `DiffusionChunkGenerator` | Integration point (hooks and chunk writes)                              | ✅ (to be refit to pipeline) |
| `Diagnostics`             | Timers, counters, overlays                                              | 🆕 Planned                  |

---

## 🔗 **Interface with VoxelTree (Exact Contract)**

**VoxelTree delivers (per model):**

1. `model.onnx` (static shapes)
2. `model_config.json`

   * Input names & shapes (as listed above)
   * Normalization (heights min-max, router/aquifer z-score, flags, coord scale)
   * Block palette / `N_blocks`
3. `test_vectors.npz` (golden: inputs → outputs)

**LODiffusion guarantees:**

* Feed **exact cached inputs** (no upsampling) + stage-correct `x_parent_prev`
* Apply the same normalization fields from `model_config.json`
* Validate against `test_vectors.npz` during startup (smoke parity)
* Respect static shapes (fail fast on mismatch)

**Data flow (per chunk):**

```
NoiseTap.capture() → FeatureCache
   ↓ (no resize)
TensorPacker → ONNX (Init) → parent(1³)
   → ONNX (2³) → parent(2³)
   → ONNX (4³) → parent(4³)
   → ONNX (8³) → parent(8³)
   → ONNX (16³) → write blocks
   → vanilla carve() at LOD0
```

---

## ⚙️ **Performance Targets & Policies**

* **Sampling/cache** (first touch): ≤ 20–35 ms (depends on optional channels)
* **Inference** (all models combined, near player): ≤ 100 ms/patch on mid-range CPU
* **Memory**: ≤ \~2 MB/patch (NDArray pooling), LRU of \~128 chunks (configurable)
* **Determinism**: identical inputs → identical outputs (unit test enforced)

---

## ✅ **What We’re Keeping / Dropping**

* ✅ Keep vanilla **carve()**; only run at **LOD0**
* ✅ Keep cache-at-source (16×16 planes, 4×4×4 biomes, 16×16 Router-6 slice)

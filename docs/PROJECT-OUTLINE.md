## 🔭 **LODiffusion — Minecraft Mod (Fabric 1.21.11)**

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
| `NoiseTap` / `NoiseDumperCommand` | Capture vanilla signals: router6, heightmaps, biomes (`/dumpnoise`)  | ✅ Implemented              |
| `AnchorSampler`           | Sample height planes + router6 for model input                          | ✅ Implemented              |
| `LodGenerationService`    | 4-pass LOD generation (LOD4→LOD1), spiral ordering, parent cache        | ✅ Implemented              |
| `VoxyBlockMapper`         | Map model vocab indices → Voxy block IDs via `model_config.json`        | ✅ Implemented              |
| `VoxySectionWriter`       | Argmax → air mask → pack voxels → push to Voxy via reflection            | ✅ Implemented              |
| `VoxyCompat`              | Pure-reflection bridge to Voxy API (no compile-time dependency)          | ✅ Implemented              |
| `BlockVocabulary`         | Load block→ID mapping from `model_config.json`                          | ✅ Implemented              |
| `DistantHorizonsCompat`   | DH LOD queries + safe guards                                            | ✅ Implemented              |
| `LodiffusionCommand`      | In-game control: `/lodiffusion status\|toggle\|performance\|reload`       | ✅ Implemented              |
| `Diagnostics`             | Per-section timers, performance counters, debug overlay                  | ✅ Basic                    |

---

## 🔗 **Interface with VoxelTree (Exact Contract)**

**VoxelTree delivers:**

1. `model.onnx` (static shapes, opset ≥ 17)
2. `model_config.json`

   * Input names & shapes (v2 anchor-conditioned contract):
     - `x_parent` **[1,1,8,8,8]** float32 — binary occupancy (Mipper-derived)
     - `x_height_planes` **[1,5,16,16]** float32 — surface, ocean_floor, slope_x, slope_z, curvature
     - `x_router6` **[1,6,16,16]** float32 — temperature, vegetation, continents, erosion, depth, ridges
     - `x_biome` **[1,16,16]** int64 — vanilla biome index per (x,z)
     - `x_y_index` **[1]** int64 — vertical slab index
     - `x_lod` **[1]** int64 — coarseness token
   * Outputs: `block_logits` **[1,1102,16,16,16]**, `air_mask` **[1,1,16,16,16]**
   * `block_mapping`: Voxy-native canonical vocabulary (1102 entries from `config/voxy_vocab.json`)
   * `block_id_to_name`: reverse mapping for debugging
   * Normalization specs per input
3. `test_vectors.npz` (golden: inputs → outputs)

**LODiffusion guarantees:**

* `VoxyBlockMapper` reads `block_mapping` from `model_config.json` and maps model indices → Voxy block IDs at startup
* `AnchorSampler` provides height planes + router6 matching the model contract
* `VoxySectionWriter` pushes argmax results into Voxy via reflection (VoxyCompat)
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

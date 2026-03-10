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

**Goal:** Replace the old "single diffusion pass" with a **4-model progressive refinement ladder** and shared input contract with VoxelTree.

**What’s new**

* **Four models**: `Init→LOD4`, `LOD4→3`, `LOD3→2`, `LOD2→1` (LOD0 is vanilla-authoritative)
* **Shared conditioning inputs** (identical across all models):

  * `x_height_planes` **[1,5,16,16]** float32 — surface, ocean\_floor, slope\_x, slope\_z, curvature
  * `x_biome` **[1,16,16]** int64 — vanilla biome index per (x,z)
  * `x_y_index` **[1]** int64 — vertical slab index [0,23]
* **Per-stage parent prior** (refinement models only):

  * `x_parent` **[1,1,P,P,P]** float32 — previous stage output; P ∈ {1,2,4} (absent for Init)
* **Outputs (per stage)**: `block_logits [1,N,D,D,D]`, `air_mask [1,1,D,D,D]` where D∈{1,2,4,8}; Java runner upsamples final stage 2× to 16³

**Rules**

* **ONNX models produce static shapes.** Final stage (`refine_lod2_to_lod1`) outputs 8³; `ProgressiveModelRunner` upsamples 2× to 16³ before writing to Voxy.
* **Vanilla `carve()` at LOD0 only.** Distant terrain skips carve; near terrain (LOD0) calls vanilla carve to finalize caves/aquifers/structures.

**Deliverables**

* `LodGenerationService` (progress controller + spiral ordering), `ProgressiveModelRunner` (four-model inference chain), `VoxyBlockMapper` + `VoxySectionWriter` (post-process + write), `AnchorSampler` / `NoiseTap` (feature capture)

---

### **PHASE 2 — World Integration & Noise Capture (🆕 In-progress)**

**Goal:** Gather the *same* signals vanilla has at generation time, cache at source granularity (no upsampling).

**Components**

* **NoiseTap** (runtime sampler):

  * Heightmaps: 16×16 (surface, ocean\_floor, slope\_x, slope\_z, curvature) → `x_height_planes [1,5,16,16]` float32
  * Biomes: 16×16 vanilla biome indices → `x_biome [1,16,16]` int64
  * Vertical slab index → `x_y_index [1]` int64 [0,23]
* **FeatureCache**:

  * In-memory LRU keyed by `ChunkPos`
  * Optional sidecar: `lod_cache/<dim>/<region>/c.<x>.<z>.nf.bin` (or `.npz`)
  * Strict immutability (except `x_parent` from prev stage)

**Testing**

* Parity tests: cached fields vs direct API reads (epsilon match)
* Determinism: same seed + coords → same cached tensors

---

### **PHASE 3 — DJL Inference & Model Lifecycle (🆕 Planned)**

**Goal:** Robust, fast, memory-safe inference for four models.

**Tasks**

* **ONNX loader (DJL ONNX Runtime)**: shared `ModelZoo`, lazy load per model
* **`ProgressiveModelRunner`**: map `AnchorSampler` output → exact ONNX input names/shapes, chain four stages
* **Refinement loop**:

  1. `init_to_lod4` (D=1) → `x_parent` for next stage
  2. `refine_lod4_to_lod3` (D=2) → `refine_lod3_to_lod2` (D=4) → `refine_lod2_to_lod1` (D=8)
  3. Upsample 8³ → 16³ (`ProgressiveModelRunner`); write to Voxy via `VoxySectionWriter`
* **Perf controls**: per-stage timers, pool NDArrays, cap memory/threads

**Acceptance**

* All 4 models pass numeric parity with VoxelTree's `*_test_vectors.npz`
* Total per-chunk inference time < 100ms on target CPU

---

### **PHASE 4 — DH Integration & LOD Policy (🆕 Planned)**

**Goal:** Only generate as much as needed for current DH LOD.

**Features**

* `LODManagerCompat`: query DH LOD for a chunk
* **Work policy**:

  * LOD4/3/2: build `x_parent` progressively from each stage output
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

* Bundle: 4 × `.onnx` (`init_to_lod4`, `refine_lod4_to_lod3`, `refine_lod3_to_lod2`, `refine_lod2_to_lod1`), 4 × `_config.json`, `pipeline_manifest.json`, 4 × `_test_vectors.npz`
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

**VoxelTree delivers (contract `lodiffusion.v3.progressive`):**

1. Four ONNX model files (opset 17, static shapes):
   - `init_to_lod4.onnx`
   - `refine_lod4_to_lod3.onnx`
   - `refine_lod3_to_lod2.onnx`
   - `refine_lod2_to_lod1.onnx`
2. Four sidecar configs: `init_to_lod4_config.json`, `refine_lod4_to_lod3_config.json`, `refine_lod3_to_lod2_config.json`, `refine_lod2_to_lod1_config.json`
   * Each contains: input/output names, block vocabulary (`block_mapping`), normalization specs
3. `pipeline_manifest.json` — lists all required files; validated at startup
4. Four test-vector files: `*_test_vectors.npz` (golden inputs → outputs per model)

**Per-model tensor contract:**

| Model | `x_height_planes` | `x_biome` | `x_y_index` | `x_parent` | Output `block_logits` | Output `air_mask` |
|---|---|---|---|---|---|---|
| `init_to_lod4` | [1,5,16,16] float32 | [1,16,16] int64 | [1] int64 | — | [1,N,1,1,1] | [1,1,1,1,1] |
| `refine_lod4_to_lod3` | [1,5,16,16] float32 | [1,16,16] int64 | [1] int64 | [1,1,1,1,1] float32 | [1,N,2,2,2] | [1,1,2,2,2] |
| `refine_lod3_to_lod2` | [1,5,16,16] float32 | [1,16,16] int64 | [1] int64 | [1,1,2,2,2] float32 | [1,N,4,4,4] | [1,1,4,4,4] |
| `refine_lod2_to_lod1` | [1,5,16,16] float32 | [1,16,16] int64 | [1] int64 | [1,1,4,4,4] float32 | [1,N,8,8,8] | [1,1,8,8,8] |

**LODiffusion guarantees:**

* `ProgressiveModelRunner` chains the four models, builds exact tensor shapes, and upsamples the final 8³ output 2× to 16³
* `VoxyBlockMapper` reads `block_mapping` from `*_config.json` and maps model indices → Voxy block IDs at startup
* `VoxySectionWriter` pushes argmax results into Voxy via reflection (VoxyCompat)
* `pipeline_manifest.json` validated at startup; load fails if any required file is missing or hash-mismatched
* Respect static shapes (fail fast on mismatch)

**Data flow (per chunk):**

```
AnchorSampler.capture() → x_height_planes [1,5,16,16], x_biome [1,16,16], x_y_index [1]
   ↓
ProgressiveModelRunner:
   init_to_lod4              → block_logits/air_mask [D=1]
   refine_lod4_to_lod3  (x_parent from prev) → [D=2]
   refine_lod3_to_lod2  (x_parent from prev) → [D=4]
   refine_lod2_to_lod1  (x_parent from prev) → [D=8]
   upsample 2× → InferenceResult [16³]
   ↓
VoxySectionWriter → Voxy (LOD1–LOD4 only; LOD0 = vanilla)
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
* ✅ Cache at source: 16×16 height planes (`x_height_planes`), 16×16 biome IDs (`x_biome`), y-slab index (`x_y_index`)

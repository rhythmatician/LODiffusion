## 🔭 **AI-Diffusion Minecraft Mod – LODiffusion**

### 🎯 **Mission**

Generate plausible terrain in distant chunks using a **discrete diffusion model**, with progressive fidelity across Levels of Detail (LODs), tightly integrated with Distant Horizons.

---

## 🧭 **Refined Project Roadmap**

### **PHASE 0 — Developer Infrastructure (Done)**

* [x] Git + GitHub Actions with PR CI + Jacoco + Lint
* [x] Fabric mod scaffolding + Java 17 + Gradle 8.x
* [x] Copilot terminal permissions, auto-merge prep, TDD config
* [x] Testing suite bootstrapped with JUnit 5 + Mockito

---

### **PHASE 1 — Core Diffusion Engine (Complete ✅)**

* [x] Implemented multi-pass, tile-aware `DiffusionModel.run()`
* [x] Added LOD-sensitive diffusion with LOD intensity maps
* [x] Introduced multi-channel support: `float[][][]` (height, biome, temp)
* [x] Integrated LOD blending with `getTileEdgeFactor`
* [x] Verified via 90+ unit tests + real-world .mca patch tests
* [x] 70%+ line coverage with full CI compliance

---

### **PHASE 2 — World Integration & Hook Points (In Progress)**

* [x] `DiffusionChunkGenerator.buildSurface(...)` overloads implemented
* [x] `LODManagerCompat` with fallback for Distant Horizons
* [x] `DistantHorizonsCompat` provides runtime-safe API calls
* [x] Partial registration for world generation APIs (DH pending)

**TODO:**

* [ ] Full implementation of `IDhApiWorldGenerator` registration
* [ ] Fallback chunk generation logic when DH is not loaded
* [ ] Add configurable parameters for LOD tuning in-game

---

### **PHASE 3 — Data Extraction for Training (Planned 🧪)**

* [ ] Use `ChunkDataExtractor` to convert `.mca` regions into 8x8 patches
* [ ] Support vanilla + modded biome decoding (with palette + NBT)
* [ ] Export `.npy` or `.pt` format data for ML consumption
* [ ] Optional: Export Distant Horizons LODs for ground-truth A/B testing

---

### **PHASE 4 — Model Training Pipeline**

* [ ] `train.py`: U-Net with sinusoidal timestep embedding
* [ ] Conditional input: height + biome classmap (optional)
* [ ] Output: multi-scale diffusion (e.g., ∆heightmaps)
* [ ] Export: ONNX + metadata (input format, LOD scale factors)
* [ ] Quantized model for runtime use (DJL or custom)

---

### **PHASE 5 — Runtime Inference Engine**

* [ ] Implement `DiffusionRunner` in Java
* [ ] Load ONNX model using DJL or custom JNI backend
* [ ] Predict terrain patches at runtime with cache/memoization
* [ ] Validate runtime outputs via visual debug overlays

---

### **PHASE 6 — Tuning, Debugging, UI**

* [ ] Add debug UI to visualize LODs and model output
* [ ] Toggle between vanilla and AI terrain modes
* [ ] Per-biome tuning of model parameters
* [ ] Performance benchmarking at multiple view distances

---

### **PHASE 7 — Packaging, Distribution**

* [ ] Embed metadata (model hash, training stats) into mod
* [ ] Document setup, inference engine, patch generation
* [ ] Release build with modrinth & curseforge descriptors
* [ ] Auto-pack example-world & test suite for public use

---

## 📊 **Active Project Metrics**

| Metric             | Value                                                      |
| ------------------ | ---------------------------------------------------------- |
| Test Coverage      | 87.1% instructions, 82.2% lines, 100% classes              |
| Tests              | 96 passing                                                 |
| Diffusion Channels | height, biome (temp stubbed)                               |
| LOD Levels         | 0 (full) → 3 (coarsest)                                    |
| Real World Chunks  | 64 (.mca) files (∼16k chunks)                              |
| Runtime Mod State  | Launches successfully with DH (2.3.X)                      |
| Copilot Autonomy   | Scaffold + fix cycles work well; commit automation pending |

---

## 🧱 Modular Breakdown

| Module                    | Purpose                                             |
| ------------------------- | --------------------------------------------------- |
| `DiffusionModel`          | Core logic: LOD-aware, multi-pass diffusion         |
| `DiffusionChunkGenerator` | Minecraft terrain generator (hooks into Fabric API) |
| `ChunkDataExtractor`      | World file reader (Anvil + NBT-ready)               |
| `DistantHorizonsCompat`   | API bridge (version-safe & runtime-checked)         |
| `LODManagerCompat`        | Abstracts LOD level across DH or fallback logic     |
| `RealWorldDataTest`       | Integration suite using real .mca files             |
| `train.py`                | PyTorch model (WIP – scaffold next)                 |

---

## 📌 Recommendations to Implement Next

1. **Split Phase 4 into a `/training` subdir with**:

   * `train.py`
   * `extract-patches.py`
   * `README.md` documenting patch formats + usage

2. **Write a sample `TerrainPatchDataset` class** in PyTorch to load `.npy`/`.pt` patches.

3. **Update `PROJECT-OUTLINE.md` with this refined blueprint.**

4. **Write a simple benchmark test** in `tests/benchmark/InferenceBenchmarkTest.java` to simulate runtime model invocation (even if stubbed).

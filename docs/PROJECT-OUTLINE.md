## 🔭 **AI-Diffusion Minecraft Mod – LODiffusion**

### 🎯 **Mission**

Generate plausible terrain in distant chunks using a **discrete diffusion model**, with progressive fidelity across Levels of Detail (LODs), tightly integrated with Distant Horizons.

---

## 🗺️ **Refined Project Roadmap**

### **PHASE 0 — Developer Infrastructure (Complete ✅)**

* Git + GitHub Actions with PR CI + JaCoCo + Lint
* Fabric mod scaffolding + Java 17 + Gradle 8.x
* Terminal permissions, TDD config, test suite (JUnit 5 + Mockito)

### **PHASE 1 — Core Diffusion Engine (Complete ✅)**

* Multi-pass, tile-aware `DiffusionModel.run()` with LOD-sensitive logic
* Multi-channel support: `float[][][]` (height, biome, temp)
* Blending via `getTileEdgeFactor()`
* > 90 unit tests and real-world .mca patch tests

### **PHASE 2 — World Integration & Hook Points (Complete ✅)**

* `DiffusionChunkGenerator` API implemented and tested
* `LODManagerCompat` + `DistantHorizonsCompat` runtime-safe wrappers
* `IDhApiWorldGenerator` registration complete
* > 84% coverage on DH-related logic

### **PHASE 3 — Data Extraction for Training (Nearly Complete 🥭)**

* ✅ NBT parsing implemented with Hephaistos v1.1.8
* ✅ Handles pre-1.18 and 1.18+ biome formats
* ✅ Packed heightmaps decoded from LongArrayTag
* ✅ `RegionFileCache` supports thread-safe caching & profiling
* ✅ Performance optimizations: 48ms → 0ms cache hits, 8 chunks in 11ms
* ✅ Proper SLF4J logging conversion completed

**TODO:**

* [ ] Add unit tests for edge-case biome parsing
* [ ] Implement `extract8x8PatchesFromChunk()`
* [ ] Export `.npy` / `.pt` training data
* [ ] `feat/training-data-cli` branch
* [ ] Handle modded biomes & optional DH LODs

### **PHASE 4 — Model Training Pipeline**

* [ ] U-Net model: sinusoidal timestep, conditional input (height + biome)
* [ ] Export to ONNX with LOD metadata
* [ ] Quantized model for DJL runtime use

### **PHASE 5 — Runtime Inference Engine**

* [ ] Java ONNX loading via DJL or JNI
* [ ] `DiffusionRunner` class
* [ ] Memoized patch predictions per LOD
* [ ] Visual debug overlays

### **PHASE 6 — Tuning, Debugging, UI**

* [ ] Toggle vanilla/AI terrain
* [ ] In-game LOD parameter tuning
* [ ] Benchmarking support

### **PHASE 7 — Packaging, Distribution**

* [ ] Embed model metadata (hash, LODs, etc.)
* [ ] Publish to Modrinth + Curseforge
* [ ] Bundle example-world, test suite, CLI

---

## 📊 **Current Project Metrics**

| Metric            | Value                               |
| ----------------- | ----------------------------------- |
| Test Coverage     | 87.1% instructions, 82.2% lines     |
| Tests Passing     | 108                                 |
| Channels          | height, biome (temp stubbed)        |
| LOD Levels        | 0 (full) → 3 (coarsest)             |
| Real-world Chunks | \~16,000 from 64 `.mca` files       |
| Runtime Mod State | Works with DH 2.3.x, fallback ready |

---

## 🧱 Modular Breakdown

| Module                    | Purpose                                       | Status         |
| ------------------------- | --------------------------------------------- | -------------- |
| `DiffusionModel`          | Multi-pass, LOD-aware diffusion logic         | ✅ Complete     |
| `DiffusionChunkGenerator` | Fabric terrain hook & integration             | ✅ Complete     |
| `ChunkDataExtractor`      | Region file parsing + patch extraction        | 🥭 Nearly Complete |
| `RegionFileCache`         | I/O + coordinate cache with profiling         | ✅ Complete     |
| `DistantHorizonsCompat`   | Runtime-safe bridge to DH API                 | ✅ Complete     |
| `LODManagerCompat`        | Unified LOD query interface                   | ✅ Complete     |
| `RealWorldDataTest`       | Integration test suite on real `.mca` data    | ✅ Complete     |
| `TerrainPatch`            | 8x8 patch representation for training         | ✅ Complete     |
| `TerrainPatchDataset`     | Dataset loader for `.npy`/`.pt` training data | ✅ Complete     |
| `train.py`                | U-Net training scaffold                       | 📅 Planned     |

---

## 🔖 Next Steps (Branches to Create)

Following the **micro-commit strategy** outlined in `.github/copilot-instructions.md`:

### Immediate Next Tasks
1. `test/nbt-parsing-tests` — unit coverage of biome logic
2. `feat/patch-extraction` — 8x8 chunk patch builder
3. `feat/training-data-cli` — CLI export to `.npy`/`.pt`
4. `docs/nbt-parsing-guide` — explain format detection logic
5. `fix/gradle-cache-locks` — resolve lingering build issues

### Branch Management Guidelines
- Maximum 200 lines of changes per PR
- PRs should be reviewable in < 10 minutes
- Auto-merge enabled for Copilot-reviewed PRs with no open threads
- Commit every 15-20 minutes during development

### Recently Completed
- ✅ **ChunkDataExtractor Performance Optimization**: RegionFileCache, coordinate caching, profiling
- ✅ **SLF4J Logging Conversion**: Proper logging instead of System.out/System.err
- ✅ **Gradle Dependencies**: Fixed Hephaistos dependency issues
- ✅ **Performance Testing**: Comprehensive benchmark suite

---

## 📖 Resources & Links

* See `README.md` for CI setup and local testing commands
* See `EXAMPLE-WORLD-USAGE.md` for `.mca` region file structure
* See `CHUNK-EXTRACTOR-OPTIMIZATION-REFLECTION.md` for performance benchmarks
* See `CI-CHECKLIST.md` for pre-PR validation steps

---

## 🧵 Development Workflow

### Micro-Commit Strategy
- Each feature/fix gets its own focused branch (`test/add-xyz-test`, `feat/implement-abc`, `docs/update-def`)
- One logical change per commit:
  - Add test → `test:` commit
  - Implement → `feat:` commit
  - Fix → `fix:` commit
  - Doc → `docs:` commit

### CI Pipeline
Each commit/PR runs:
1. **Lint**: `./gradlew lint` — must pass first
2. **Test + Coverage**: `./gradlew test jacocoTestReport`
3. **Build Mod**: `./gradlew build` (only if lint + test pass)

Local equivalent:
```bash
./gradlew clean lint test jacocoTestReport build
```

### Testing Rules
- Target **80%+ code coverage per commit**
- Tests may live in:
  - `src/test/java/com/...` — core unit and integration tests
  - `src/test/java/data/` — synthetic dataset tests
  - `src/test/java/benchmark/` — performance benchmarks
- Use JUnit 5 and Mockito
- Use tags for clarity: `@Tag("ci")`, `@Tag("inference")`, `@Tag("benchmark")`

---

## 🏗️ Architecture Notes

### Chunk Generation & Diffusion
- In `DiffusionChunkGenerator.buildSurface(...)`:
  - Sample vanilla heightmap + biomes
  - Call `DiffusionModel.run(...)`
- **LOD chaining is required**: each refinement builds on the prior LOD
- Stubbed multi-channel logic must be test-guided and forward-compatible

### Distant Horizons Integration
- Runtime detection only (via `ModDetection.isDistantHorizonsLoaded()`)
- API dependency is `compileOnly`
- Use `LODManager.getChunkLOD(...)` for LOD level detection
- Implement fallback wrappers in `LODManagerCompat`, `DistantHorizonsCompat`

### Performance Considerations
- **RegionFileCache**: Avoid reopening .mca files for every chunk
- **Coordinate Caching**: Cache parsed region coordinates
- **Profiling Infrastructure**: Optional timing measurements for optimization
- **Batch Processing**: Process multiple chunks from same region efficiently

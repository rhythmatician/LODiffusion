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

### **PHASE 3 — Data Extraction for Training (Complete ✅)**

* ✅ NBT parsing implemented with Hephaistos v1.1.8
* ✅ Handles pre-1.18 and 1.18+ biome formats
* ✅ Packed heightmaps decoded from LongArrayTag
* ✅ `RegionFileCache` supports thread-safe caching & profiling
* ✅ Performance optimizations: 48ms → 0ms cache hits, 8 chunks in 11ms
* ✅ Proper SLF4J logging conversion completed
* ✅ **Architectural cleanup completed**: Separated production code from test infrastructure
* ✅ **Test infrastructure optimized**: `TerrainPatchDatasetFixture` with intelligent caching
* ✅ **Build reliability improved**: Git-tracked test data for CI compatibility
* ✅ **Code quality enhanced**: All @Test formatting violations fixed

**PHASE 3 COMPLETE** - Ready for training pipeline development

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

## 🧱 Modular Breakdown

| Module                    | Purpose                                       | Status         |
| ------------------------- | --------------------------------------------- | -------------- |
| `DiffusionModel`          | Multi-pass, LOD-aware diffusion logic         | ✅ Complete     |
| `DiffusionChunkGenerator` | Fabric terrain hook & integration             | ✅ Complete     |
| `ChunkDataExtractor`      | Region file parsing + patch extraction        | ✅ Complete     |
| `RegionFileCache`         | I/O + coordinate cache with profiling         | ✅ Complete     |
| `DistantHorizonsCompat`   | Runtime-safe bridge to DH API                 | ✅ Complete     |
| `LODManagerCompat`        | Unified LOD query interface                   | ✅ Complete     |
| `RealWorldDataTest`       | Integration test suite on real `.mca` data    | ✅ Complete     |
| `TerrainPatch`            | 8x8 patch representation for training         | ✅ Complete     |
| `TerrainPatchDataset`     | Dataset loader for `.npy`/`.pt` training data | ✅ Complete     |
| `TestWorldFixtures`       | Test infrastructure and data management       | ✅ Complete     |
| `TerrainPatchDatasetFixture` | Performance-optimized test caching         | ✅ Complete     |
| `train.py`                | U-Net training scaffold                       | 📅 Planned      |

---

## 🔖 Next Steps (Branches to Create)

Following the **micro-commit strategy** outlined in `.github/copilot-instructions.md`:

### Immediate Next Tasks
1. `feat/training-data-cli` — CLI export to `.npy`/`.pt` training data
2. `feat/u-net-training` — U-Net model training pipeline with PyTorch
3. `feat/onnx-export` — Model export to ONNX format for Java inference
4. `feat/djl-inference` — Java DJL integration for runtime inference
5. `feat/visual-debug` — In-game debug overlays and terrain toggles

### Training Pipeline Priority
- **Phase 4** is now ready to begin with clean data extraction foundation
- Focus on U-Net architecture with sinusoidal timestep encoding
- Conditional input support for height + biome data
- LOD-aware training for progressive detail generation

### Branch Management Guidelines
- Maximum 200 lines of changes per PR
- PRs should be reviewable in < 10 minutes
- Auto-merge enabled for Copilot-reviewed PRs with no open threads
- Commit every 15-20 minutes during development

### Recently Completed ✅
- ✅ **Architectural Cleanup Complete**: Clean separation of production vs test code
- ✅ **Performance Optimization**: `TerrainPatchDatasetFixture` with intelligent caching
- ✅ **Build Reliability**: Git-tracked test data for CI compatibility
- ✅ **Code Quality**: All @Test formatting violations fixed, lint clean
- ✅ **Test Infrastructure**: `TestWorldFixtures` for centralized test data management
- ✅ **ChunkDataExtractor Optimization**: RegionFileCache, coordinate caching, profiling
- ✅ **SLF4J Logging Conversion**: Proper logging instead of System.out/System.err

---

## 📖 Resources & Links

* See `README.md` for CI setup and local testing commands
* See `EXAMPLE-WORLD-USAGE.md` for `.mca` region file structure
* See `ARCHITECTURAL-CLEANUP-REFLECTION.md` for recent architectural improvements
* See `CHUNK-EXTRACTOR-OPTIMIZATION-REFLECTION.md` for performance benchmarks
* See `CI-CHECKLIST.md` for pre-PR validation steps
* See `COVERAGE-IMPROVEMENT-REFLECTION.md` for coverage optimization insights

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
- Target **70%+ code coverage per commit**
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
- **Test Infrastructure**: `TerrainPatchDatasetFixture` caching eliminates redundant NBT parsing
- **CI Reliability**: Git-tracked test data ensures consistent builds across environments

# AI-Diffusion Minecraft Mod (LODiffusion)

## 1. Vision
- **Mission:** Feed vanilla terrain heightmap and biome data into a multi-LOD discrete diffusion pipeline.
- **Success Criteria:** ...

## 2. High-Level Phases
1. Mod setup & Hello World
2. Chunk & LOD hook-ins
3. Diffusion algorithm integration
4. Data generation & model training
5. Distant Horizons LOD integration
6. In-game tuning & visual debug
7. Packaging, release & documentation
8. Polishing, extensions & cave diffusion

## 3. Phase Details

### 1. Mod setup & Hello World
- [x] Install JDK 17+ and Fabric Loom
- [x] Clone `fabric-example-mod` and configure `build.gradle`
- [x] TODO: Copilot, scaffold `src/test/java` directory structure and add JUnit 5 (`testImplementation`) and JaCoCo (`jacoco`) plugins to `build.gradle`
- [x] Create `HelloTerrainMod` implementing `ModInitializer`
- [x] Verify console log "[HelloTerrain] Mod initialized!" on server start (TDD: HelloTerrainModTest)
- [x] Verify gradle works

### 2. Chunk & LOD hook-ins
- [x] Research Fabric chunk-generation API and relevant events
- [x] Generate `ChunkGeneratorStubTest` skeleton
- [x] Stub `DiffusionChunkGenerator.buildSurface(...)` with multiple overloads
- [x] Implement basic heightmap modification for TDD validation
- [x] Integrate with `DiffusionModel.run()` method

### 3. Diffusion algorithm integration  
- [x] Write unit tests for `DiffusionModel.run(...)`
- [x] Implement basic diffusion algorithm with biome-aware processing
- [x] Add LOD-aware processing with buildSurfaceWithLOD() method
- [x] Implement progressive refinement across LOD levels (0-3)
- [x] Comprehensive LOD integration testing
- [ ] TODO: Copilot, modify `DiffusionModel.run(...)` for channels
- [ ] TODO: Copilot, integrate LODManager and call `run()`

### 4. Data generation & model training
- [ ] TODO: Copilot, scaffold PyTorch U-Net in `train.py`
- [ ] Train on 8x8 heightmap patches; export ONNX
- [ ] **Extension:** Conditional Diffusion
  - TODO: Copilot, extend U-Net for optional biome channel
  - TODO: Copilot, write `BiomeSamplingTest`

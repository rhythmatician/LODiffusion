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
- [ ] Install JDK 17+ and Fabric Loom
- [ ] Clone `fabric-example-mod` and configure `build.gradle`
- [ ] TODO: Copilot, scaffold `src/test/java` directory structure and add JUnit 5 (`testImplementation`) and JaCoCo (`jacoco`) plugins to `build.gradle`
- [ ] Create `HelloTerrainMod` implementing `ModInitializer`
- [ ] Verify console log "[HelloTerrain] Mod initialized!" on server start

### 2. Chunk & LOD hook-ins
- [ ] Research Fabric chunk-generation API and relevant events
- [ ] TODO: Copilot, generate `ChunkGeneratorStubTest` skeleton
- [ ] Stub `DiffusionChunkGenerator.buildSurface(...)`

### 3. Diffusion algorithm integration
- [ ] TODO: Copilot, write unit tests for `DiffusionModel.run(...)`
- [ ] TODO: Copilot, modify `DiffusionModel.run(...)` for channels
- [ ] TODO: Copilot, integrate LODManager and call `run()`

### 4. Data generation & model training
- [ ] TODO: Copilot, scaffold PyTorch U-Net in `train.py`
- [ ] Train on 8x8 heightmap patches; export ONNX
- [ ] **Extension:** Conditional Diffusion
  - TODO: Copilot, extend U-Net for optional biome channel
  - TODO: Copilot, write `BiomeSamplingTest`

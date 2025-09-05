# NoiseTap Integration Guide

## Overview

The `NoiseTap` interface provides efficient vanilla noise signal capture at native API granularities, exactly as you specified. This replaces our previous approach with a clean, zero-upsampling solution that respects Minecraft's native resolutions.

## Key Features

✅ **Native API Granularities**: 
- Router fields: 16×16×16 (block-level)
- Biomes: 4×4×4 (lattice storage)  
- Heightmaps: 16×16 (chunk-level)

✅ **Performance-Tiered Field Selection**:
- Core tier (~15ms): Essential Tier A fields
- Extended tier (~32ms): + Fluid/environmental fields
- Cave-aware tier (~66ms): + 3D density fields
- Full tier (~137ms): All 15 NoiseRouter fields

✅ **Zero Upsampling**: Raw cache at API resolution, downsample in models only

## Usage Examples

### Basic Usage

```java
// Bind to chunk context
var noiseTap = NoiseTap.bind(chunk, noiseConfig, biomeAccess, worldSeed);

// Capture core signals for real-time generation
var cache = noiseTap.captureAll(
    NoiseTap.getTierFields(PerformanceTier.CORE),
    NoiseTap.getDefaultHeightmaps()
);

// Access cached data
float[][][] temperatureData = cache.getRouterField(RouterField.TEMPERATURE); // [16][16][16]
short[][] surfaceHeights = cache.getHeightmap(Heightmap.Type.WORLD_SURFACE_WG); // [16][16]
int biomeId = cache.getBiomeId(0, 0, 0); // 4×4×4 lattice coordinates
```

### Performance-Optimized Usage

```java
// For real-time generation - use core tier only
EnumSet<RouterField> coreFields = NoiseTap.getTierFields(PerformanceTier.CORE);
// Contains: TEMPERATURE, VEGETATION, CONTINENTS, EROSION, DEPTH, RIDGES

// For specialized biomes - add environmental context  
EnumSet<RouterField> extendedFields = NoiseTap.getTierFields(PerformanceTier.EXTENDED);
// Adds: FLUID_FLOODEDNESS, FLUID_SPREAD, LAVA, BARRIER

// For cave-aware models - add 3D density
EnumSet<RouterField> caveFields = NoiseTap.getTierFields(PerformanceTier.CAVE_AWARE);
// Adds: INITIAL_DENSITY_NO_JAG, FINAL_DENSITY
```

### Memory Management

```java
var cache = noiseTap.captureAll(coreFields, defaultHeightmaps);

// Monitor memory usage
long memoryFootprint = cache.getMemoryFootprint(); // bytes
int fieldCount = cache.getRouterFieldCount();

System.out.printf("Cached %d fields using %d KB\n", 
    fieldCount, memoryFootprint / 1024);
```

## Integration with LODiffusion

### Replace TerrainDataCollector

The `NoiseTap` should replace our current `TerrainDataCollector` approach:

**Before** (old approach):
```java
// Old: Manual sampling with potential upsampling issues
var collector = new TerrainDataCollector(chunk, noiseConfig);
float[][] temperatureMap = collector.getTemperatureMap(8, 8); // Manual resolution
```

**After** (NoiseTap approach):
```java
// New: Native resolution capture, downsample in model
var noiseTap = NoiseTap.bind(chunk, noiseConfig, biomeAccess, worldSeed);
var cache = noiseTap.captureAll(
    NoiseTap.getTierFields(PerformanceTier.CORE),
    NoiseTap.getDefaultHeightmaps()
);

// Let the model handle downsampling from 16×16×16 to desired resolution
float[][][] temperatureData = cache.getRouterField(RouterField.TEMPERATURE);
```

### Model Input Pipeline

```java
public class OptimizedTerrainGenerator {
    
    public void generateTerrain(Chunk chunk, NoiseConfig noiseConfig, BiomeAccess biomeAccess) {
        // 1. Capture signals at native resolution
        var noiseTap = NoiseTap.bind(chunk, noiseConfig, biomeAccess, worldSeed);
        var cache = noiseTap.captureAll(
            selectedFields,  // Based on model requirements
            selectedHeightmaps
        );
        
        // 2. Convert to model inputs (downsample as needed)
        OnnxTensor inputs = createModelInputs(cache);
        
        // 3. Run inference
        var outputs = onnxSession.run(inputs);
        
        // 4. Apply results to chunk
        applyTerrainResults(chunk, outputs);
    }
    
    private OnnxTensor createModelInputs(NoiseTap.Cache cache) {
        // Model decides downsampling strategy
        // e.g., 16×16×16 → 8×8×1 for surface features
        // e.g., 4×4×4 → keep as-is for biome lattice
        // e.g., 16×16 → 8×8 for heightmap derivatives
    }
}
```

## Performance Budget

Based on our comprehensive noise analysis:

| Tier | Fields | Cost | Use Case |
|------|--------|------|----------|
| **Core** | 6 (Tier A) | ~15ms | Real-time generation |
| **Extended** | 10 (A+B) | ~32ms | Specialized biomes |
| **Cave-aware** | 12 (A+B+C) | ~66ms | Underground focus |
| **Full** | 15 (all) | ~137ms | Research/training only |

## API Reference

### RouterField Enum

Maps exactly to NoiseRouter's 15 DensityFunction fields:

**Tier A (Surface/Climate)**:
- `TEMPERATURE`, `VEGETATION`, `CONTINENTS`, `EROSION`, `DEPTH`, `RIDGES`

**Tier B (Fluid/Environment)**:
- `FLUID_FLOODEDNESS`, `FLUID_SPREAD`, `LAVA`, `BARRIER`

**Tier C (3D Density)**:
- `INITIAL_DENSITY_NO_JAG`, `FINAL_DENSITY`

**Tier D (Vein/Ore)**:
- `VEIN_TOGGLE`, `VEIN_RIDGED`, `VEIN_GAP`

### Cache Record

```java
record Cache(
    Map<RouterField, float[][][]> router,     // [16][16][16] per field
    int[][][] biomes4,                        // [4][4][4] biome IDs
    Map<Heightmap.Type, short[][]> heightmaps16, // [16][16] per type
    int chunkMinY, int chunkHeight,           // chunk bounds
    int chunkX, int chunkZ,                   // chunk position  
    long seed                                 // world seed
)
```

## Next Steps

1. **Replace TerrainDataCollector**: Migrate existing code to use NoiseTap
2. **Update model inputs**: Modify ONNX input creation to use native-resolution cache
3. **Performance validation**: Compare NoiseTap vs. current approach with NoiseSpeedProbe
4. **Training data alignment**: Update VoxelTree training pipeline to match NoiseTap shapes

This provides the exact "tiny, practical" interface you specified - efficient, native-resolution capture with zero naive upsampling! 🎯

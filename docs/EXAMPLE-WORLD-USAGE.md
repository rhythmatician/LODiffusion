# Using example-world Data for Testing

## 📁 **Available Test Data**

The `example-world/` folder contains a complete Minecraft world save with real terrain data that can be used for comprehensive testing of our LODiffusion algorithms.

### **Structure**
```
example-world/
├── region/          # 16 region files (.mca) with real chunk data
│   ├── r.-2.-2.mca  # Northwest region
│   ├── r.-1.-1.mca  # 
│   ├── r.0.0.mca    # Origin region (spawn area)
│   └── ...          # Additional regions
├── level.dat        # World metadata and settings
├── DIM-1/           # Nether dimension data
├── DIM1/            # End dimension data
└── playerdata/      # Player save data
```

### **Coverage**
- **16 region files** = ~16,384 chunks of real terrain
- **4x4 region grid** covering coordinates from (-64,-64) to (63,63) in chunk coordinates
- **Real biomes, heightmaps, and block data** from actual Minecraft generation

## 🧪 **Current Test Integration**

### **RealWorldDataTest.java**
- ✅ **Basic validation** - Confirms world data exists and is accessible
- ✅ **File format verification** - Validates .mca file naming conventions
- ✅ **Realistic data simulation** - Tests with terrain patterns resembling real world data
- ⏳ **NBT parsing** (planned) - Extract actual heightmaps from .mca files

### **ChunkDataExtractor.java**
- ✅ **World data detection** - Check if example world is available
- ✅ **Region file enumeration** - List available .mca files
- ✅ **Coordinate calculations** - Convert between region and chunk coordinates
- ⏳ **NBT parsing utilities** (planned) - Extract real heightmaps and biomes

## 🎯 **How to Use for Development**

### **1. Integration Testing**
```java
// Test your diffusion algorithms against realistic data patterns
@Test
void testWithRealisticTerrain() {
    int[][] heightmap = generateRealisticHeightmap(); // Based on real world patterns
    model.run(heightmap, biomes);
    // Verify diffusion preserves realistic characteristics
}
```

### **2. Algorithm Validation**
- Use realistic heightmap patterns to ensure diffusion doesn't create unrealistic terrain
- Test LOD processing with terrain that matches real Minecraft generation
- Validate biome-aware processing with actual biome distributions

### **3. Performance Benchmarking**
- Test processing speed with realistic data sizes
- Measure memory usage with real chunk data volumes
- Benchmark different LOD levels against actual terrain complexity

## 🔮 **Future Enhancements**

### **NBT Parsing Integration**
When NBT parsing is implemented, we can:

1. **Extract Real Heightmaps**
   ```java
   int[][] realHeightmap = ChunkDataExtractor.extractHeightmapFromChunk(regionFile, 0, 0);
   model.run(realHeightmap, biomes);
   ```

2. **Train with Real Data**
   ```python
   # In train.py - use real heightmaps for training
   heightmaps = extract_heightmaps_from_world("example-world/")
   train_diffusion_model(heightmaps)
   ```

3. **A/B Testing**
   - Compare diffusion output against vanilla terrain
   - Measure similarity metrics and terrain quality
   - Validate that LOD processing maintains terrain character

### **Suggested NBT Libraries**
- **Java**: `net.querz:nbt` or custom parser
- **Python**: `anvil-parser` or `nbtlib` for data extraction scripts

## ✨ **Current Benefits**

Even without NBT parsing, the example-world data provides:

1. **Realistic test patterns** - Generated based on real world structure
2. **Validation reference** - Understand what real terrain looks like
3. **Integration testing** - Verify file access and coordinate systems work
4. **Future-proofing** - Infrastructure ready for when NBT parsing is added

The world data is a valuable asset for developing robust, realistic terrain diffusion algorithms! 🏔️

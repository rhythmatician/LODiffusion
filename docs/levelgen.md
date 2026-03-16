## Blending

## Package Purpose (worldgen blending)

Goal: Smoothly blend between “old” (pre-existing) terrain and newly-generated terrain when Minecraft switches noise generation algorithms or world versions. This prevents harsh seams / sudden jumps at chunk borders where generation methods differ.
Scope: Operates during chunk generation to:
Blend height/density values near borders of “old-generated” chunks.
Blend biome selection across those borders.
Apply carving masks + tick propagation to avoid visible discontinuities.
🧱 Key Classes & Responsibilities
Blender
Primary entry point for blending logic.
Creates a blending strategy for a chunk region via Blender.of(WorldGenRegion).
Holds two main maps keyed by chunk position:
heightAndBiomeBlendingData – used for blending surface heights + biomes.
densityBlendingData – used for blending underground density/noise.
Core blending functions:
blendOffsetAndFactor(x,z) → returns (alpha, offset) used during height blending.
blendDensity(context, noiseValue) → returns blended density for caves/terrain based on neighboring old-gen data.
getBiomeResolver(biomeResolver) → wraps the normal biome resolver to optionally override with old-gen biome near borders.
Other responsibilities (important entry points called during gen)
generateBorderTicks(region, chunk)
Ensures neighboring cells that might be affected by blending (e.g., leaves, fluids) get marked for post-processing to avoid visual artifacts.
addAroundOldChunksCarvingMaskFilter(region, protoChunk)
Adds a carving mask for terrain carving (caves etc.) near old/new boundaries to avoid carving bleeding across blended boundaries.
Helper structures:
BlendingOutput (alpha+offset)
DistanceGetter (used for carving mask geometry)
BlendingData
Represents cached boundary samples from “old-gen” chunks.
Stores:
Heights at a grid along chunk borders + corners
Biome column samples (at quart resolution) along those same borders
Density columns (for noise/density blending in vertical columns)
Works against a fixed grid resolution:
CELL_WIDTH=4 (quart-based), CELL_HEIGHT=8 (vertical cell size)
Stores border data using indices derived from quart positions, enabling quick access on boundaries.
Key responsibilities:
Capturing and caching per-chunk boundary data once per chunk (calculateData)
Providing accessors for:
getHeight(cellX, cellY, cellZ)
getDensity(cellX, cellY, cellZ)
Iteration methods used by Blender:
iterateHeights() (for height blending)
iterateDensities() (for density blending)
iterateBiomes() (for biome blending)
Supporting serialization via Packed with Codec for network/storage (min/max section + optional height array).
🔗 How These Classes Relate to World Generation
Entry Point: Blender.of(WorldGenRegion) is called during generation when a region may straddle old/new generation boundaries.
Data Flow:
Determine region chunks that are “old-generation” around the target chunk.
Fetch or compute BlendingData for those old chunks.
Use Blender to:
Blend height/density/noise as generation runs.
Override biome selection near boundaries.
Apply carving masks and tick propagation at boundaries.
Runtime Use:
During terrain height/terrain noise generation, the generator calls blendOffsetAndFactor and blendDensity to softly mix new noise with the old pregen values.
When selecting biomes for a block, it uses getBiomeResolver to potentially return an old biome at the border instead of generating a new one.
🧠 Important Patterns & Concepts
Spatial Sampling Grid: Uses quart-block and "cell" granularity (4x4 horizontal, 8 vertical) rather than per-block, to keep blending stable and efficient.
Distance-weighted blending: blendOffsetAndFactor and blendDensity use inverse-quad weighting based on distance from sampled values, giving smooth falloff.
“Old generation” adjacency detection: Uses BlendingData.sideByGenerationAge() to find boundary faces where old and new gen meet.
Masking with noise jitter: Uses SHIFT_NOISE to jitter blending boundaries slightly, reducing grid-aligned artifacts.
Chunk-based caches: Stores blending data per chunk in Long2ObjectOpenHashMap, keyed by packed chunk coordinates. This avoids recomputing on every access.
🗝️ Main “Entry Point” Methods Systems Rely On
Blender.of(WorldGenRegion) → build blending context for a chunk region
Blender.blendOffsetAndFactor(...) → used during terrain height generation
Blender.blendDensity(...) → used during density/noise-based generation (caves/etc)
Blender.getBiomeResolver(...) → used to override biome selection near boundaries
Blender.generateBorderTicks(...) → ensures post-processing on boundary blocks
Blender.addAroundOldChunksCarvingMaskFilter(...) → prevents carving artifacts across boundaries
If you want, I can also map which generator stages (terrain, caves, biomes) call these methods by tracing usage from the levelgen pipeline in nearby packages.


## Block Predicates

This package defines a small data-driven predicate system used during world generation to decide whether a particular world position (a BlockPos) satisfies certain block/space conditions. It’s primarily used to filter/allow feature placements and generation steps based on the existing blocks/fluids around the target position.

All predicates are serializable via Minecraft’s Codec system, meaning they can be expressed in data (JSON / datapacks / configured features) and combined in a flexible way.

🧩 Core Abstractions
BlockPredicate
Interface extending BiPredicate<WorldGenLevel, BlockPos>.
Provides factory helpers for common predicate types (matching blocks, tags, fluids, offsets, etc.).
Includes built-in constants like:
ONLY_IN_AIR_PREDICATE
ONLY_IN_AIR_OR_WATER_PREDICATE
BlockPredicateType
Acts as the registry/type system for each predicate implementation.
Each predicate implementation registers its own type (e.g., MATCHING_BLOCKS, REPLACEABLE, ANY_OF, etc.).
Used to serialize/deserialize predicates in worldgen data.
🔑 Key Predicate Implementations (What They Test)
✅ Block-state / Tag / Fluid-based Tests
MatchingBlocksPredicate
Tests whether the block at the target position matches one of a provided set of blocks.
MatchingBlockTagPredicate
Tests whether the block at the position matches a block tag (e.g., minecraft:logs).
MatchingFluidsPredicate
Tests whether the fluid at the position is in a provided set (e.g., water/lava).
✅ Structural / Placement Validity Tests
ReplaceablePredicate
Passes when the target block “can be replaced” (usually leaves, plants, air, etc.).
WouldSurvivePredicate
Tests if a specific BlockState would survive at the given position (often used for plant/vegetation placement rules).
HasSturdyFacePredicate
Tests if a block face in a given direction is “sturdy” (supports attachments, torches, etc.).
InsideWorldBoundsPredicate
Ensures the target position is within build height limits.
UnobstructedPredicate
Checks whether a shape can fit (“unobstructed”) at the target position (used to ensure space is empty).
SolidPredicate (deprecated)
Tests BlockState.isSolid(), included likely for legacy reasons.
✅ Logical Combinators
AllOfPredicate
Passes only if all child predicates pass.
AnyOfPredicate
Passes if any child predicate passes.
NotPredicate
Inverts the result of another predicate.
TrueBlockPredicate
Always passes (useful as a default/no-op condition).
🔗 How These Are Used in Generation Rules
✅ Placement Filtering
The main use is in placement modifier filtering, via:

BlockPredicateFilter (net.minecraft.world.level.levelgen.placement.BlockPredicateFilter)
Wraps a BlockPredicate
Called during feature placement to decide whether a candidate BlockPos should actually be used (e.g., a flower only spawns if the ground block matches a condition).
This makes block predicates a central decision point in:

Placed feature filtering (e.g., “only place here if the block is replaceable AND inside world bounds”)
Data-driven feature configs, where predicate logic is composed and serialized into feature placement rules.
🧠 High-level Summary (Why This Matters)
Purpose: Provide reusable, composable “can we place here?” tests based on world block state.
Scope: Applies to blocks, fluids, world bounds, and general placement constraints.
Usage: Found in placement pipelines to guard feature generation; supports rich data-driven worldgen rules through serialization-friendly codecs.

## Carver

This package implements cave/canyon carving logic used during world generation to carve out underground spaces and surface canyons. It defines the core algorithms, configuration types, and runtime context used by the chunk generator to modify terrain blocks into air/lava/water based on procedural noise and biome rules.

🧱 Core Concepts & Key Classes
🔧 WorldCarver<C extends CarverConfiguration>
Base abstract class for all carvers (caves, nether caves, canyons).
Contains the shared carving logic:
carveEllipsoid(...) — iterates over a 3D ellipsoid and decides which blocks to replace.
carveBlock(...) — decides whether a block can be replaced, handles grass & topsoil correction, uses Aquifer to choose fluid (air/water/lava).
getCarveState(...) — chooses final block state (air, cave air, lava, debug blocks).
Maintains list of replaceable fluids (default water; overridden by Nether carver).
Exposes isStartChunk(...) and carve(...) as abstract methods implemented by each concrete carver.
🧩 Configured Carver Wrapper
🗃 ConfiguredWorldCarver<WC extends CarverConfiguration>
Combines a WorldCarver instance with a specific configuration object.
Holds codecs for serialization via worldgen JSON/registry (CONFIGURED_CARVER).
Used during world generation to:
Determine if carver should start in a chunk (isStartChunk)
Execute carving on a chunk (carve(...))
This is what worldgen actually stores and iterates over in dimension settings.
🧠 Runtime Context
🧩 CarvingContext
Extends WorldGenerationContext
Provides access to:
NoiseChunk (noise / surface shape info)
RandomState (surface rules, etc)
SurfaceRules.RuleSource (for top material decisions)
RegistryAccess (biome/blocks/etc)
Its topMaterial() is used to restore surface blocks (e.g., grass/dirt) when carving exposes them.
⚙️ Configuration Types
🧾 CarverConfiguration
Base config used by all carvers.

Important fields:

probability — chance a chunk will start carving
y (HeightProvider) — vertical placement distribution
yScale (FloatProvider) — vertical scaling factor for tunnels
lavaLevel (VerticalAnchor) — below this, tunnels fill with lava
debugSettings — optional debug-block replacement support
replaceable — what blocks carving can replace (e.g., stone, deepslate)
🕳 CaveCarverConfiguration (extends CarverConfiguration)
Adds:

horizontalRadiusMultiplier
verticalRadiusMultiplier
floorLevel (controls when carve is skipped below a level)
Used by:

CaveWorldCarver (overworld cave carving)
NetherWorldCarver (nether cave carving)
🏞 CanyonCarverConfiguration (extends CarverConfiguration)
Adds:

verticalRotation
shape (nested CanyonShapeConfiguration), which includes:
distanceFactor
thickness
widthSmoothness
horizontalRadiusFactor
vertical radius tuning factors
Used by:

CanyonWorldCarver
🧪 CarverDebugSettings
Provides debug support that can replace carved air/water/lava with special debug blocks when:

SharedConstants.DEBUG_CARVERS == true, or
config.debugSettings.debugMode == true
This is mainly for development/visualization of carving behavior.

🧨 Main Carver Implementations
🕳 CaveWorldCarver
Produces standard cave networks.
Key behaviors:
Randomly chooses a starting point in chunk based on probability.
Generates cave “rooms” and “tunnels”.
Uses random walk with sin/cos rotations to create winding tunnels.
Skips carving below floorLevel, and avoids carving below bottom bounds.
Uses carveEllipsoid to hollow out tunnels.
🔥 NetherWorldCarver
Extends CaveWorldCarver but tailors behavior for the Nether:
Uses both lava and water fluids for carving (liquids set contains both).
Thicker tunnels and much higher vertical scaling (getYScale() returns 5.0).
Overrides carveBlock() to place lava below a fixed depth (< minY + 31).
🏔 CanyonWorldCarver
Generates canyon-shaped canyons (wide, surface-cutting gullies).
Key behaviors:
Uses a “width factor per height” profile (initWidthFactors) to shape cross-section.
Adjusts vertical radius based on distance along tunnel (wider in middle, narrower ends).
Uses a custom skip function that considers a per-height width factor to form canyon walls.
Works in extended ranges to cut across multiple chunks.
🔁 How Carvers Are Invoked During World Generation
While not in this package, the typical invocation pipeline is:

Dimension/biome settings define a list of ConfiguredWorldCarver objects (registered via Registries.CONFIGURED_CARVER).
During chunk generation (e.g., in NoiseBasedChunkGenerator / NoiseChunk), the chunk generator:
Builds a CarvingContext
Iterates configured carvers for the dimension
Calls isStartChunk(random) to decide which carvers run in each chunk
Calls `carve(...)` for those carvers, passing:
- The chunk to modify
- A carving mask (to prevent double-carving)
- The Aquifer system (for fluid placement)
- The biome getter (for surface material fixes)

### ✅ Summary (High-Level)
- The package is all about carving caves/canyons by replacing blocks with air/water/lava using noise + random algorithms.
- `WorldCarver` is the core algorithm (ellipsoid carving, skip checks, debug blocks).
- `ConfiguredWorldCarver` ties carvers to the worldgen registry/config system (JSON-backed).
- `CarverConfiguration` + subclasses control behavior (frequency, size, shape, depth).
- `CarvingContext` ties carvers into the broader world gen (noise, surface rules, biomes).

**Core carvers used:**
- `CaveWorldCarver` (overworld caves)
- `NetherWorldCarver` (nether caves, lava-focused)
- `CanyonWorldCarver` (surface canyons, varying width by height)

> If you want, I can also point to where in the chunk generator these configured carvers are evaluated and applied (e.g., `NoiseBasedChunkGenerator.carve()`).

---

## Features (worldgen decorations)

This package is the core implementation of “world generation features” in Minecraft (trees, ores, lakes, dripstone, vines, desert wells, etc.). It defines the building blocks that the chunk generator uses during terrain decoration.

### 🧩 Key Concepts & Responsibilities

#### 1) Feature = “What to generate”
- `Feature<FC>` is the abstract base class for all generation features (ore veins, trees, dripstone, etc.).
- Each specific feature type extends `Feature` and implements `place(FeaturePlaceContext<FC>)` (the generation algorithm).
- Examples: `OreFeature`, `TreeFeature`, `LakeFeature`, `DripstoneClusterFeature`.

#### 2) Feature Configurations = “How to parameterize it”
- Features are generic over a config type (`FC extends FeatureConfiguration`).
- Configs live in `configurations/` (e.g., `OreConfiguration`, `TreeConfiguration`, `SpikeConfiguration`).
- Each config holds parameters that define the feature (block state lists, size, count, placement rules).

#### 3) Configured Feature = “A feature + its concrete parameters”
- `ConfiguredFeature<FC, Feature>` pairs a `Feature` instance with a specific config.
- Provides:
  - A codec for serialization/deserialization (via `BuiltInRegistries.FEATURE`)
  - A `place(...)` method that delegates to the underlying feature implementation.

#### 4) Placement Context = runtime inputs for generation
`FeaturePlaceContext<FC>` provides everything needed at generation time:
- `WorldGenLevel` (world access)
- `ChunkGenerator`
- `RandomSource`
- `BlockPos origin`
- Optional `topFeature` (for nesting/recursion)
- The feature config instance (`FC`)

#### 5) Feature Registry / Built‑in Feature Definitions
- `Feature` contains static fields for all built-in feature types (e.g., `ORE`, `TREE`, `LAKE`).
- These are registered via `Feature.register(name, featureInstance)` (uses `Registry.register(BuiltInRegistries.FEATURE, name, feature)`).
- This mapping is how Minecraft knows “ore” means `OreFeature`, etc.

### 🧠 Feature Registration & Serialization Patterns

✅ **Registry Pattern**
- Every feature type is registered in `BuiltInRegistries.FEATURE` using a string key.
- Enables data-driven references (biomes, datapacks) to point at features by identifier.

✅ **Codec-based Serialization**
- `Feature` uses Mojang `Codec` to (de)serialize configurations.
- `Feature.configuredCodec()` returns a `MapCodec<ConfiguredFeature<...>>` for the `config` field.
- `ConfiguredFeature.DIRECT_CODEC` uses `BuiltInRegistries.FEATURE.byNameCodec()` and dispatches to the feature’s codec.

This enables features to be fully data-driven via JSON/datapacks (e.g., biome builder, worldgen settings).

---

## Flat World Generation (superflat)

This package defines the data model and presets for “flat” world generation (aka superflat). It encapsulates the configuration of layers, biome, structures, lakes, and decoration behavior used when generating a flat world.

### 🔑 Key Classes & Responsibilities

- **FlatLayerInfo**
  - Represents one layer in a flat world.
  - Stores:
    - `height` (number of blocks thick)
    - `block` (which `Block` is used)
  - Provides:
    - A human-readable string form like `3*minecraft:dirt`
    - A codec for serialization/deserialization (`CODEC`)
    - A helper to enforce max height (`heightLimited`)

- **FlatLevelGeneratorSettings**
  - Core configuration object for a flat world.
  - Contains:
    - Ordered list of `FlatLayerInfo` → used to build the actual column of blocks (`layers`)
    - `biome` (the biome used for generation)
    - `structureOverrides` (which structure sets should be placed)
    - `decoration` / `addLakes` flags (toggle generation of features like ores, trees, lakes)
  - Derived `voidGen` boolean (true if all layers are air)
  - Key behaviors:
    - Builds a flattened `List<BlockState>` (`layers`) from the `layersInfo`
    - Validates total height against `DimensionType.Y_SIZE`
    - Adjusts biome generation settings to:
      - Add lakes (lava lakes by default)
      - Add/remove decorations depending on config
      - Replace “non-opaque” layers via a world-gen feature (`FILL_LAYER`) so things like water/lava layers generate correctly
  - Serialization:
    - Exposes a `CODEC` used by the data-driven system (for flat world JSON strings; used by world creation UI and data packs)

- **FlatLevelGeneratorPreset**
  - Simple wrapper that couples:
    - A display `Item` (for UI icon in world creation)
    - A `FlatLevelGeneratorSettings` instance
  - Supplies codec + registry registration support (via `RegistryFileCodec`)

- **FlatLevelGeneratorPresets**
  - Provides static registry keys for built-in presets (e.g., `CLASSIC_FLAT`, `THE_VOID`, `OVERWORLD`)
  - Implements the bootstrap registry logic that:
    - Creates default presets (layer stacks, biome, structures, lakes/decorations)
    - Registers them into the `FLAT_LEVEL_GENERATOR_PRESET` registry
  - Acts as the primary entry point for built-in flat generator types

### 🧱 Flat World Configuration & Layering

- Layers are defined as a list of `FlatLayerInfo`.
- Example preset stack (top → bottom):
  - `1x GRASS_BLOCK`
  - `2x DIRT`
  - `1x BEDROCK`
- `FlatLevelGeneratorSettings.updateLayers()` expands that into a full height list (`List<BlockState> layers`) by repeating each block N times.
- Total height is validated against `DimensionType.Y_SIZE`.
- If all expanded layers are `AIR`, the world is treated as a void world (`voidGen` flag).

### 🚪 Key Entry Points for Flat World Generation

- **Preset registration (bootstrap)**
  - `FlatLevelGeneratorPresets.bootstrap(...)` — called during data pack/registry bootstrap to register built-in presets.

- **Flat settings serialization**
  - `FlatLevelGeneratorSettings.CODEC` — used when reading flat world settings from JSON / user input / data packs.

- **Flat generation behavior adjustment**
  - `FlatLevelGeneratorSettings.adjustGenerationSettings(...)` — applied when biome generation settings are gathered, to:
    - Insert lakes/features
    - Replace non-opaque layers via `Feature.FILL_LAYER`

✅ **How They Relate (High-Level Flow)**
- Preset registry (`FlatLevelGeneratorPresets`) defines named presets with:
  - Biome
  - Layer stack
  - Structure/decoration toggles
- Those presets yield `FlatLevelGeneratorSettings`.
- During world creation, flat settings are deserialized via `CODEC`.
- Generation code uses the expanded layers list to construct each chunk column and adjusts feature generation via the `adjustGenerationSettings()` path.

🔑 Key Classes & Their Responsibilities
FlatLayerInfo
Represents one layer in a flat world.
Stores:
height (number of blocks thick)
block (which Block is used)
Provides:
A human-readable string form like 3*minecraft:dirt
A codec for serialization/deserialization (CODEC)
A helper to enforce max height (heightLimited)
FlatLevelGeneratorSettings
Core configuration object for a flat world.
Contains:
Ordered list of FlatLayerInfo → used to build the actual column of blocks (layers)
biome (the biome used for generation)
structureOverrides (which structure sets should be placed)
decoration / addLakes flags (toggle generation of features like ores, trees, lakes)
Derived voidGen boolean (true if all layers are air)
Key behaviors:
Builds a flattened List<BlockState> (layers) from the layersInfo
Validates total height against DimensionType.Y_SIZE
Adjusts biome generation settings to:
Add lakes (lava lakes by default)
Add/remove decorations depending on config
Replace “non-opaque” layers via a world-gen feature (FILL_LAYER) so things like water/lava layers generate correctly
Serialization:
Exposes a CODEC used by the data-driven system (for flat world JSON strings; used by world creation UI and data packs)
## Height Providers

This package defines how Minecraft picks a Y‑coordinate for placing features during world generation (e.g., ores, trees, structures).
A HeightProvider is a pluggable “height selector” used by worldgen code to sample a Y value given a RNG and generation context.
It supports data‑driven configuration via Mojang’s Codec system and is registered in the BuiltInRegistries.HEIGHT_PROVIDER_TYPE registry.
📌 Core Abstractions
HeightProvider (abstract base)
Defines a single key method: int sample(RandomSource random, WorldGenerationContext context)
Used anywhere worldgen needs a Y-coordinate (placement height).
Provides a Codec<HeightProvider> that can decode either:
a raw VerticalAnchor (constant height), or
a registered HeightProvider subtype via HeightProviderType.
HeightProviderType<P extends HeightProvider>
Registry wrapper for each provider kind.
Provides the MapCodec used for serialization/deserialization.
Built‑in registered types in this package:
constant, uniform, biased_to_bottom, very_biased_to_bottom, trapezoid, weighted_list
🧱 Main Height Provider Types (and What They Do)
ConstantHeight
Always returns a constant Y.
Uses VerticalAnchor, so it can be absolute or relative to world height (e.g., top, bottom, above_bottom).
Used when a feature must be placed at a fixed vertical level.
UniformHeight
Picks a height uniformly between two anchors: min_inclusive and max_inclusive.
If the range is invalid (min > max), it logs a warning and returns min.
Used when any height within a range is equally likely (common for ores, features, etc.).
BiasedToBottomHeight
Picks a height between min and max but biased toward the bottom.
Works by:
Sampling a random limit within the range.
Sampling a second value between min and min + limit + offset.
The inner parameter controls how strong the bottom bias is (higher means stronger bottom bias).
VeryBiasedToBottomHeight
Like BiasedToBottomHeight, but applies an extra level of “bias squaring”:
Pick an upper bound randomly between (min + inner) and max.
Pick a biased upper bound again between min and (upper bound - 1).
Pick final Y between min and (biased upper - 1 + inner).
Produces a steeper falloff toward bottom — used when only bottommost range should be common.
TrapezoidHeight
Produces a triangular/trapezoidal distribution across a range:
When plateau == 0, it’s a triangle distribution (more values near center).
When plateau > 0, it’s trapezoid: a flat top region (uniform) surrounded by sloped edges.
Useful to favor mid‑range heights but allow extremes less frequently.
WeightedListHeight
Holds a WeightedList<HeightProvider> (each entry is itself a HeightProvider).
Chooses one provider by weight, then delegates sample(...) to it.
Used when you want a mixed strategy (e.g., sometimes uniform, sometimes biased) and control how often each strategy is chosen.
🔄 How These Are Used in Generation
World generation code asks a HeightProvider for a height via sample(random, context).
WorldGenerationContext provides world-specific data needed to resolve VerticalAnchor (e.g., world height, sea level).
The chosen Y is then used for actual placement of blocks/features (like ores, vegetation, structures, etc.).
Because they’re Codec-driven, height providers are commonly configured in JSON/datapacks (e.g., worldgen feature configs).
🧩 Relationship Summary
HeightProvider = interface for “choose a Y”
HeightProviderType = registry + codec dispatcher for provider kinds
Concrete providers (ConstantHeight, UniformHeight, etc.) = different probability shapes for sampling heights
WeightedListHeight = meta-provider that mixes other providers by weight


## Material

Core role: Provides the material-selection layer used during noise-based chunk generation.
Where it fits: NoiseChunk builds a MaterialRuleList to decide, for each generated position, whether a special material/BlockState should override the default terrain fill (stone, deepslate, etc.).
Mechanism: It applies a prioritized list of “material rules” (block state fillers) and returns the first non-null result; otherwise generation falls back to the chunk generator’s default block.
🔑 Key Class & Responsibilities
MaterialRuleList
Type: record MaterialRuleList(NoiseChunk.BlockStateFiller[] materialRuleList)
Responsibility: Implements NoiseChunk.BlockStateFiller by sequentially running each rule in materialRuleList and returning the first non-null BlockState.
Usage: Built in NoiseChunk (line ~160 of NoiseChunk.java) and used in NoiseBasedChunkGenerator to decide the material for a given noise point.
🧩 How It Relates to World Generation (Material Selection Flow)
Noise values → density → base solid/air decision
NoiseChunk works with density functions to compute whether a location is “solid” or “air”.
Material rules are invoked when density indicates “solid” (density ≤ 0)
NoiseChunk.getInterpolatedState() calls the MaterialRuleList.
First non-null material wins
The first rule producing a BlockState is used.
If all return null, generation uses the generator’s defaultBlock() (usually stone/deepslate/etc, via NoiseBasedChunkGenerator).
🪨 Main Material Types / Rule Sources (in this version)
1) Fluids (Water/Lava) – Aquifer
Implemented via Aquifer.NoiseBasedAquifer (or a disabled stub when aquifers are off).
Responsibility:
Determine whether underground space becomes water or lava based on:
density (solid vs empty)
nearby “aquifer centers” (grid-based sampling)
biome/global fluid rules (via FluidPicker)
noise-based “pressure/barrier” interactions
Optionally mark locations that need fluid updates (shouldScheduleFluidUpdate()).
Material types produced:
Blocks.WATER / Blocks.LAVA (or Blocks.AIR in debug/disable mode)
Used for underground seas, lava pockets, and fluid-filled caves.
2) Ore Veins – OreVeinifier
Responsibility:
Decide whether a location becomes ore (copper/iron) or filler rock based on noise functions and depth constraints.
Uses multiple noise inputs:
veinToggle (decides vein region / type)
veinRidged (vein density/strength)
veinGap (gap/spacing)
Applies randomness (positional RNG) for richness/rarity and “raw ore block” chance.
Material types produced:
Blocks.DEEPSLATE_IRON_ORE, Blocks.COPPER_ORE
Blocks.RAW_IRON_BLOCK, Blocks.RAW_COPPER_BLOCK (rare)
Fillers (TUFF, GRANITE) for the surrounding matrix when veins are “present” but not dense
### 🧠 How These Rules Are Used in Terrain / Feature Generation

- In `NoiseBasedChunkGenerator.iterateNoiseColumn(...)`, the generator:
  1. Builds a `NoiseChunk` for a column.
  2. Iterates cell-by-cell, computing noise interpolation.
  3. Calls `noiseChunk.getInterpolatedState()`:
     - If non-null → use returned material (fluid/ore/filler).
     - If null → use the generator’s `defaultBlock()` (stone/deepslate etc.).
- Surface system and carvers later reshape terrain using the same `NoiseChunk`/`Aquifer` data.

### ✅ Summary (What This Package Provides)
- A rule-chain mechanism (`MaterialRuleList`) that decides per-block material based on noise, fluids, and veins.
- Integration points for:
  - Aquifer fluid generation (underground lakes/lava)
  - Ore vein placement
- Enables extensible material logic — new rules can be added as new `NoiseChunk.BlockStateFiller` implementations without modifying generator core.

> If you'd like, I can map precisely where `NoiseChunk.getInterpolatedState()` is invoked during chunk generation and which defaults (stone/deepslate/etc.) are used when all rules return null.

## Placement

## Placement

The `net.minecraft.world.level.levelgen.placement` package defines the placement pipeline that takes a configured feature (e.g., tree, ore vein, lake) and decides **where** in a chunk/biome it should actually try to generate.

### What it combines
- **ConfiguredFeature** – what to place
- **PlacementModifiers** – how/where to place it
- **PlacementContext** – provides world + generator + optional “top feature” metadata

The core idea: a `PlacedFeature` starts from an origin point and threads it through a chain of placement modifiers; each modifier can expand, filter, or reposition that position before the feature attempts to place.

### 🔑 Core Classes

- **PlacedFeature**
  - Represents a configured feature plus a list of placement modifiers.
  - Runs through the modifier chain, producing a stream of candidate `BlockPos` values.
  - Attempts to place the feature at each resulting position via `ConfiguredFeature.place(...)`.

- **PlacementModifier**
  - Base abstract type for anything that transforms an input position into 0+ output positions.
  - Implements:
    - `Stream<BlockPos> getPositions(PlacementContext, RandomSource, BlockPos origin)`
    - `PlacementModifierType<?> type()` (used for registry/codecs)

- **PlacementContext**
  - Provides contextual data (world, chunk generator, heightmaps, top feature reference, etc.).
  - Used by modifiers for world queries (height, biome, block state, etc.).

- **PlacementModifierType**
  - Registry wrapper for placement modifier codec/deserialization.
  - Contains built-in modifier types (e.g., `PlacementModifierType.COUNT`, `PlacementModifierType.BIOME_FILTER`).

### 🧩 Modifier Categories & Key Types

#### 1) Filter Modifiers (may prevent placement entirely)
Filters take an origin and either return it (allowed) or return empty (blocked).

- **RarityFilter**
  - Randomly rejects placement based on chance (`1 in N` average).

- **BiomeFilter**
  - Ensures the current biome actually contains the feature (checks configured biome features vs the “top feature”).

- **BlockPredicateFilter**
  - Uses a `BlockPredicate` to allow placement only if the block at origin (or neighbors) matches conditions.

- **SurfaceRelativeThresholdFilter**
  - Allows placement only if the Y coordinate is within a range relative to the local surface heightmap.

- **SurfaceWaterDepthFilter**
  - Allows placement only if water depth above ocean floor is ≤ max (prevents deep-water placements).

#### 2) Count/Repeat Modifiers (generate multiple placement attempts per origin)

- **CountPlacement**
  - Repeat placement N times (fixed or random via `IntProvider`).

- **NoiseBasedCountPlacement**
  - Uses biome noise to vary count by position (e.g., “flower density” noise pattern).

- **NoiseThresholdCountPlacement**
  - Picks one of two counts depending on noise being above/below a threshold.

- **CountOnEveryLayerPlacement** *(deprecated)*
  - Attempts placement on multiple vertical “layers” by scanning from top down each time.

#### 3) Position/Offset Modifiers (move or scatter placement)

- **InSquarePlacement**
  - Picks a random X/Z within the chunk (standard “spread within chunk” behavior).

- **RandomOffsetPlacement**
  - Random X/Z/Y offset added to the origin (scatter around it).

- **FixedPlacement**
  - Uses a fixed list of explicit block positions (often used for small hard-coded structures).

- **HeightRangePlacement**
  - Chooses a Y coordinate randomly from a height distribution (uniform, trapezoid, etc.).

- **HeightmapPlacement**
  - Snaps placement to a heightmap surface position (e.g., top of terrain).

- **EnvironmentScanPlacement**
  - Scans in a direction (typically up/down) searching for a target block condition, optionally bounded by an allowed condition (e.g., first solid block below air).

### 🔄 How Placement Modifier Chains Work

1. `PlacedFeature.place(...)` starts with `Stream.of(origin)`.
2. Each modifier in `PlacedFeature.placement` is applied in sequence:
   - `stream = stream.flatMap(pos -> modifier.getPositions(context, random, pos))`
3. Result is a stream of final candidate positions.
4. For each resulting position, the underlying feature is attempted (`feature.place(...)`).

So: placement modifiers are composable — filters can prune, repeaters can expand, and offsets/height modifiers can move positions. That’s how generation logic is assembled declaratively (often from JSON/registry data) rather than hard-coded.

### ✅ Summary (why this matters)

- This package is the placement engine for Minecraft’s worldgen features.
- Modifiers are the “domain-specific language” used in data-driven feature placement.

Placement is shaped by:
- **where** the feature is allowed (`BiomeFilter`, `BlockPredicateFilter`, etc.)
- **how often** it should try (`Count*`, `RarityFilter`, noise-based counts)
- **where** in the chunk/height it targets (`InSquare`, `Heightmap`, `HeightRange`, etc.)

The combination of modifiers defines the final distribution of features in biomes/world generation.

## Presets

The net.minecraft.world.level.levelgen.presets package defines named “world presets” used to bootstrap and register common world-generation configurations (overworld/nether/end). These presets tie together:

Dimension types (DimensionType / LevelStem)
Chunk generators (NoiseBasedChunkGenerator, FlatLevelSource, DebugLevelSource)
Biome sources (multi-noise presets, fixed biome, The End biome set)
Noise settings (NoiseGeneratorSettings variants like OVERWORLD, LARGE_BIOMES, AMPLIFIED)
Presets are registered into Minecraft’s registry system and are the canonical way the engine selects “normal/flat/amplified/debug” world generation behavior at world creation time.

🔑 Key Classes & Responsibilities
WorldPreset
Core model for a world preset.
Contains a map of ResourceKey<LevelStem> → LevelStem, representing the configured dimensions (Overworld/Nether/End).
Provides methods to:
Validate that an overworld exists (requireOverworld)
Create concrete WorldDimensions instances used by the world generator (createWorldDimensions())
Access the overworld stem (overworld())
It also defines serialization codecs so presets can be stored/loaded via data-driven registries.

WorldPresets
Registry holder + bootstrap for built-in presets.
Defines the standard preset keys:
NORMAL
FLAT
LARGE_BIOMES
AMPLIFIED
SINGLE_BIOME_SURFACE
DEBUG
Contains bootstrap(...) that builds each preset by:
Fetching registry-provided noise settings, biome sources, structure sets, etc.
Creating LevelStem objects for each dimension (overworld/nether/end)
Linking the overworld’s chunk generator variant to a preset key
Key helper methods:

createNormalWorldDimensions(...) / createFlatWorldDimensions(...): build a WorldDimensions object from registry presets
getNormalOverworld(...): fetches the normal overworld stem
fromSettings(WorldDimensions): attempts to map an existing world’s generator back to a preset key (by inspecting the overworld’s ChunkGenerator type)
🧩 Main Preset Types & How They Are Built
1) Normal / Large Biomes / Amplified
Built using NoiseBasedChunkGenerator (noise-based terrain)
Differ only by the NoiseGeneratorSettings used:
OVERWORLD → NORMAL
LARGE_BIOMES → LARGE_BIOMES
AMPLIFIED → AMPLIFIED
Use the same biome source preset (MultiNoiseBiomeSource from OVERWORLD preset)
2) Flat
Built using FlatLevelSource
Uses FlatLevelGeneratorSettings.getDefault(...), which defines flat layers + structures.
3) Single Biome Surface
Uses FixedBiomeSource with Plains biome (always one biome) + normal overworld noise settings.
4) Debug
Uses DebugLevelSource (renders all block states for debugging)
🔁 How Presets Bootstrap World Generation
Registry data is created via BootstrapContext<WorldPreset> in WorldPresets.bootstrap().
Each named preset is registered into Registries.WORLD_PRESET with its ResourceKey.
When the game creates a world, it selects a preset (e.g., minecraft:normal) and calls:
WorldPreset.createWorldDimensions() → produces WorldDimensions containing the configured LevelStems.
WorldDimensions drives dimension creation, feeding each LevelStem into the world loading pipeline (generator, dimension type, etc.).
🔗 How It Connects to Biomes & Noise Settings
Biome presets are retrieved via MultiNoiseBiomeSourceParameterLists and used to build MultiNoiseBiomeSource, which drives biome selection in noise worlds.
Noise settings (NoiseGeneratorSettings) define how terrain is shaped and are selected per preset (OVERWORLD / LARGE_BIOMES / AMPLIFIED / NETHER / END).
The preset system ensures a single entry point for “world type” selection while keeping the per-dimension stems consistent (overworld + nether + end).
If you want, I can trace how a specific preset (e.g., AMPLIFIED) is referenced from the world creation UI and where the registry wires it into the save file metadata.

## Structure

This package defines the framework for Minecraft’s world structure generation (strongholds, villages, monuments, etc.).
It includes:
structure definitions (Structure, StructureType, StructureSettings)
placement logic (when/where structures can generate)
structure composition (pieces, templates, pools, serialization)
registry keys for built-in structures and structure sets
🧱 Core Classes & Responsibilities
Structure
Abstract base for all structure types (villages, strongholds, ruins, etc.).
Contains generation entrypoint (generate(...)) and validation:
Uses findValidGenerationPoint() to pick a starting location.
Wraps generated pieces into a StructureStart.
Holds configuration via StructureSettings (biomes, spawn overrides, generation step, terrain adaptation).
Handles bounding box adjustments and terrain adaptation behavior.
StructureStart
Represents a concrete instance of a structure once placed in the world (per chunk).
Stores:
The Structure type
A PiecesContainer of StructurePieces
Chunk position + reference count (for chunk referencing mechanics)
Responsible for placing pieces into the world during chunk generation.
Handles serialization/deserialization of generated structures.
StructurePiece (and subclasses)
Base class representing a single building block of a structure (a room, corridor, room segment).
Handles:
Bounding boxes
Rotation/mirroring
Block placement (postProcess)
Serialization of piece state
Important subclass: TemplateStructurePiece
Uses Minecraft’s template system (StructureTemplate) and Jigsaw blocks
Handles:
template placement
data markers (STRUCTURE_BLOCK / JIGSAW processing)
applying template processors (gravity, terrain matching, etc.)
📦 Structure Composition: Pools & Jigsaw System
StructureTemplatePool
Central to “jigsaw” structures (villages, bastions, etc.).
Defines a weighted pool of pool elements (StructurePoolElement), which is used to choose the next piece during generation.
Supports fallback pools and tracks the max piece height.
Includes Projection modes (terrain matching vs rigid) which apply template processors (e.g., gravity adjustments).
StructurePoolElement / SinglePoolElement / ListPoolElement / EmptyPoolElement
Represent selectable “pieces” in the pool, including templates and special features.
Used during recursive assembly of complex structures.
📍 Placement Configuration (Chunk-level decision making)
StructurePlacement (and subclasses)
Defines whether a given chunk is allowed to start a structure and the distribution algorithm.

Key behavior:

Decides if a chunk is a structure chunk (isStructureChunk)
Implements frequency control (chance reduction, “salt”, exclusion zones)
Determines locate position offset (for /locate command support)
Notable Placement Types:
RandomSpreadStructurePlacement
Used by most structures (villages, strongholds, temples, etc.)
Defines spacing, separation, and spread_type
ConcentricRingsStructurePlacement
Used for structures like end cities / ancient cities
Generates positions in rings around origin
Placement also includes optional exclusion zones (avoid placing near other structure sets)
🗂️ Structure Sets & Registry Wiring
StructureSet
Defines a set of structures + the placement rules for the set.
Used by the world generator to decide which structure to attempt in a given valid chunk.
Structure sets can contain multiple weighted structures (e.g., village types across biomes).
BuiltinStructures / BuiltinStructureSets
Provide the registry keys for every built-in structure and set (e.g., STRONGHOLD, VILLAGE_PLAINS, ANCIENT_CITIES).
These keys are referenced by datapacks and worldgen JSON to configure generation behavior.
🔧 How Structure Generation Works (High-Level Flow)
During world generation, the chunk generator asks: “Is this a structure chunk?”
The chunk is tested against a StructurePlacement (e.g., random spread rules, ring rules).
If eligible, the generator attempts a structure start:
Calls Structure.generate(...)
Structure chooses a valid location (biome check, height reference, etc.)
If successful, the structure returns a StructureStart containing pieces
During chunk population:
StructureStart.placeInChunk(...) runs each StructurePiece.postProcess(...)
Pieces place blocks and handle template processing / jigsaw chaining
⚙️ Where Configuration Lives (How You Change Structure Spawning)
Registry JSONs under data/minecraft/worldgen/structures/ and worldgen/structure_sets/
Define what structures exist and how they’re placed
The core classes above are what interpret those configurations during generation.
If you want, I can also map specific structure types (stronghold, village, etc.) to their generation mechanics by inspecting their individual structures.*Structure and pieces.* classes.

## Synth (noise synthesis)

This package contains the core noise synthesis primitives Minecraft uses to generate terrain and related world features.
It provides Perlin-like and Simplex noise implementations, octave blending, scaling, and mixing behavior used by higher-level worldgen systems.
The classes here are low-level noise samplers (often wrapped by `DensityFunction`s elsewhere) used to produce the continuous pseudo-random fields that drive heightmaps, biomes, caves, and more.

### 🧠 Key Classes & Responsibilities

- **SimplexNoise**
  - Implements classic 2D/3D simplex noise.
  - Provides raw noise values via `getValue(x,y)` and `getValue(x,y,z)`.
  - Supplies the gradient table and permutation array used by `ImprovedNoise` and `PerlinSimplexNoise`.

- **ImprovedNoise**
  - Perlin-style gradient noise with:
    - Permutation table + offset (`xo/yo/zo`)
    - Optional derivative output (`noiseWithDerivative()`)
  - Used as the building block for octave-based Perlin noise in `PerlinNoise`.

- **PerlinNoise**
  - Combines many `ImprovedNoise` octaves to build multi-octave Perlin noise.
  - Configurable:
    - `firstOctave`
    - `amplitudes` (per-octave weights)
  - Provides:
    - `getValue()` producing blended multi-octave result
    - `maxBrokenValue()` for bounding return ranges
  - Used as a core input to higher-level noise functions (e.g., `BlendedNoise`, `NormalNoise`, and worldgen density functions).

- **PerlinSimplexNoise**
  - Similar to `PerlinNoise`, but built on `SimplexNoise` octaves.
  - Blends several simplex-noise octaves using a configurable octave set:
    - Each octave sampled at increasing frequencies
    - Optional "noise start" offsets via `useNoiseStart`
  - Used in places where simplex noise is preferred (often for 2D/heightmap components).

- **NormalNoise**
  - Builds on two `PerlinNoise` instances (`first` and `second`) to create a noise field with a controlled statistical distribution.
  - Applies a scaling factor (`valueFactor`) to normalize deviation and caps `maxValue`.
  - Designed for general-purpose terrain noise (biome noise, density functions, etc.).
  - Includes `NoiseParameters` record (with codec support) for configurable amplitude/octave patterns.

- **BlendedNoise**
  - Uses three `PerlinNoise` fields:
    - `mainNoise` (drives the blend factor)
    - `minLimitNoise` (lower bound noise)
    - `maxLimitNoise` (upper bound noise)
  - Computes a blend between min/max noise based on the “main” noise value.
  - Provides a noise field used in terrain density generation, particularly for smooth transitions between extremes (e.g., ground vs. air).

- **NoiseUtils**
  - Small utility helpers used by noise classes.
  - Includes functions like `biasTowardsExtreme()` (non-linear distortion) and parity/debug string building for testing.

### 🔗 How These Classes Relate in Noise Generation

- **Raw noise primitives:** `SimplexNoise` and `ImprovedNoise` generate single-octave noise samples.
- **Octave stacking + scaling:** `PerlinNoise` and `PerlinSimplexNoise` compose multiple octave layers into a coherent multi-frequency signal.
- **Statistical normalizing:** `NormalNoise` wraps two `PerlinNoise` instances and scales the output to expected deviation bounds.
- **Blended terrain volume fields:** `BlendedNoise` uses Perlin stacks to implement terrain height/density blending (critical for smooth transitions in 3D density functions).
- **Consumers:** These classes are typically consumed by higher-level `DensityFunction` implementations (e.g., `DensityFunction.Noise`, `DensityFunction.NoiseProvider`) that drive the actual chunk generation logic (biome shaping, caves, noise-based features).

### 🧩 Where “Synth” Noise Is Used in Worldgen

- 3D density fields (terrain shape / cave carving)
- Biome noise (plateau vs. deep biome transitions)
- Feature placement randomness (using noise to scatter features)
- Any system requiring stable pseudo-random spatial variation in world generation

## Root

This section summarizes the **top-level `levelgen/` classes** that are not in a subpackage. These form the core plumbing used by the noise-based chunk generator, world settings, RNG, and runtime surface/structure adjustments.

### Core Worldgen Engines

- **NoiseBasedChunkGenerator** – The main noise-based chunk generator. Drives `NoiseChunk` creation, calls surface rules, carving, feature placement, and blending.
- **NoiseChunk** – Computes the 3D density field, applies material rules (ores, aquifers), and provides block state lookup for a chunk column.
- **FlatLevelSource** – Chunk generator used for flat worlds (superflat), driven by `FlatLevelGeneratorSettings`.
- **DebugLevelSource** – Special generator that renders every block state in a grid for debugging.

### Noise & Density System

- **DensityFunction / DensityFunctions** – The full DSL used to describe density/noise fields. Supports composable functions, codecs, and evaluation over a `FunctionContext`.
- **Density** – Defines key constants (surface/dense/thin) used throughout density calculations.
- **NoiseSettings / NoiseGeneratorSettings / NoiseRouter / NoiseRouterData / Noises** – Configuration objects and bootstrap data for the noise system (dimension height, noise scales, router inputs, ore/vein settings, blending values).

### Noise RNG & Utilities

- **PositionalRandomFactory** – Interface for producing position-dependent RNGs (seeded by coords or names).
- **RandomState** – Bundles noise parameters, router, and RNG factories (surface, aquifer, ore) for a full worldgen run.
- **WorldgenRandom** – Wrapper used for feature/decoration RNG with deterministic seeding schemes (`setFeatureSeed`, etc.).
- **LegacyRandomSource / SingleThreadedRandomSource / ThreadSafeLegacyRandomSource / XoroshiroRandomSource** – Concrete RNG implementations (legacy LCG and xoroshiro), all of which implement `BitRandomSource`.
- **RandomSupport** – Seed mixing utilities and 128-bit seed handling.
- **MarsagliaPolarGaussian** – Provides Gaussian sampling used by RNGs.
- **BitRandomSource** – Low-level interface for bit-level random sampling used by RNG implementations.

### Fluid & Material Generation

- **Aquifer** – Generates underground fluids (water/lava) based on noise, depth, and biome rules. Used by `NoiseChunk` during block state selection.
- **OreVeinifier** – Converts noise patterns into ore vein blocks (iron/copper/rare raw ore) and filler blocks.

### Surface & Structure Adjustments

- **SurfaceRules / SurfaceSystem** – Defines how the surface blocks are chosen (grass/dirt/stone/etc.) and applies rules for caves, cliffs, and features.
- **Beardifier** – Applies a smoothing kernel around structures (jigsaw-based) to blend the terrain and avoid sharp cuts.
- **Column** – Utility for defining and scanning vertical ranges in a column (used by surface/feature scanning).

### Structure & World Config

- **GenerationStep** – Enum of worldgen stages (raw generation, lakes, structures, ores, vegetation, etc.).
- **WorldDimensions / WorldGenerationContext / WorldGenSettings / WorldOptions** – World definition structures (dimensions, generator settings, seed/options).
- **VerticalAnchor** – Represents a vertical coordinate (absolute or relative to world top/bottom) used widely in height providers, carvers, and placement.

### Misc & Support

- **Heightmap** – Tracks per-chunk heightmaps for different surface rules (motion-blocking, world-surface, ocean floor, etc.).
- **PatrolSpawner / PhantomSpawner** – Spawn logic for patrols and phantoms (not worldgen but part of the broader levelgen package).

---


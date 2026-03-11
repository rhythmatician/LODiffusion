package com.rhythmatician.lodiffusion.voxy;

import java.lang.reflect.Method;
import java.util.logging.Logger;

/**
 * Runtime compatibility layer for Voxy.
 *
 * <p>Voxy has no public modding API, so we detect its presence via class loading
 * and access its internals through reflection.  All Voxy interactions go through
 * this class so that breakage from Voxy updates is contained.
 *
 * <p><b>Key Voxy concepts:</b>
 * <ul>
 *   <li>{@code WorldEngine} — the root object holding block/biome mappings and
 *       the section storage backend (RocksDB).</li>
 *   <li>{@code VoxelizedSection} — a 16³ ingest chunk with a built-in mip pyramid
 *       (16³ + 8³ + 4³ + 2³ + 1 = 4681 packed voxels).</li>
 *   <li>{@code WorldUpdater.insertUpdate(engine, section)} — the blocking call
 *       that pushes a section into the world, propagating through LOD layers.</li>
 *   <li>{@code Mapper} — translates Minecraft {@code BlockState}/{@code Biome}
 *       into Voxy's compact 64-bit voxel encoding.</li>
 * </ul>
 */
public final class VoxyCompat {

    private static final Logger LOGGER = Logger.getLogger(VoxyCompat.class.getName());

    /** Cached availability flag — computed once at first access. */
    private static volatile Boolean available;

    /**
     * Whether the full engine bindings (WorldEngine, WorldUpdater, Mapper, etc.)
     * have been resolved.  Separate from {@link #available} because those classes
     * have transitive Minecraft class references (e.g. class_2841) that fail to
     * load in the yarn-mapped test environment.  Tests only need {@code available}
     * to be {@code true} (VoxelizedSection is MC-free).
     */
    private static volatile boolean engineBindingsReady;

    /** Whether the MC-dependent WorldIdentifier bindings have been resolved. */
    private static volatile boolean worldBindingsReady;

    // Reflected classes (resolved lazily)
    private static Class<?> worldEngineClass;
    private static Class<?> worldUpdaterClass;
    private static Class<?> voxelizedSectionClass;
    private static Class<?> mapperClass;

    // Reflected methods
    private static Method insertUpdateMethod;      // WorldUpdater.insertUpdate(WorldEngine, VoxelizedSection)
    private static Method createEmptyMethod;        // VoxelizedSection.createEmpty()
    private static Method getMapperMethod;          // WorldEngine.getMapper()
    private static Method mipSectionMethod;         // WorldConversionFactory.mipSection(VoxelizedSection, Mapper)
    private static Method ofEngineMethod;           // WorldIdentifier.ofEngine(World)
    private static Method ofEngineNullableMethod;   // WorldIdentifier.ofEngineNullable(World)
    private static Method acquireIfExistsMethod;    // WorldEngine.acquireIfExists(int, int, int, int)
    private static Method acquireMethod;            // WorldEngine.acquire(int, int, int, int)
    private static Method worldSectionReleaseMethod; // WorldSection.release()
    private static Method markDirtyMethod;          // WorldEngine.markDirty(WorldSection)

    // Reflected fields for WorldSection direct access
    private static java.lang.reflect.Field worldSectionDataField;   // WorldSection.data (long[])
    private static java.lang.reflect.Field worldSectionNonEmptyChildrenField; // WorldSection.nonEmptyChildren (volatile byte)
    private static Class<?> worldSectionClass;

    private VoxyCompat() {}

    // ------------------------------------------------------------------ //
    //  Detection
    // ------------------------------------------------------------------ //

    /**
     * True if the core Voxy voxelization API ({@code VoxelizedSection}) is on
     * the classpath.
     *
     * <p>{@code VoxelizedSection} has <em>no</em> transitive Minecraft class
     * references, so this check succeeds even in the yarn-mapped JUnit environment
     * where Voxy's engine classes (which reference intermediary MC names like
     * {@code class_2841}) would fail to load.
     *
     * <p>The engine bindings (WorldEngine, WorldUpdater, Mapper, etc.) are
     * deferred to {@link #ensureEngineBindings()}, called lazily from the write
     * path ({@link #mipSection}, {@link #insertUpdate}, {@link #sectionExists}).
     * The {@code WorldIdentifier} bindings that require a live
     * {@code net.minecraft.world.World} are deferred further to
     * {@link #ensureWorldBindings()}.
     */
    public static boolean isAvailable() {
        Boolean cached = available;
        if (cached != null) return cached;

        synchronized (VoxyCompat.class) {
            if (available != null) return available;
            try {
                // VoxelizedSection has NO transitive MC class references:
                // fields are long[], int primitives only.  Safe to load in tests.
                voxelizedSectionClass = Class.forName(
                        "me.cortex.voxy.common.voxelization.VoxelizedSection");
                createEmptyMethod = voxelizedSectionClass.getMethod("createEmpty");

                // WorldEngine / Mapper / etc. have MC refs — deferred to ensureEngineBindings().

                available = true;
                LOGGER.info("Voxy detected — VoxelizedSection bindings resolved");
            } catch (ClassNotFoundException | NoSuchMethodException e) {
                available = false;
                LOGGER.info("Voxy not found: " + e.getMessage());
            } catch (LinkageError e) {
                available = false;
                LOGGER.info("Voxy class loading failed: " + e.getMessage());
            }
            return available;
        }
    }

    /**
     * Lazily bind the engine classes (WorldEngine, WorldUpdater, Mapper,
     * WorldConversionFactory, WorldSection) that have transitive Minecraft
     * class references.
     *
     * <p>Called from the write path ({@link #getMapper}, {@link #mipSection},
     * {@link #insertUpdate}, {@link #sectionExists}).  Unit tests never reach
     * the write path, so they only need {@link #isAvailable()} to return true.
     *
     * @throws IllegalStateException if the engine classes cannot be loaded
     */
    private static void ensureEngineBindings() {
        ensureAvailable();
        if (engineBindingsReady) return;
        synchronized (VoxyCompat.class) {
            if (engineBindingsReady) return;
            try {
                worldEngineClass  = Class.forName("me.cortex.voxy.common.world.WorldEngine");
                worldUpdaterClass = Class.forName("me.cortex.voxy.common.world.WorldUpdater");
                mapperClass       = Class.forName("me.cortex.voxy.common.world.other.Mapper");

                insertUpdateMethod = worldUpdaterClass.getMethod("insertUpdate",
                        worldEngineClass, voxelizedSectionClass);
                getMapperMethod    = worldEngineClass.getMethod("getMapper");

                Class<?> convFactoryClass = Class.forName(
                        "me.cortex.voxy.common.voxelization.WorldConversionFactory");
                mipSectionMethod = convFactoryClass.getMethod("mipSection",
                        voxelizedSectionClass, mapperClass);

                acquireIfExistsMethod = worldEngineClass.getMethod("acquireIfExists",
                        int.class, int.class, int.class, int.class);
                acquireMethod = worldEngineClass.getMethod("acquire",
                        int.class, int.class, int.class, int.class);
                worldSectionClass = Class.forName(
                        "me.cortex.voxy.common.world.WorldSection");
                worldSectionReleaseMethod = worldSectionClass.getMethod("release");
                markDirtyMethod = worldEngineClass.getMethod("markDirty",
                        worldSectionClass);
                worldSectionDataField = worldSectionClass.getDeclaredField("data");
                worldSectionDataField.setAccessible(true);
                worldSectionNonEmptyChildrenField = worldSectionClass.getDeclaredField("nonEmptyChildren");
                worldSectionNonEmptyChildrenField.setAccessible(true);

                engineBindingsReady = true;
                LOGGER.info("Voxy engine bindings resolved");
            } catch (ClassNotFoundException | NoSuchMethodException | NoSuchFieldException e) {
                throw new IllegalStateException(
                        "Voxy engine classes not available: " + e.getMessage(), e);
            } catch (LinkageError e) {
                throw new IllegalStateException(
                        "Voxy engine class loading failed (MC remapping needed?): "
                        + e.getMessage(), e);
            }
        }
    }

    // ------------------------------------------------------------------ //
    //  Accessors for reflected types
    // ------------------------------------------------------------------ //

    /** Create an empty {@code VoxelizedSection} (16³ + mip pyramid). */
    public static Object createEmptySection() {
        ensureAvailable();
        try {
            return createEmptyMethod.invoke(null);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create VoxelizedSection", e);
        }
    }

    /** Get the Mapper from a WorldEngine instance. */
    public static Object getMapper(Object worldEngine) {
        ensureEngineBindings();
        try {
            return getMapperMethod.invoke(worldEngine);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get Mapper from WorldEngine", e);
        }
    }

    /** Compute the mip pyramid for a VoxelizedSection. */
    public static void mipSection(Object section, Object mapper) {
        ensureEngineBindings();
        try {
            mipSectionMethod.invoke(null, section, mapper);
        } catch (Exception e) {
            throw new RuntimeException("Failed to mip section", e);
        }
    }

    /** Insert a VoxelizedSection into a WorldEngine (blocking). */
    public static void insertUpdate(Object worldEngine, Object section) {
        ensureEngineBindings();
        try {
            insertUpdateMethod.invoke(null, worldEngine, section);
        } catch (Exception e) {
            throw new RuntimeException("Failed to insert section into Voxy world", e);
        }
    }

    /**
     * Check whether Voxy already has data for a section at the given level-0
     * coordinates.  Used to avoid overwriting real terrain with generated LODs.
     *
     * @param worldEngine the Voxy WorldEngine
     * @param sectionX    section X (blockX / 16)
     * @param sectionY    section Y (blockY / 16)
     * @param sectionZ    section Z (blockZ / 16)
     * @return true if Voxy already holds data for this section
     */
    public static boolean sectionExists(Object worldEngine,
                                         int sectionX, int sectionY, int sectionZ) {
        ensureEngineBindings();
        try {
            // acquireIfExists(lvl=0, x, y, z) returns null if no data
            Object section = acquireIfExistsMethod.invoke(
                    worldEngine, 0, sectionX, sectionY, sectionZ);
            if (section != null) {
                worldSectionReleaseMethod.invoke(section);  // release the ref
                return true;
            }
            return false;
        } catch (Exception e) {
            LOGGER.warning("sectionExists check failed: " + e.getMessage());
            return false;  // fail open — allow generation
        }
    }

    /**
     * Lazily bind the {@code WorldIdentifier} methods that require
     * {@code net.minecraft.world.World} as a parameter.
     *
     * <p>Deferred from {@link #isAvailable()} so that the core section API
     * (create/fill/mip/insert) is available in test environments where Minecraft
     * classes are not on the classpath.
     */
    private static void ensureWorldBindings() {
        ensureAvailable();
        if (worldBindingsReady) return;
        synchronized (VoxyCompat.class) {
            if (worldBindingsReady) return;
            try {
                Class<?> worldIdClass = Class.forName("me.cortex.voxy.commonImpl.WorldIdentifier");
                ofEngineMethod = worldIdClass.getMethod("ofEngine",
                        net.minecraft.world.World.class);
                ofEngineNullableMethod = worldIdClass.getMethod("ofEngineNullable",
                        net.minecraft.world.World.class);
                worldBindingsReady = true;
                LOGGER.info("Voxy WorldIdentifier bindings resolved");
            } catch (ClassNotFoundException | NoSuchMethodException | LinkageError e) {
                throw new IllegalStateException(
                        "Failed to bind Voxy WorldIdentifier (Minecraft not available?): " + e.getMessage(), e);
            }
        }
    }

    /**
     * Get the Voxy WorldEngine for a given Minecraft World.
     *
     * <p>Uses {@code WorldIdentifier.ofEngineNullable(World)} which returns null
     * if Voxy hasn't created the engine for this world yet.
     *
     * @param world the Minecraft World instance
     * @return the WorldEngine, or null if not yet available
     */
    public static Object getWorldEngine(net.minecraft.world.World world) {
        ensureWorldBindings();
        try {
            return ofEngineNullableMethod.invoke(null, world);
        } catch (Exception e) {
            LOGGER.warning("Failed to get WorldEngine: " + e.getMessage());
            return null;
        }
    }

    /**
     * Get the Voxy WorldEngine for a given Minecraft World, creating it if needed.
     *
     * @param world the Minecraft World instance
     * @return the WorldEngine (never null if Voxy is available)
     */
    public static Object getOrCreateWorldEngine(net.minecraft.world.World world) {
        ensureWorldBindings();
        try {
            return ofEngineMethod.invoke(null, world);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get/create WorldEngine", e);
        }
    }

    // ------------------------------------------------------------------ //
    //  VoxelizedSection field access
    // ------------------------------------------------------------------ //

    /** Set the position fields (x, y, z) on a VoxelizedSection. */
    public static void setSectionPosition(Object section, int x, int y, int z) {
        try {
            var xField = voxelizedSectionClass.getField("x");
            var yField = voxelizedSectionClass.getField("y");
            var zField = voxelizedSectionClass.getField("z");
            xField.setInt(section, x);
            yField.setInt(section, y);
            zField.setInt(section, z);
        } catch (Exception e) {
            throw new RuntimeException("Failed to set section position", e);
        }
    }

    /** Get the raw voxel data array from a VoxelizedSection. */
    public static long[] getSectionData(Object section) {
        try {
            var field = voxelizedSectionClass.getField("section");
            return (long[]) field.get(section);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get section data", e);
        }
    }

    /** Set the L0 non-air count on a VoxelizedSection. */
    public static void setNonAirCount(Object section, int count) {
        try {
            var field = voxelizedSectionClass.getField("lvl0NonAirCount");
            field.setInt(section, count);
        } catch (Exception e) {
            throw new RuntimeException("Failed to set non-air count", e);
        }
    }

    // ------------------------------------------------------------------ //
    //  Voxy 64-bit voxel encoding helpers
    // ------------------------------------------------------------------ //

    /** Voxel bit layout constants matching Voxy's Mapper. */
    public static final int BLOCK_ID_SHIFT = 27;
    public static final int BLOCK_ID_BITS  = 20;
    public static final long BLOCK_ID_MASK = ((1L << BLOCK_ID_BITS) - 1) << BLOCK_ID_SHIFT;

    public static final int BIOME_ID_SHIFT = 47;
    public static final int BIOME_ID_BITS  = 9;
    public static final long BIOME_ID_MASK = ((1L << BIOME_ID_BITS) - 1) << BIOME_ID_SHIFT;

    public static final int LIGHT_SHIFT = 56;

    /** Compose a 64-bit voxel value from block ID, biome ID, and light. */
    public static long composeVoxel(int blockId, int biomeId, int light) {
        return ((long) light << LIGHT_SHIFT)
             | ((long)(biomeId & 0x1FF) << BIOME_ID_SHIFT)
             | ((long)(blockId & ((1 << BLOCK_ID_BITS) - 1)) << BLOCK_ID_SHIFT);
    }

    /** True if the voxel is air (block ID field is zero). */
    public static boolean isAir(long voxel) {
        return (voxel & BLOCK_ID_MASK) == 0;
    }

    /** L0 index into VoxelizedSection.section[] for (x, y, z) in [0,15]. */
    public static int l0Index(int x, int y, int z) {
        return (y << 8) | (z << 4) | x;
    }

    // ------------------------------------------------------------------ //
    //  Direct WorldSection level writes (bypass insertUpdate)
    // ------------------------------------------------------------------ //

    /**
     * Write voxel data directly into a Voxy {@code WorldSection} at a specific
     * LOD level, bypassing the {@code insertUpdate()} path that always starts
     * at L0.
     *
     * <p>This is the core primitive for progressive LOD generation.  Each model
     * stage outputs block predictions at a specific resolution that maps 1:1
     * to a Voxy storage level.  We acquire the target-level WorldSection,
     * write voxels at the correct sub-position within the 32³ grid, mark it
     * dirty, and release.
     *
     * <h4>Coordinate math (from WorldUpdater.java):</h4>
     * <pre>
     *   WorldSection coords: (lvl, sectionX >> (lvl+1), sectionY >> (lvl+1), sectionZ >> (lvl+1))
     *   Sub-position:        bx = (sectionX & mask) << (4 - lvl)   where mask = (1 << (lvl+1)) - 1
     *   World section index: bx | (bz << 5) | (by << 10)
     * </pre>
     *
     * @param worldEngine the Voxy WorldEngine instance
     * @param lvl         Voxy storage level (1=LOD1 8³, 2=LOD2 4³, 3=LOD3 2³, 4=LOD4 1³)
     * @param sectionX    L0 section X (blockX / 16)
     * @param sectionY    L0 section Y (blockY / 16)
     * @param sectionZ    L0 section Z (blockZ / 16)
     * @param voxels      packed 64-bit voxel data to write, sized (16>>lvl)³,
     *                    indexed in YZX order: {@code voxels[(ly << (2*(4-lvl))) | (lz << (4-lvl)) | lx]}
     * @return number of non-air voxels written
     */
    public static int writeAtLevel(Object worldEngine, int lvl,
                                    int sectionX, int sectionY, int sectionZ,
                                    long[] voxels) {
        // Validate parameters before trying to load engine bindings
        if (lvl < 1 || lvl > 4) {
            throw new IllegalArgumentException("writeAtLevel: lvl must be 1-4, got " + lvl);
        }

        int cellsPerAxis = 16 >> lvl;  // 8,4,2,1 for lvl 1,2,3,4
        int expectedSize = cellsPerAxis * cellsPerAxis * cellsPerAxis;
        if (voxels.length != expectedSize) {
            throw new IllegalArgumentException("writeAtLevel: expected " + expectedSize
                    + " voxels for lvl " + lvl + ", got " + voxels.length);
        }

        ensureEngineBindings();

        // WorldSection coords at this level
        int wsX = sectionX >> (lvl + 1);
        int wsY = sectionY >> (lvl + 1);
        int wsZ = sectionZ >> (lvl + 1);

        try {
            // Acquire (or create) the WorldSection at the target level
            Object worldSection = acquireMethod.invoke(worldEngine, lvl, wsX, wsY, wsZ);

            // Get the raw 32³ data array
            long[] data = (long[]) worldSectionDataField.get(worldSection);

            // Compute base offset within the 32³ grid
            int mask = (1 << (lvl + 1)) - 1;
            int bx = (sectionX & mask) << (4 - lvl);
            int by = (sectionY & mask) << (4 - lvl);
            int bz = (sectionZ & mask) << (4 - lvl);

            // Write voxels into the correct sub-region
            int nonAir = 0;
            int srcIdx = 0;
            for (int ly = 0; ly < cellsPerAxis; ly++) {
                for (int lz = 0; lz < cellsPerAxis; lz++) {
                    for (int lx = 0; lx < cellsPerAxis; lx++) {
                        int dstIdx = (bx + lx) | ((bz + lz) << 5) | ((by + ly) << 10);
                        data[dstIdx] = voxels[srcIdx];
                        if (!isAir(voxels[srcIdx])) {
                            nonAir++;
                        }
                        srcIdx++;
                    }
                }
            }

            // Mark dirty → triggers save + mesh rebuild
            markDirtyMethod.invoke(worldEngine, worldSection);

            // Release the section
            worldSectionReleaseMethod.invoke(worldSection);

            // Propagate child existence bits to parent WorldSections
            // so Voxy's GPU octree traversal can navigate down to this data
            if (lvl < 4) {
                propagateChildExistence(worldEngine, lvl, sectionX, sectionY, sectionZ);
            }

            return nonAir;

        } catch (Exception e) {
            throw new RuntimeException("writeAtLevel failed at lvl=" + lvl
                    + " section=(" + sectionX + "," + sectionY + "," + sectionZ + ")", e);
        }
    }

    /**
     * Propagate child existence bits from the written level up to LOD4.
     *
     * <p>After writing voxels at {@code writtenLvl}, each ancestor WorldSection
     * needs its {@code nonEmptyChildren} byte updated so Voxy's GPU octree
     * traversal can navigate down to the written data.  Without this, the
     * shader sees {@code hasChildren(node) == false} and either skips the
     * subtree or renders only the coarsest fallback.
     *
     * <p>For each parent level from {@code writtenLvl + 1} to 4:
     * <ol>
     *   <li>Compute the child's octant index:  {@code (wsX&1) | ((wsZ&1)<<1) | ((wsY&1)<<2)}</li>
     *   <li>Acquire the parent WorldSection</li>
     *   <li>OR the child's bit into the parent's {@code nonEmptyChildren}</li>
     *   <li>{@code markDirty()} the parent so the render tree picks up the change</li>
     * </ol>
     *
     * @param worldEngine the Voxy WorldEngine instance
     * @param writtenLvl  the level we just wrote data to (1-4)
     * @param sectionX    L0 section X coordinate
     * @param sectionY    L0 section Y coordinate
     * @param sectionZ    L0 section Z coordinate
     */
    private static void propagateChildExistence(Object worldEngine,
                                                 int writtenLvl,
                                                 int sectionX, int sectionY,
                                                 int sectionZ) {
        try {
            for (int parentLvl = writtenLvl + 1; parentLvl <= 4; parentLvl++) {
                int childLvl = parentLvl - 1;

                // Child's WorldSection coords at childLvl
                int childWsX = sectionX >> (childLvl + 1);
                int childWsY = sectionY >> (childLvl + 1);
                int childWsZ = sectionZ >> (childLvl + 1);

                // Octant index matches WorldSection.getChildIndex(x, y, z)
                int childIdx = (childWsX & 1)
                             | ((childWsZ & 1) << 1)
                             | ((childWsY & 1) << 2);
                byte childBit = (byte) (1 << childIdx);

                // Parent's WorldSection coords at parentLvl
                int parentWsX = sectionX >> (parentLvl + 1);
                int parentWsY = sectionY >> (parentLvl + 1);
                int parentWsZ = sectionZ >> (parentLvl + 1);

                Object parentSection = acquireMethod.invoke(
                        worldEngine, parentLvl, parentWsX, parentWsY, parentWsZ);

                // Read current nonEmptyChildren, OR in the child bit
                byte current = worldSectionNonEmptyChildrenField.getByte(parentSection);
                byte updated = (byte) (current | childBit);
                if (updated != current) {
                    worldSectionNonEmptyChildrenField.setByte(parentSection, updated);
                    // Mark dirty → triggers mesh rebuild + child existence
                    // propagation to the GPU octree via processChildChange()
                    markDirtyMethod.invoke(worldEngine, parentSection);
                }

                worldSectionReleaseMethod.invoke(parentSection);
            }
        } catch (Exception e) {
            LOGGER.warning("propagateChildExistence failed at writtenLvl="
                    + writtenLvl + ": " + e.getMessage());
        }
    }

    /**
     * Check whether a Voxy WorldSection exists at a specific level and
     * WorldSection coordinate.
     *
     * @param worldEngine the Voxy WorldEngine
     * @param lvl         storage level (0-4)
     * @param wsX         WorldSection X at this level
     * @param wsY         WorldSection Y at this level
     * @param wsZ         WorldSection Z at this level
     * @return true if Voxy already holds data at this level/position
     */
    public static boolean sectionExistsAtLevel(Object worldEngine, int lvl,
                                                int wsX, int wsY, int wsZ) {
        ensureEngineBindings();
        try {
            Object section = acquireIfExistsMethod.invoke(worldEngine, lvl, wsX, wsY, wsZ);
            if (section != null) {
                worldSectionReleaseMethod.invoke(section);
                return true;
            }
            return false;
        } catch (Exception e) {
            LOGGER.warning("sectionExistsAtLevel check failed: " + e.getMessage());
            return false;
        }
    }

    // ------------------------------------------------------------------ //
    //  Internal
    // ------------------------------------------------------------------ //

    private static void ensureAvailable() {
        if (!isAvailable()) {
            throw new IllegalStateException("Voxy is not available");
        }
    }
}

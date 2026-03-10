package com.rhythmatician.lodiffusion.voxy;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the heightmap fallback terrain generator.
 *
 * <p>Tests the block-selection logic ({@link HeightmapFallbackGenerator#pickBlockId})
 * and biome classification ({@link HeightmapFallbackGenerator#isSandyBiome})
 * without requiring Voxy or Minecraft runtime bindings.
 */
class HeightmapFallbackGeneratorTest {

    // Dummy Voxy block IDs for testing (arbitrary distinct values)
    private static final HeightmapFallbackGenerator.FallbackBlockIds IDS =
            new HeightmapFallbackGenerator.FallbackBlockIds(
                    0,   // air
                    1,   // stone
                    2,   // deepslate
                    3,   // dirt
                    4,   // grassBlock
                    5,   // sand
                    6    // water
            );

    // ------------------------------------------------------------------ //
    //  Biome classification
    // ------------------------------------------------------------------ //

    @Test
    @DisplayName("Desert (index 12) is a sandy biome")
    void desertIsSandy() {
        assertTrue(HeightmapFallbackGenerator.isSandyBiome(12));
    }

    @Test
    @DisplayName("Beach (index 2) is a sandy biome")
    void beachIsSandy() {
        assertTrue(HeightmapFallbackGenerator.isSandyBiome(2));
    }

    @Test
    @DisplayName("Plains (index 34) is not a sandy biome")
    void plainsIsNotSandy() {
        assertFalse(HeightmapFallbackGenerator.isSandyBiome(34));
    }

    @Test
    @DisplayName("Forest (index 16) is not a sandy biome")
    void forestIsNotSandy() {
        assertFalse(HeightmapFallbackGenerator.isSandyBiome(16));
    }

    // ------------------------------------------------------------------ //
    //  Block selection — normal (non-sandy) biome, surface at y=70
    //  No water: groundBlockY == waterSurfaceBlockY == 70
    // ------------------------------------------------------------------ //

    @Test
    @DisplayName("y=69 (surface-1) → grass_block (top solid block)")
    void topBlockIsGrass() {
        assertEquals(IDS.grassBlock(),
                HeightmapFallbackGenerator.pickBlockId(69, 70, 70, false, IDS));
    }

    @Test
    @DisplayName("y=68 (surface-2) → dirt")
    void secondBlockIsDirt() {
        assertEquals(IDS.dirt(),
                HeightmapFallbackGenerator.pickBlockId(68, 70, 70, false, IDS));
    }

    @Test
    @DisplayName("y=67 (surface-3) → dirt")
    void thirdBlockIsDirt() {
        assertEquals(IDS.dirt(),
                HeightmapFallbackGenerator.pickBlockId(67, 70, 70, false, IDS));
    }

    @Test
    @DisplayName("y=66 (below top 3, above y=0) → stone")
    void belowDirtIsStone() {
        assertEquals(IDS.stone(),
                HeightmapFallbackGenerator.pickBlockId(66, 70, 70, false, IDS));
    }

    @Test
    @DisplayName("y=1 (above y=0) → stone")
    void justAboveZeroIsStone() {
        assertEquals(IDS.stone(),
                HeightmapFallbackGenerator.pickBlockId(1, 70, 70, false, IDS));
    }

    @Test
    @DisplayName("y=0 (within top 3 check range if surface is 3, otherwise stone)")
    void yZeroWithHighSurface() {
        assertEquals(IDS.stone(),
                HeightmapFallbackGenerator.pickBlockId(0, 70, 70, false, IDS));
    }

    @Test
    @DisplayName("y=-1 (below y=0) → deepslate")
    void belowZeroIsDeepslate() {
        assertEquals(IDS.deepslate(),
                HeightmapFallbackGenerator.pickBlockId(-1, 70, 70, false, IDS));
    }

    @Test
    @DisplayName("y=-64 (deep underground) → deepslate")
    void deepUndergroundIsDeepslate() {
        assertEquals(IDS.deepslate(),
                HeightmapFallbackGenerator.pickBlockId(-64, 70, 70, false, IDS));
    }

    @Test
    @DisplayName("y=70 (at surface, above sea level) → air")
    void atSurfaceAboveSeaLevelIsAir() {
        assertEquals(IDS.air(),
                HeightmapFallbackGenerator.pickBlockId(70, 70, 70, false, IDS));
    }

    @Test
    @DisplayName("y=100 (well above surface, above sea level) → air")
    void highAboveSurfaceIsAir() {
        assertEquals(IDS.air(),
                HeightmapFallbackGenerator.pickBlockId(100, 70, 70, false, IDS));
    }

    // ------------------------------------------------------------------ //
    //  Block selection — sandy biome, surface at y=70
    // ------------------------------------------------------------------ //

    @Test
    @DisplayName("Sandy biome: top block (y=69) → sand")
    void sandyTopBlockIsSand() {
        assertEquals(IDS.sand(),
                HeightmapFallbackGenerator.pickBlockId(69, 70, 70, true, IDS));
    }

    @Test
    @DisplayName("Sandy biome: second block (y=68) → sand")
    void sandySecondBlockIsSand() {
        assertEquals(IDS.sand(),
                HeightmapFallbackGenerator.pickBlockId(68, 70, 70, true, IDS));
    }

    @Test
    @DisplayName("Sandy biome: third block (y=67) → sand")
    void sandyThirdBlockIsSand() {
        assertEquals(IDS.sand(),
                HeightmapFallbackGenerator.pickBlockId(67, 70, 70, true, IDS));
    }

    @Test
    @DisplayName("Sandy biome: below top 3 (y=66) → stone (not sand)")
    void sandyBelowTopThreeIsStone() {
        assertEquals(IDS.stone(),
                HeightmapFallbackGenerator.pickBlockId(66, 70, 70, true, IDS));
    }

    // ------------------------------------------------------------------ //
    //  Block selection — ocean/deep underwater (no ocean floor data)
    //  groundBlockY == waterSurfaceBlockY because ocean floor matches
    //  surface heightmap when there's no separate floor data
    // ------------------------------------------------------------------ //

    @Test
    @DisplayName("Underwater: y=60 (at surface, below sea level) → water")
    void underwaterAtSurface() {
        assertEquals(IDS.water(),
                HeightmapFallbackGenerator.pickBlockId(60, 60, 60, false, IDS));
    }

    @Test
    @DisplayName("Underwater: y=62 (above surface 55, below sea level 63) → water")
    void underwaterAboveSurfaceBelowSea() {
        assertEquals(IDS.water(),
                HeightmapFallbackGenerator.pickBlockId(62, 55, 55, false, IDS));
    }

    @Test
    @DisplayName("Underwater: y=63 (at sea level) → air")
    void atSeaLevelIsAir() {
        assertEquals(IDS.air(),
                HeightmapFallbackGenerator.pickBlockId(63, 55, 55, false, IDS));
    }

    @Test
    @DisplayName("Underwater: solid blocks still have correct top 3")
    void underwaterSolidBlocksCorrect() {
        assertEquals(IDS.grassBlock(),
                HeightmapFallbackGenerator.pickBlockId(54, 55, 55, false, IDS));
        assertEquals(IDS.dirt(),
                HeightmapFallbackGenerator.pickBlockId(53, 55, 55, false, IDS));
        assertEquals(IDS.dirt(),
                HeightmapFallbackGenerator.pickBlockId(52, 55, 55, false, IDS));
        assertEquals(IDS.stone(),
                HeightmapFallbackGenerator.pickBlockId(51, 55, 55, false, IDS));
    }

    // ------------------------------------------------------------------ //
    //  Block selection — river / shallow water
    //  River: groundBlockY=58, waterSurfaceBlockY=62
    //  Water fills between y=58..61 inclusive
    // ------------------------------------------------------------------ //

    @Test
    @DisplayName("River: y=57 (ground - 1) → grass_block (top solid)")
    void riverTopSolidIsGrass() {
        assertEquals(IDS.grassBlock(),
                HeightmapFallbackGenerator.pickBlockId(57, 58, 62, false, IDS));
    }

    @Test
    @DisplayName("River: y=56 (ground - 2) → dirt")
    void riverSecondSolidIsDirt() {
        assertEquals(IDS.dirt(),
                HeightmapFallbackGenerator.pickBlockId(56, 58, 62, false, IDS));
    }

    @Test
    @DisplayName("River: y=55 (ground - 3) → dirt")
    void riverThirdSolidIsDirt() {
        assertEquals(IDS.dirt(),
                HeightmapFallbackGenerator.pickBlockId(55, 58, 62, false, IDS));
    }

    @Test
    @DisplayName("River: y=54 (below top 3, above 0) → stone")
    void riverBelowTopThreeIsStone() {
        assertEquals(IDS.stone(),
                HeightmapFallbackGenerator.pickBlockId(54, 58, 62, false, IDS));
    }

    @Test
    @DisplayName("River: y=58 (at ground level, below water surface) → water")
    void riverAtGroundIsWater() {
        assertEquals(IDS.water(),
                HeightmapFallbackGenerator.pickBlockId(58, 58, 62, false, IDS));
    }

    @Test
    @DisplayName("River: y=60 (between ground and water surface) → water")
    void riverMidWaterIsWater() {
        assertEquals(IDS.water(),
                HeightmapFallbackGenerator.pickBlockId(60, 58, 62, false, IDS));
    }

    @Test
    @DisplayName("River: y=61 (just below water surface) → water")
    void riverJustBelowWaterSurfaceIsWater() {
        assertEquals(IDS.water(),
                HeightmapFallbackGenerator.pickBlockId(61, 58, 62, false, IDS));
    }

    @Test
    @DisplayName("River: y=62 (at water surface, below sea level) → water")
    void riverAtWaterSurfaceBelowSeaLevel() {
        // waterSurfaceBlockY = 62, y=62 >= ground, >= waterSurface, but < SEA_LEVEL (63)
        assertEquals(IDS.water(),
                HeightmapFallbackGenerator.pickBlockId(62, 58, 62, false, IDS));
    }

    @Test
    @DisplayName("River: y=63 (above water surface, at sea level) → air")
    void riverAboveWaterSurfaceIsAir() {
        assertEquals(IDS.air(),
                HeightmapFallbackGenerator.pickBlockId(63, 58, 62, false, IDS));
    }

    @Test
    @DisplayName("River: sandy biome → sand for top 3 solid, water above")
    void riverSandyBiomeSand() {
        assertEquals(IDS.sand(),
                HeightmapFallbackGenerator.pickBlockId(57, 58, 62, true, IDS));
        assertEquals(IDS.sand(),
                HeightmapFallbackGenerator.pickBlockId(56, 58, 62, true, IDS));
        assertEquals(IDS.sand(),
                HeightmapFallbackGenerator.pickBlockId(55, 58, 62, true, IDS));
        assertEquals(IDS.water(),
                HeightmapFallbackGenerator.pickBlockId(58, 58, 62, true, IDS));
    }

    // ------------------------------------------------------------------ //
    //  Block selection — ocean (deep water)
    //  Ocean: groundBlockY=40, waterSurfaceBlockY=62
    // ------------------------------------------------------------------ //

    @Test
    @DisplayName("Ocean: y=39 (ground - 1) → grass_block (top solid)")
    void oceanTopSolidIsGrass() {
        assertEquals(IDS.grassBlock(),
                HeightmapFallbackGenerator.pickBlockId(39, 40, 62, false, IDS));
    }

    @Test
    @DisplayName("Ocean: y=40 (at ground, below water surface) → water")
    void oceanAtGroundIsWater() {
        assertEquals(IDS.water(),
                HeightmapFallbackGenerator.pickBlockId(40, 40, 62, false, IDS));
    }

    @Test
    @DisplayName("Ocean: y=50 (mid-water) → water")
    void oceanMidWaterIsWater() {
        assertEquals(IDS.water(),
                HeightmapFallbackGenerator.pickBlockId(50, 40, 62, false, IDS));
    }

    @Test
    @DisplayName("Ocean: y=62 (at water surface, below sea level) → water")
    void oceanAtWaterSurfaceIsWater() {
        assertEquals(IDS.water(),
                HeightmapFallbackGenerator.pickBlockId(62, 40, 62, false, IDS));
    }

    @Test
    @DisplayName("Ocean: y=63 (above water surface, at sea level) → air")
    void oceanAboveIsAir() {
        assertEquals(IDS.air(),
                HeightmapFallbackGenerator.pickBlockId(63, 40, 62, false, IDS));
    }

    // ------------------------------------------------------------------ //
    //  Edge case: very low surface (surface at y=2)
    // ------------------------------------------------------------------ //

    @Test
    @DisplayName("Low surface (y=2): top block y=1 → grass, y=0 → dirt, y=-1 → dirt")
    void lowSurfaceTransition() {
        assertEquals(IDS.grassBlock(),
                HeightmapFallbackGenerator.pickBlockId(1, 2, 2, false, IDS));
        assertEquals(IDS.dirt(),
                HeightmapFallbackGenerator.pickBlockId(0, 2, 2, false, IDS));
        // y=-1 is within top 3 (depth=2), top-3 check takes priority over deepslate
        assertEquals(IDS.dirt(),
                HeightmapFallbackGenerator.pickBlockId(-1, 2, 2, false, IDS));
        assertEquals(IDS.deepslate(),
                HeightmapFallbackGenerator.pickBlockId(-2, 2, 2, false, IDS));
    }

    // ------------------------------------------------------------------ //
    //  Sea level constant
    // ------------------------------------------------------------------ //

    @Test
    @DisplayName("Sea level is 63")
    void seaLevelIs63() {
        assertEquals(63, HeightmapFallbackGenerator.SEA_LEVEL);
    }
}

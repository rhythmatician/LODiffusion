package com.rhythmatician.lodiffusion.training;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for the {@link Voxel} class.
 * Validates sparse voxel coordinate storage used in octree-aligned terrain representation.
 */
class VoxelTest {

    @Test
    void testConstructorAndGetters() {
        Voxel v = new Voxel(3, 5, 7);
        assertEquals(3, v.getX(), "X should be 3");
        assertEquals(5, v.getY(), "Y should be 5");
        assertEquals(7, v.getZ(), "Z should be 7");
    }

    @Test
    void testConstructorWithZeroCoordinates() {
        Voxel v = new Voxel(0, 0, 0);
        assertEquals(0, v.getX());
        assertEquals(0, v.getY());
        assertEquals(0, v.getZ());
    }

    @Test
    void testEquals_SameCoordinates() {
        Voxel a = new Voxel(1, 2, 3);
        Voxel b = new Voxel(1, 2, 3);
        assertEquals(a, b, "Voxels with same coordinates should be equal");
    }

    @Test
    void testEquals_DifferentX() {
        Voxel a = new Voxel(1, 2, 3);
        Voxel b = new Voxel(9, 2, 3);
        assertNotEquals(a, b, "Voxels with different X should not be equal");
    }

    @Test
    void testEquals_DifferentY() {
        Voxel a = new Voxel(1, 2, 3);
        Voxel b = new Voxel(1, 9, 3);
        assertNotEquals(a, b, "Voxels with different Y should not be equal");
    }

    @Test
    void testEquals_DifferentZ() {
        Voxel a = new Voxel(1, 2, 3);
        Voxel b = new Voxel(1, 2, 9);
        assertNotEquals(a, b, "Voxels with different Z should not be equal");
    }

    @Test
    void testEquals_Self() {
        Voxel v = new Voxel(4, 5, 6);
        assertEquals(v, v, "Voxel should equal itself");
    }

    @Test
    void testEquals_Null() {
        Voxel v = new Voxel(1, 2, 3);
        assertNotEquals(v, null, "Voxel should not equal null");
    }

    @Test
    void testEquals_DifferentType() {
        Voxel v = new Voxel(1, 2, 3);
        assertNotEquals(v, "not a voxel", "Voxel should not equal a non-Voxel object");
    }

    @Test
    void testHashCode_EqualVoxels() {
        Voxel a = new Voxel(2, 4, 6);
        Voxel b = new Voxel(2, 4, 6);
        assertEquals(a.hashCode(), b.hashCode(), "Equal voxels must have identical hash codes");
    }

    @Test
    void testHashCode_UsedAsSetKey() {
        java.util.Set<Voxel> set = new java.util.HashSet<>();
        set.add(new Voxel(0, 0, 0));
        set.add(new Voxel(0, 0, 0)); // duplicate
        set.add(new Voxel(1, 0, 0));
        assertEquals(2, set.size(), "Set should deduplicate equal voxels");
    }

    @Test
    void testToString_ContainsCoordinates() {
        Voxel v = new Voxel(1, 2, 3);
        String s = v.toString();
        assertNotNull(s);
        assertTrue(s.contains("1"), "toString should include X");
        assertTrue(s.contains("2"), "toString should include Y");
        assertTrue(s.contains("3"), "toString should include Z");
    }
}

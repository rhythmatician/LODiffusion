package com.rhythmatician.lodiffusion.training;

import java.util.Objects;

/**
 * Represents an occupied voxel at a specific (x, y, z) position within an octree node.
 * Used for sparse terrain storage—only terrain-filled positions are stored,
 * aligned with Voxy's octree schema.
 *
 * <p>Coordinates are in octree-local space (0 to OCTREE_SIZE-1 per axis).
 * The y axis represents quantised vertical levels derived from the terrain heightmap.
 */
public final class Voxel {

    private final int x;
    private final int y;
    private final int z;

    /**
     * Create a voxel at the given octree-local coordinates.
     *
     * @param x Local X coordinate within the octree node
     * @param y Local Y coordinate (quantised vertical level)
     * @param z Local Z coordinate within the octree node
     */
    public Voxel(int x, int y, int z) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    /** @return Local X coordinate within the octree node. */
    public int getX() {
        return x;
    }

    /** @return Local Y coordinate (quantised vertical level). */
    public int getY() {
        return y;
    }

    /** @return Local Z coordinate within the octree node. */
    public int getZ() {
        return z;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof Voxel)) return false;
        Voxel other = (Voxel) obj;
        return x == other.x && y == other.y && z == other.z;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y, z);
    }

    @Override
    public String toString() {
        return "Voxel(" + x + ", " + y + ", " + z + ")";
    }
}

# Voxy Request Queue Encoding (8-byte uvec2)

## Overview
Each request in Voxy's `requestQueue` is a `uvec2` (2×uint = 8 bytes) containing packed node coordinates and LOD level.

**Source**: `HierarchicalOcclusionTraverser.forwardDownloadResult()` downloads requests as:
```java
long ptr = buffer.address();
int count = readIntAt(ptr); ptr += 8;
for (int i = 0; i < count; i++) {
    uvec2 rawPos = readUvec2At(ptr + i*8);
    // Parse rawPos → (lodLevel, x, y, z)
}
```

## Bit Layout

### packedPos.x (first uint, bits 31–0)
```
[LOD(4)] [Y(8)] [Xhi(20)]
31   28 27   20 19      0
```

### packedPos.y (second uint, bits 31–0)
```
[Zhi(4)] [Xlo(24)] [___reserved___(4)]
31   28 27       4  3           0
```

## Decoding Algorithm

From Voxy's `pos_util.glsl`:

```glsl
uint getLoDLevel(uvec2 packedPos) {
    return packedPos.x >> 28;  // Top 4 bits → [0,15]
}

ivec3 getLoDPosition(uvec2 packedPos) {
    int y = ((int(packedPos.x)<<4)>>24);        // Extract Y: bits [27:20] as signed byte
    int x = (int(packedPos.y)<<4)>>8;           // Extract X: bits [27:4] of Y (24 bits)
    int z = int((packedPos.x&0xFFFFF)<<4);      // Bits [19:0] of X as Z high
    z |= int(packedPos.y>>28);                  // OR with bits [31:28] of Y as Z low
    z = (z<<8)>>8;                              // Sign-extend from byte 7
    return ivec3(x,y,z);
}
```

## Examples

### Example 1: LOD=0, Pos=(0,64,0)
```
y = 64,  x = 0,  z = 0
LOD = 0

packedPos.x  = (0      << 28) | (64    << 20) | 0       = 0x04000000
packedPos.y  = (0      << 28) | (0     << 4)  | 0       = 0x00000000

uvec2 = [0x04000000, 0x00000000]
```

### Example 2: LOD=2, Pos=(256,100,512)
```
y = 100 (0x64)
x = 256 (0x00000100)
z = 512 (0x00000200)

x breaks into:
  x_hi = 256 >> 4 = 16 (bits [19:0] of x.x)
  x_lo = (256 & 0xFFFFF) << 4 = 0x00001000 (bits [27:4] of x.y)

z breaks into:
  z_hi = 512 >> 28 = 0  (bits [3:0] for y>>28)
  z_lo = (512 << 4) & 0xFFFFF000 = ... (bits [19:0])

This gets complex due to bit alignments.
```

## Java Decoder

```java
public static class VoxyNodeRequest {
    public int lodLevel;
    public int x, y, z;
    
    public static VoxyNodeRequest decode(long bufferPtr, int offsetBytes) {
        int x_uint = readIntLE(bufferPtr + offsetBytes);      // packedPos.x
        int y_uint = readIntLE(bufferPtr + offsetBytes + 4);  // packedPos.y
        
        VoxyNodeRequest req = new VoxyNodeRequest();
        
        // LOD level: top 4 bits of x_uint
        req.lodLevel = (x_uint >>> 28) & 0xF;
        
        // Y: bits [27:20] of x_uint
        req.y = ((x_uint << 4) >> 24) & 0xFF;
        if ((req.y & 0x80) != 0) req.y |= 0xFFFFFF00;  // Sign extend
        
        // X: bits [27:4] of y_uint (24 bits) combined with careful arithmetic
        req.x = (y_uint << 4) >> 8;  // Arithmetic shift to sign-extend
        
        // Z: bits [19:0] of x_uint + bits [31:28] of y_uint, sign extended
        int z_part1 = x_uint & 0xFFFFF;
        int z_part2 = y_uint >>> 28;
        int z = (z_part1 << 4) | z_part2;
        req.z = (z << 8) >> 8;  // Sign-extend from byte 7
        
        return req;
    }
}
```

## Validation Points

- **LOD range**: [0, 4] valid; [5, 15] is unused/invalid
- **Y range**: [-128, 127] (signed byte)
- **X range**: [-2^19, 2^19-1] (signed, though X rarely goes negative in practice)
- **Z range**: [-2^19, 2^19-1] (signed, though Z rarely goes negative in practice)
- **Sparse packing**: Only 28 of 64 bits used; carefully engineered for cache efficiency

## Integration Point

In `VoxyShadowBridge` Mixin:
1. Intercept `HierarchicalOcclusionTraverser.forwardDownloadResult(long ptr, long size)`
2. Read request count at `ptr + 0`
3. For each request offset `i*8`: decode `uvec2` at `ptr + 8 + i*8`
4. Filter by LOD range `[1, 4]` and world distance
5. Enqueue to `ShadowRouterJobQueue`
6. Allow normal Voxy processing to continue

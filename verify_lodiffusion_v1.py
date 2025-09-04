import sys

import numpy as np
import onnx
import onnxruntime as ort

# Progressive LOD refinement contracts (updated for flexible_unet3d models)
PROGRESSIVE_LOD_CONTRACTS = {
    "lod4to3": {
        "inputs": {
            "parent_voxel": [1, 1, 1, 1, 1],  # Single voxel (entire subchunk)
            "biome_patch": [1, 256, 16, 16],  # Biome data (one-hot encoded)
            "heightmap_patch": [1, 1, 16, 16, 1],  # Height data (16x16 chunk)
            "river_patch": [1, 1, 16, 16, 1],  # River feature data
        },
        "outputs": {
            "air_mask_logits": [1, 1, 2, 2, 2],  # 2x2x2 air/solid mask
            "block_type_logits": [1, 1104, 2, 2, 2],  # 2x2x2 block types
        },
    },
    "lod3to2": {
        "inputs": {
            "parent_voxel": [1, 1, 2, 2, 2],  # 2x2x2 parent voxels
            "biome_patch": [1, 256, 16, 16],  # Biome data (one-hot encoded)
            "heightmap_patch": [1, 1, 16, 16, 1],  # Height data (16x16 chunk)
            "river_patch": [1, 1, 16, 16, 1],  # River feature data
        },
        "outputs": {
            "air_mask_logits": [1, 1, 4, 4, 4],  # 4x4x4 air/solid mask
            "block_type_logits": [1, 1104, 4, 4, 4],  # 4x4x4 block types
        },
    },
    "lod2to1": {
        "inputs": {
            "parent_voxel": [1, 1, 4, 4, 4],  # 4x4x4 parent voxels
            "biome_patch": [1, 256, 16, 16],  # Biome data (one-hot encoded)
            "heightmap_patch": [1, 1, 16, 16, 1],  # Height data (16x16 chunk)
            "river_patch": [1, 1, 16, 16, 1],  # River feature data
        },
        "outputs": {
            "air_mask_logits": [1, 1, 8, 8, 8],  # 8x8x8 air/solid mask
            "block_type_logits": [1, 1104, 8, 8, 8],  # 8x8x8 block types
        },
    },
    "lod1to0": {
        "inputs": {
            "parent_voxel": [1, 1, 8, 8, 8],  # 8x8x8 parent voxels
            "biome_patch": [1, 256, 16, 16],  # Biome data (one-hot encoded)
            "heightmap_patch": [1, 1, 16, 16, 1],  # Height data (16x16 chunk)
            "river_patch": [1, 1, 16, 16, 1],  # River feature data
        },
        "outputs": {
            "air_mask_logits": [1, 1, 16, 16, 16],  # 16x16x16 air/solid mask
            "block_type_logits": [1, 1104, 16, 16, 16],  # 16x16x16 block types
        },
    },
}

# Common requirements for all LOD levels
COMMON_REQ = {
    "opset_min": 17,
    "forbid_ops": {"Loop", "If", "Scan"},
}

# Legacy contract for backwards compatibility
LEGACY_REQ = {
    "inputs": {
        "x_parent": [1, 1, 8, 8, 8],
        "x_biome": [1, "NB", 16, 16, 1],  # NB > 1, 16x16 chunk size
        "x_height": [1, 1, 16, 16, 1],  # 16x16 chunk size
        "x_lod": [1, 1],
    },
    "outputs": {"air_mask": [1, 1, 16, 16, 16], "block_logits": [1, 1104, 16, 16, 16]},
    **COMMON_REQ,
}


def get_shape(v):
    t = v.type.tensor_type
    dims = []
    for d in t.shape.dim:
        if d.dim_value:
            dims.append(d.dim_value)
        elif d.dim_param:
            return None  # dynamic
        else:
            return None
    return dims


def main(path, lod_type="lod1to0"):
    """
    Verify ONNX model against progressive LOD contracts.

    Args:
        path: Path to ONNX model file
        lod_type: LOD refinement type to verify against
                 ("lod4to3", "lod3to2", "lod2to1", "lod1to0", or "legacy")
    """
    if lod_type == "legacy":
        REQ = LEGACY_REQ
    elif lod_type in PROGRESSIVE_LOD_CONTRACTS:
        contract = PROGRESSIVE_LOD_CONTRACTS[lod_type]
        REQ = {**contract, **COMMON_REQ}
    else:
        print(f"FAIL: Unknown LOD type '{lod_type}'")
        available = list(PROGRESSIVE_LOD_CONTRACTS.keys()) + ["legacy"]
        print(f"Available: {available}")
        return 1

    m = onnx.load(path)
    onnx.checker.check_model(m)

    # opset
    opset = max(o.version for o in m.opset_import)
    if opset < REQ["opset_min"]:
        print(f"FAIL: opset {opset} < {REQ['opset_min']}")
        return 1

    # control-flow ban
    bad = {n.op_type for n in m.graph.node} & REQ["forbid_ops"]
    if bad:
        print(f"FAIL: control-flow ops present: {sorted(bad)}")
        return 1

    # names → shapes
    ins = {i.name: get_shape(i) for i in m.graph.input}
    outs = {o.name: get_shape(o) for o in m.graph.output}

    # required names
    for name, shp in REQ["inputs"].items():
        if name not in ins:
            print(f"FAIL: missing input {name}")
            return 1
        if ins[name] is None:
            print(f"FAIL: input {name} has dynamic shape")
            return 1
    for name, shp in REQ["outputs"].items():
        if name not in outs:
            print(f"FAIL: missing output {name}")
            return 1
        if outs[name] is None:
            print(f"FAIL: output {name} has dynamic shape")
            return 1

    # shape checks (handle new input names for progressive models)
    def eq(a, b):
        return all(x == y for x, y in zip(a, b))

    if lod_type == "legacy":
        # Legacy validation with x_parent, x_biome, x_height, x_lod
        if ins["x_parent"] != REQ["inputs"]["x_parent"]:
            print(
                f"FAIL: x_parent shape - got: {ins['x_parent']}, "
                f"want: {REQ['inputs']['x_parent']}"
            )
            return 1
        if ins["x_height"] != REQ["inputs"]["x_height"]:
            print(
                f"FAIL: x_height shape - got: {ins['x_height']}, "
                f"want: {REQ['inputs']['x_height']}"
            )
            return 1
        if ins["x_lod"] != REQ["inputs"]["x_lod"]:
            print(
                f"FAIL: x_lod shape - got: {ins['x_lod']}, "
                f"want: {REQ['inputs']['x_lod']}"
            )
            return 1
        xb = ins["x_biome"]
        if not (validate_biome_shape(xb)):
            print(f"FAIL: x_biome shape {xb}")
            return 1

        if outs["air_mask"] != REQ["outputs"]["air_mask"]:
            print(f"FAIL: air_mask shape {outs['air_mask']}")
            return 1
        if outs["block_logits"] != REQ["outputs"]["block_logits"]:
            print(f"FAIL: block_logits shape {outs['block_logits']}")
            return 1
    else:
        # Progressive model validation with new input names
        if ins["parent_voxel"] != REQ["inputs"]["parent_voxel"]:
            print(
                f"FAIL: parent_voxel shape - got: {ins['parent_voxel']}, "
                f"want: {REQ['inputs']['parent_voxel']}"
            )
            return 1
        if ins["biome_patch"] != REQ["inputs"]["biome_patch"]:
            print(
                f"FAIL: biome_patch shape - got: {ins['biome_patch']}, "
                f"want: {REQ['inputs']['biome_patch']}"
            )
            return 1
        if ins["heightmap_patch"] != REQ["inputs"]["heightmap_patch"]:
            print(
                f"FAIL: heightmap_patch shape - "
                f"got: {ins['heightmap_patch']}, "
                f"want: {REQ['inputs']['heightmap_patch']}"
            )
            return 1
        if ins["river_patch"] != REQ["inputs"]["river_patch"]:
            print(
                f"FAIL: river_patch shape - got: {ins['river_patch']}, "
                f"want: {REQ['inputs']['river_patch']}"
            )
            return 1

        if outs["air_mask_logits"] != REQ["outputs"]["air_mask_logits"]:
            print(f"FAIL: air_mask_logits shape {outs['air_mask_logits']}")
            return 1
        if outs["block_type_logits"] != REQ["outputs"]["block_type_logits"]:
            print(f"FAIL: block_type_logits shape {outs['block_type_logits']}")
            return 1

    # runtime smoke: single-thread, dummy inputs
    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 1
    sess_opt.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        path, sess_options=sess_opt, providers=["CPUExecutionProvider"]
    )

    if lod_type == "legacy":
        NB = xb[1]
        # Create test inputs based on parent shape
        parent_shape = REQ["inputs"]["x_parent"]
        x = {
            "x_parent": np.zeros(parent_shape, np.float32),
            "x_biome": np.eye(NB, dtype=np.float32)[
                np.zeros((1, 16, 16, 1), int)
            ].transpose(0, 4, 1, 2, 3),
            "x_height": np.zeros((1, 1, 16, 16, 1), np.float32),
            "x_lod": np.zeros((1, 1), np.float32),
        }
        y = sess.run(None, x)

        expected_air_shape = tuple(REQ["outputs"]["air_mask"])
        expected_logits_shape = tuple(REQ["outputs"]["block_logits"])
        ok = y[1].shape == expected_air_shape and y[0].shape == expected_logits_shape
        if not ok:
            print(f"FAIL: runtime output shapes {y[0].shape}, {y[1].shape}")
            print(f"Expected: {expected_logits_shape}, {expected_air_shape}")
            return 1
        NB_str = f"NB={NB}"
    else:
        # Progressive models with new input names
        parent_shape = REQ["inputs"]["parent_voxel"]
        x = {
            "parent_voxel": np.zeros(parent_shape, np.float32),
            "biome_patch": np.zeros((1, 256, 16, 16), np.float32),
            "heightmap_patch": np.zeros((1, 1, 16, 16, 1), np.float32),
            "river_patch": np.zeros((1, 1, 16, 16, 1), np.float32),
        }
        y = sess.run(None, x)

        expected_air_shape = tuple(REQ["outputs"]["air_mask_logits"])
        expected_logits_shape = tuple(REQ["outputs"]["block_type_logits"])
        ok = y[0].shape == expected_air_shape and y[1].shape == expected_logits_shape
        if not ok:
            print(f"FAIL: runtime output shapes {y[0].shape}, {y[1].shape}")
            print(f"Expected: {expected_air_shape}, {expected_logits_shape}")
            return 1
        NB_str = "biomes=256"

    print(
        f"READY: {lod_type} contract, opset {opset}, {NB_str}, "
        f"shapes OK, runtime OK"
    )
    return 0


def validate_biome_shape(xb):
    return (
        len(xb) == 5
        and xb[0] == 1
        and xb[2:] == [16, 16, 1]
        and isinstance(xb[1], int)
        and xb[1] > 1
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify ONNX model contracts")
    parser.add_argument(
        "model_path",
        nargs="?",
        default="artifacts/chunk_16x16/model.onnx",
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "--lod-type",
        default="lod1to0",
        choices=list(PROGRESSIVE_LOD_CONTRACTS.keys()) + ["legacy"],
        help="LOD contract to verify against",
    )
    parser.add_argument(
        "--print-contract",
        action="store_true",
        help="Print the contract definition and exit",
    )
    args = parser.parse_args()

    if args.print_contract:
        lod_type = args.lod_type
        if lod_type == "legacy":
            # Use the original contract for legacy mode
            print(f"=== Contract for {lod_type} ===")
            import json

            legacy_contract = {
                "inputs": {
                    "x_parent": [1, 1, 8, 8, 8],
                    "x_biome": [1, "NB", 16, 16, 1],
                    "x_height": [1, 1, 16, 16, 1],
                    "x_lod": [1, 1],
                },
                "outputs": {
                    "air_mask": [1, 1, 16, 16, 16],
                    "block_logits": [1, 1104, 16, 16, 16],
                },
            }
            print(json.dumps(legacy_contract, indent=2))
            sys.exit(0)
        elif lod_type in PROGRESSIVE_LOD_CONTRACTS:
            print(f"=== Contract for {lod_type} ===")
            import json

            print(json.dumps(PROGRESSIVE_LOD_CONTRACTS[lod_type], indent=2))
            sys.exit(0)
        else:
            print(f"FAIL: Unknown LOD type '{lod_type}'")
            available = list(PROGRESSIVE_LOD_CONTRACTS.keys()) + ["legacy"]
            print(f"Available: {available}")
            sys.exit(1)

    sys.exit(main(args.model_path, args.lod_type))

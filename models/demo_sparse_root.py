"""Small demo to validate the SparseRootModel forward pass."""

import sys
import pathlib
import torch

# Ensure repo root is on sys.path so the demo can be run as a script
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from LODiffusion.models import SparseRootModel


def main():
    B = 2
    N_root = 1
    in_ch = 16

    # synthetic root features
    root = torch.randn(B, N_root, in_ch)

    model = SparseRootModel(in_channels=in_ch, hidden=64, num_classes=20, levels=5)
    out = model(root)

    for lvl in sorted(out.keys(), reverse=True):
        split = out[lvl]["split"]
        label = out[lvl]["label"]
        print(f"{lvl}: split {tuple(split.shape)}, label {tuple(label.shape)}")


if __name__ == "__main__":
    main()

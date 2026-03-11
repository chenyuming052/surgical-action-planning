#!/usr/bin/env python3
from __future__ import annotations

"""Extract frozen frame features to HDF5.

Supported:
- dinov3_vitb16 -> 768-d CLS token
- lemonfm -> 1536-d global average pooled features
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--dataset', required=True, choices=['cholec80', 'cholect50', 'endoscapes'])
    parser.add_argument('--backbone', required=True, choices=['dinov3_vitb16', 'lemonfm', 'dinov2_vitb14', 'resnet50'])
    parser.add_argument('--out-h5', required=True)
    args = parser.parse_args()
    raise SystemExit('TODO: implement extract_features.py')


if __name__ == '__main__':
    main()

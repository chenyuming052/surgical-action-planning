#!/usr/bin/env python3
from __future__ import annotations

"""Build triplet groups via hybrid co-occurrence + semantic clustering.

Uses CholecT50 triplet annotations to compute co-occurrence matrix,
optionally combines with semantic embeddings, and produces G clusters.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description='Build triplet groups')
    parser.add_argument('--registry', required=True, help='Path to registry.json')
    parser.add_argument('--npz-root', required=True, help='Root directory of NPZ label files')
    parser.add_argument('--triplet-names', required=True, help='Path to triplet name list')
    parser.add_argument('--n-clusters', type=int, default=18, help='Number of triplet groups')
    parser.add_argument('--alpha', type=float, default=0.7, help='Weight for co-occurrence vs semantic similarity')
    parser.add_argument('--out-dir', required=True, help='Output directory for clustering artifacts')
    args = parser.parse_args()
    raise SystemExit('TODO: implement build_triplet_groups.py')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
from __future__ import annotations

"""Copy-Current baseline: predict current state persists.

Evaluates the naive baseline that predicts no change will occur,
i.e., the current instrument set and phase will persist at all horizons.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description='Copy-Current baseline evaluation')
    parser.add_argument('--registry', required=True, help='Path to registry.json')
    parser.add_argument('--npz-root', required=True, help='Root directory of NPZ label files')
    parser.add_argument('--split', default='test', help='Split to evaluate on')
    parser.add_argument('--config-eval', required=True, help='Path to eval config YAML')
    parser.add_argument('--out-dir', required=True, help='Output directory for results')
    args = parser.parse_args()
    raise SystemExit('TODO: implement copy_current.py')


if __name__ == '__main__':
    main()

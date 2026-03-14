#!/usr/bin/env python3
from __future__ import annotations

"""MML-SurgAdapt-style baseline: multi-modal multi-task learning baseline.

Reimplements the MML-SurgAdapt approach adapted to the SurgCast
multi-source training setup for fair comparison.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description='MML-SurgAdapt-style baseline')
    parser.add_argument('--config-data', required=True)
    parser.add_argument('--config-model', required=True)
    parser.add_argument('--config-train', required=True)
    parser.add_argument('--config-eval', required=True)
    parser.add_argument('--registry', required=True, help='Path to registry.json')
    parser.add_argument('--features-root', required=True, help='Root directory of feature HDF5s')
    parser.add_argument('--npz-root', required=True, help='Root directory of NPZ label files')
    parser.add_argument('--run-name', required=True, help='Experiment run name')
    parser.add_argument('--out-dir', required=True, help='Output directory')
    args = parser.parse_args()
    raise SystemExit('TODO: implement mml_surgadapt_style.py')


if __name__ == '__main__':
    main()

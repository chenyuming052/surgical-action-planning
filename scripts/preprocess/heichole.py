#!/usr/bin/env python3
from __future__ import annotations

"""Preprocess HeiChole dataset for external validation.

Parses HeiChole annotations (phase, instrument, action, skill)
and converts to NPZ format matching the dataset contract.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description='Preprocess HeiChole dataset')
    parser.add_argument('--root', required=True, help='Path to raw HeiChole data')
    parser.add_argument('--registry', required=True, help='Path to registry.json')
    parser.add_argument('--out-dir', required=True, help='Output directory for NPZ files')
    parser.add_argument('--ontology-map', default='', help='Path to ontology mapping YAML')
    args = parser.parse_args()
    raise SystemExit('TODO: implement preprocess_heichole.py')


if __name__ == '__main__':
    main()

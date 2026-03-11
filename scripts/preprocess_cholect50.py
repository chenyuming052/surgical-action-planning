#!/usr/bin/env python3
from __future__ import annotations

"""Preprocess CholecT50.

Outputs per-video npz:
- triplets / instruments / verbs / targets / phase
- triplet_groups
- dual TTC targets and censoring flags
- clipping event labels
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--group-def', default='')
    args = parser.parse_args()
    raise SystemExit('TODO: implement preprocess_cholect50.py')


if __name__ == '__main__':
    main()

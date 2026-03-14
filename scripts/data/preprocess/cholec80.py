#!/usr/bin/env python3
from __future__ import annotations

"""Preprocess Cholec80.

Key constraints:
- align 1 fps frames to dense phase annotations via original frame IDs
- fix 1-indexed image names vs 0-indexed annotation IDs
- map 7-tool presence to 6-class instrument labels (drop specimen bag)
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    raise SystemExit('TODO: implement preprocess_cholec80.py')


if __name__ == '__main__':
    main()

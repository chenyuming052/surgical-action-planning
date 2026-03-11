#!/usr/bin/env python3
from __future__ import annotations

"""Preprocess Endoscapes.

Must-do:
- read all_metadata.csv for CVS scores
- repair test coverage using filesystem list rather than test annotation_coco_vid.json
- extract anatomy_presence(5) and anatomy_obs_mask(5)
- parse scientific notation vids.txt entries
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    raise SystemExit('TODO: implement preprocess_endoscapes.py')


if __name__ == '__main__':
    main()

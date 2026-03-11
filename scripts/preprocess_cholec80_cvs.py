#!/usr/bin/env python3
from __future__ import annotations

"""Preprocess Cholec80-CVS from raw XLSX.

Must-do:
- direct XLSX parsing to full pre-clip/cut 1fps labels
- do NOT use official 85% truncation + 5fps pipeline
- drop 3 malformed intervals with final < initial
- truncate 63 out-of-bound intervals to first clipping boundary
- preserve ordinal 0/1/2 scores for three CVS criteria
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlsx', required=True)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--cholec80-npz-dir', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    raise SystemExit('TODO: implement preprocess_cholec80_cvs.py')


if __name__ == '__main__':
    main()

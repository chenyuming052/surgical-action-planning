#!/usr/bin/env python3
from __future__ import annotations

"""Evaluate one checkpoint on tiered benchmarks.

Tier 1: core action-change on CholecT50-test
Tier 2a: CVS safety on Endoscapes-test + Cholec80-test
Tier 2b: CVS-at-clipping on G2 test subset
Tier 3: phase
Tier 4: instrument
Tier 5: anatomy
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config-eval', required=True)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--features-root', required=True)
    parser.add_argument('--npz-root', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    raise SystemExit('TODO: implement evaluate.py')


if __name__ == '__main__':
    main()

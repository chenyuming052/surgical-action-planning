#!/usr/bin/env python3
from __future__ import annotations

"""Build static prior and evidence weights.

Only use training split videos.
Outputs:
- static_prior.pkl
- evidence_weights.pkl
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--npz-root', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    raise SystemExit('TODO: implement build_priors.py')


if __name__ == '__main__':
    main()

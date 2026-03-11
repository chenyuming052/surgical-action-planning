#!/usr/bin/env python3
from __future__ import annotations

"""Train baseline or full SurgCast.

Example stages:
- cholec_only
- plus_phase
- plus_tool_presence
- plus_cvs
- plus_endoscapes
- plus_masking
- plus_transition
- plus_static_prior
- full
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-data', required=True)
    parser.add_argument('--config-model', required=True)
    parser.add_argument('--config-train', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--registry', required=True)
    parser.add_argument('--features-root', required=True)
    parser.add_argument('--npz-root', required=True)
    parser.add_argument('--priors-dir', default='')
    parser.add_argument('--run-name', required=True)
    args = parser.parse_args()
    raise SystemExit('TODO: implement train.py')


if __name__ == '__main__':
    main()

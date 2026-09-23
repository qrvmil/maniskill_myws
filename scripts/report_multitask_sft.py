#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from maniskill_myws.pld.multitask_data import WORK,ROOT
from maniskill_myws.pld.multitask_report import export
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--allow-incomplete-trajectory',action='store_true',help='Publish completed N50 diagnostics only; final cells remain mandatory')
args=parser.parse_args()
export(WORK/'eval',ROOT/'docs/multitask_sft',allow_incomplete_trajectory=args.allow_incomplete_trajectory)

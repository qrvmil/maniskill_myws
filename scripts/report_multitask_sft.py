#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from maniskill_myws.pld.multitask_data import WORK,ROOT
from maniskill_myws.pld.multitask_report import export
export(WORK/'eval',ROOT/'docs/multitask_sft')

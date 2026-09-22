#!/usr/bin/env python3
"""Run using third_party/openpi/.venv/bin/python; see notebook for interactive API."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from maniskill_myws.pld.multitask_eval import main,build_parser
from maniskill_myws.pld.multitask_parallel import dispatch
if __name__=='__main__':
    args=build_parser().parse_args()
    if not dispatch(args):main()

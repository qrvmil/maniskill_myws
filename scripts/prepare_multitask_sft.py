#!/usr/bin/env python3
import argparse
from maniskill_myws.pld.multitask_data import fetch,prepare
p=argparse.ArgumentParser();p.add_argument('mode',choices=['fetch','prepare']);p.add_argument('--variant',choices=['A','B','C'])
a=p.parse_args()
if a.mode=='fetch':fetch()
elif a.variant:prepare(a.variant)
else:
    for variant in ('A','B','C'):prepare(variant)

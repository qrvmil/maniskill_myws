#!/usr/bin/env python3
import argparse
from maniskill_myws.pld.multitask_data import WORK
from maniskill_myws.pld.multitask_train import train
from maniskill_myws.pld.libero_artifacts import RunArtifacts
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['A','B','C'],required=True);a=p.parse_args()
with RunArtifacts(WORK/'runtime'/('train_'+a.variant),vars(a)):
    train(a.variant)

import argparse
from pathlib import Path
from eval_utils import ROOT, load_policy, evaluate

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--protocol',choices=['reference','official'],default='reference')
    parser.add_argument('--episodes',type=int,default=50)
    parser.add_argument('--tasks',default='D0,D1,D2,H1,H2')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-videos',action='store_true')
    args=parser.parse_args()
    policy=load_policy()
    evaluate(policy,args.output,args.tasks.split(','),args.episodes,protocol=args.protocol,save_videos=not args.no_videos)

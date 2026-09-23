#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from maniskill_myws.pld.libero_summary import gather_transfer_results

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args();result=gather_transfer_results(args.runs)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for name in ['tasks','buckets']:
        rows=result[name]
        with (output/f'{name}.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader()
            for row in rows:writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    print(f"Wrote {len(result['tasks'])} task results to {output}")

if __name__=='__main__':main()

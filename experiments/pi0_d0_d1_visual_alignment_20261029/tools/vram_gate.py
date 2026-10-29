"""Fail closed on incomplete VRAM coverage, hardware errors or a failed process."""
from pathlib import Path
import json,re
MiB=1<<20

def validate(evidence):
 p=Path(evidence)
 if p.joinpath('cuda_memtest.exitcode').read_text().strip()!='0':raise ValueError('VRAM test did not exit successfully')
 log=p.joinpath('cuda_memtest.log').read_text()
 if 'ERROR' in log or 'main thread: Program exits' not in log:raise ValueError('VRAM test error or incomplete log')
 done={int(i) for i in re.findall(r'Test(\d+) finished in',log)}
 expected=set(range(9))|{10}
 if done!=expected:raise ValueError('Required pattern/address/stress tests incomplete: '+str(sorted(done)))
 before=p.joinpath('nvidia-smi-before.txt').read_text()
 fb=before.split('FB Memory Usage')[1]
 nvml=int(re.search(r'Total\s*:\s*(\d+) MiB',fb)[1]);reserved=int(re.search(r'Reserved\s*:\s*(\d+) MiB',fb)[1])
 cuda_total=int(re.search(r'global memory size=(\d+)',log)[1])
 allocated=int(re.search(r'Allocated (\d+) MB',log)[1])*MiB
 if nvml!=65536:raise ValueError('Detected NVML memory differs from requested 65536MiB')
 if cuda_total/MiB<nvml-reserved-2:raise ValueError('CUDA memory is smaller than NVML minus driver reservation')
 # Allow CUDA context allocation, but never silently test only a small memory subset.
 if cuda_total-allocated>512*MiB:raise ValueError('Unexplained VRAM allocation coverage shortfall')
 free=re.search(r'driver reported at least (\d+) bytes are free',log)
 if free and int(free[1])-allocated>4*MiB:raise ValueError('More than4MiB of reported free VRAM not covered')
 after=p.joinpath('nvidia-smi-after.txt').read_text()
 ecc=[int(i) for i in re.findall(r'(?:SRAM Correctable|SRAM Uncorrectable Parity|SRAM Uncorrectable SEC-DED|DRAM Correctable|DRAM Uncorrectable)\s*:\s*(\d+)',after)]
 if len(ecc)<10 or any(ecc):raise ValueError('ECC counters are missing or nonzero')
 stress=p.joinpath('vram-stress64.log').read_text()
 if p.joinpath('vram-stress64.exitcode').read_text().strip()!='0' or 'ERROR' in stress or 'Test10 finished in' not in stress or 'main thread: Program exits' not in stress:raise ValueError('Corrected64bit full-volume stress incomplete or failed')
 stressed=int(re.search(r'Allocated (\d+) MB',stress)[1])*MiB
 if stressed<allocated-4*MiB:raise ValueError('Corrected stress did not cover the detected volume')
 tool=json.loads(p.joinpath('stress64-tool.json').read_text())
 if tool['memsize_bits']!=64 or tool['iterations']!=1000:raise ValueError('Full stress tool has unverified argument width or iteration budget')
 final=p.joinpath('nvidia-smi-after-stress64.txt').read_text()
 final_ecc=[int(i) for i in re.findall(r'(?:SRAM Correctable|SRAM Uncorrectable Parity|SRAM Uncorrectable SEC-DED|DRAM Correctable|DRAM Uncorrectable)\s*:\s*(\d+)',final)]
 if len(final_ecc)<10 or any(final_ecc):raise ValueError('ECC error after corrected64bit stress')
 return dict(passed=True,nvml_total_MiB=nvml,driver_reserved_MiB=reserved,cuda_total_bytes=cuda_total,
  stress64=dict(tested_bytes=stressed,iterations=1000,memsize_bits=64,ecc_counters=final_ecc,tool=tool),
  upstream_test10_limitation='int32 memsize truncated large allocations; superseded by the corrected64bit full-volume supplemental stress',
  tested_bytes=allocated,tested_MiB=allocated//MiB,cuda_context_and_unallocated_MiB=(cuda_total-allocated)/MiB,
  completed_tests=sorted(done),ecc_counters=ecc,bit_fade_test9='upstream default disabled; not performed',
  limitations='Tests all allocatable detected VRAM; driver/context-reserved cells cannot be allocated. Does not prove indefinite stability or the physical origin of unlocked capacity.')

if __name__=='__main__':
 import sys
 p=Path(sys.argv[1]);result=validate(p);p.joinpath('vram-gate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

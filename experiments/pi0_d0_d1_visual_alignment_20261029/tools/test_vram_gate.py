from pathlib import Path
import json
import pytest
from vram_gate import validate

def inputs(tmp_path):
 p=tmp_path
 p.joinpath('cuda_memtest.exitcode').write_text('0\n')
 p.joinpath('cuda_memtest.log').write_text('global memory size=68070735872\nAllocated 64606 MB\n'+''.join(f'Test{i} finished in 1.0 seconds\n' for i in [0,1,2,3,4,5,6,7,8,10])+'main thread: Program exits\n')
 p.joinpath('stress64-tool.json').write_text(json.dumps(dict(memsize_bits=64,iterations=1000)))
 p.joinpath('vram-stress64.exitcode').write_text('0\n')
 p.joinpath('vram-stress64.log').write_text('Allocated 64606 MB\nTest10 finished in 100.0 seconds\nmain thread: Program exits\n')
 p.joinpath('nvidia-smi-before.txt').write_text('FB Memory Usage\n Total : 65536 MiB\n Reserved : 619 MiB\n')
 p.joinpath('nvidia-smi-after.txt').write_text('\n'.join(['DRAM Correctable : 0','DRAM Uncorrectable : 0','SRAM Correctable : 0','SRAM Uncorrectable Parity : 0','SRAM Uncorrectable SEC-DED : 0']*2))
 p.joinpath('nvidia-smi-after-stress64.txt').write_text(p.joinpath('nvidia-smi-after.txt').read_text())
 return p

def test_complete_detected_volume_passes_with_driver_reservation(tmp_path):
 assert validate(inputs(tmp_path))['passed']

@pytest.mark.parametrize('case',['incomplete','nonzero_exit','memory_mismatch','allocation_shortfall','pattern_error','ecc_error','stress64_missing','stress64_short','stress64_error','stress64_ecc'])
def test_unsafe_or_incomplete_test_keeps_gate_closed(tmp_path,case):
 p=inputs(tmp_path)
 if case=='incomplete':p.joinpath('cuda_memtest.exitcode').unlink()
 if case=='nonzero_exit':p.joinpath('cuda_memtest.exitcode').write_text('1')
 if case=='memory_mismatch':p.joinpath('cuda_memtest.log').write_text(p.joinpath('cuda_memtest.log').read_text().replace('68070735872','8589934592'))
 if case=='allocation_shortfall':p.joinpath('cuda_memtest.log').write_text(p.joinpath('cuda_memtest.log').read_text().replace('64606 MB','8192 MB'))
 if case=='pattern_error':p.joinpath('cuda_memtest.log').write_text(p.joinpath('cuda_memtest.log').read_text()+'ERROR: data mismatch\n')
 if case=='ecc_error':p.joinpath('nvidia-smi-after.txt').write_text(p.joinpath('nvidia-smi-after.txt').read_text().replace('DRAM Correctable : 0','DRAM Correctable : 1'))
 if case=='stress64_missing':p.joinpath('vram-stress64.exitcode').unlink()
 if case=='stress64_short':p.joinpath('vram-stress64.log').write_text(p.joinpath('vram-stress64.log').read_text().replace('64606 MB','8192 MB'))
 if case=='stress64_error':p.joinpath('vram-stress64.log').write_text(p.joinpath('vram-stress64.log').read_text()+'ERROR: mismatch\n')
 if case=='stress64_ecc':p.joinpath('nvidia-smi-after-stress64.txt').write_text(p.joinpath('nvidia-smi-after-stress64.txt').read_text().replace('DRAM Correctable : 0','DRAM Correctable : 1'))
 with pytest.raises((ValueError,FileNotFoundError)):validate(p)

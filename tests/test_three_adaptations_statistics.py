"""Known paired outcomes, N50 cells, matched noise, and idempotent episode identities."""
import copy,hashlib,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/adaptation'))
from report import analyze

def corpus():
 rows=[]
 for method in ('full','expert','residual'):
  for task in ('D0','D1'):
   for index in range(50):
    success=index<25 if method=='full' else 10<=index<45 if method=='expert' else index%2==0
    length=100 if success else 220;seed=30000+index
    h=lambda s:hashlib.sha256(s.encode()).hexdigest()
    rows.append(dict(method=method,task=task,seed=seed,success=success,length=length,
       reset_hash=h(f'{task}:{seed}'),initial_observation_hash=h(f'obs:{task}:{seed}'),flow_noise_hashes=[h(f'{seed}:{i}') for i in range((length+4)//5)]))
 return rows

def test_known_success_rates_and_paired_exact_test():
 cells,pairs=analyze(corpus())
 assert len(cells)==6 and all(x['n']==50 for x in cells.values())
 assert cells['full_D0']['successes']==25 and cells['expert_D1']['successes']==35
 assert cells['full_D0']['wilson_95ci']==pytest.approx([.366445143168,.633554856832])
 result=next(p for p in pairs if (p['task'],p['before'],p['after'])==('D0','full','expert'))
 assert (result['n'],result['rescue'],result['harm'],result['gain_pp'])==(50,20,10,20.)
 assert result['mcnemar_exact_p']==pytest.approx(.098737146705)
 assert all(p['mcnemar_holm_p']>=p['mcnemar_exact_p'] for p in pairs)

@pytest.mark.parametrize('damage',['duplicate','reset','observation','noise','short_failure'])
def test_invalid_paired_dataset_is_rejected(damage):
 rows=corpus()
 if damage=='duplicate':rows[1]=copy.deepcopy(rows[0])
 elif damage=='reset':rows[100]['reset_hash']='other'
 elif damage=='observation':rows[100]['initial_observation_hash']='other'
 elif damage=='noise':rows[100]['flow_noise_hashes'][0]='other'
 else:
  row=next(r for r in rows if not r['success']);row['length']=100;row['flow_noise_hashes']=row['flow_noise_hashes'][:20]
 with pytest.raises(AssertionError):analyze(rows)

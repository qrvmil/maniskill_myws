"""Episode-level exports, paired statistics, completeness gate and concise Russian report."""
import csv,itertools
from common import *
METHODS=['full','expert','residual']

def load_rows():
 return [json.loads(p.read_text()) for p in sorted((WORK/'final').glob('*/*/*/episode.json'))]

def interim():
 rows=load_rows();dest=ROOT/'experiments/pi0_d0_d1_three_adaptations/results';dest.mkdir(parents=True,exist_ok=True)
 compact=[{k:r[k] for k in ('method','task','seed','success','length','reset_hash','trajectory_hash','video','video_sha256','checkpoint','normalization_sha256')} for r in rows]
 write(dest/'episodes.json',compact)
 with (dest/'episodes.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(compact[0]) if compact else ['method','task','seed','success']);w.writeheader();w.writerows(compact)
 return rows

def main():
 from scipy.stats import binomtest
 from maniskill_myws.pld.multitask_protocol import summarize,paired_change
 from evaluate import verified_episode
 from storage import publish,push_progress
 rows=interim();selection=json.loads((WORK/'final_selection.json').read_text());cells={}
 assert len(rows)==300 and len({(r['method'],r['task'],r['seed']) for r in rows})==300
 for method in METHODS:
  for task in TASKS:
   cell=[r for r in rows if (r['method'],r['task'])==(method,task)]
   assert sorted(r['seed'] for r in cell)==list(range(30000,30050))
   assert len({r['reset_hash'] for r in cell})==50,'Repeated initial scenes within a task'
   for r in cell:
    verified_episode(WORK/'final'/method/task/str(r['seed']),method,task,r['seed'],selection[method]['checkpoint'])
   cells[method+'_'+task]=summarize(cell)
 for task in TASKS:
  for seed in range(30000,30050):
   triple=[r for r in rows if (r['task'],r['seed'])==(task,seed)]
   assert len({r['reset_hash'] for r in triple})==1
   common=min(len(r['flow_noise_hashes']) for r in triple)
   assert all(r['flow_noise_hashes'][:common]==triple[0]['flow_noise_hashes'][:common] for r in triple)
 pairs=[]
 for task in TASKS:
  for a,b in itertools.combinations(METHODS,2):
   result=paired_change([r for r in rows if r['method']==a and r['task']==task],[r for r in rows if r['method']==b and r['task']==task])
   n=result['rescue']+result['harm']
   result.update(task=task,before=a,after=b,mcnemar_exact_p=binomtest(result['rescue'],n,.5).pvalue if n else 1.)
   pairs.append(result)
 order=sorted(range(len(pairs)),key=lambda i:pairs[i]['mcnemar_exact_p']);prev=0
 for rank,i in enumerate(order):
  prev=max(prev,min(1.,(len(pairs)-rank)*pairs[i]['mcnemar_exact_p']));pairs[i]['mcnemar_holm_p']=prev
 dest=ROOT/'experiments/pi0_d0_d1_three_adaptations/results'
 write(dest/'success_rates.json',cells);write(dest/'paired_comparisons.json',pairs)
 write(dest/'completeness.json',dict(passed=True,episodes=300,videos=300,cells=6,n_per_cell=50,terminal_frames_verified=True,reset_hashes_equal=True,common_flow_noise_prefixes_equal=True))
 with (dest/'success_rates.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=['method','task','n','successes','success_rate','wilson_low','wilson_high']);w.writeheader()
  for method in METHODS:
   for task in TASKS:
    s=cells[method+'_'+task];w.writerow(dict(method=method,task=task,**{k:s[k] for k in ('n','successes','success_rate')},wilson_low=s['wilson_95ci'][0],wilson_high=s['wilson_95ci'][1]))
 def fmt(method,task):
  s=cells[method+'_'+task];lo,hi=s['wilson_95ci'];return f"{s['successes']}/50 — {100*s['success_rate']:.0f}% [{100*lo:.1f}; {100*hi:.1f}]"
 table='\n'.join('| '+label+' | '+fmt(m,'D0')+' | '+fmt(m,'D1')+' |' for m,label in [('full','Full SFT'),('expert','Action-expert SFT'),('residual','Residual RL')])
 best=max(METHODS,key=lambda m:cells[m+'_D1']['success_rate']);retain=max(METHODS,key=lambda m:cells[m+'_D0']['success_rate'])
 offline=json.loads((WORK/'offline/complete.json').read_text());teacher=json.loads((WORK/'teacher.json').read_text());rl=json.loads((WORK/'evidence/residual_complete.json').read_text())
 report=f'''# Три способа адаптации π₀: D0 → D1

Это сравнение освоения D1 и сохранения D0 на двух **знакомых задачах LIBERO**. D0 — миска в центре стола; D1 — миска рядом с тарелкой. Во всех SFT использованы все 50 исходных native demonstrations соответствующей задачи. Финальная оценка — новые generated scenes этих задач; это не held-out demonstrations и не OOD.

## Что обучали

Один общий full-SFT checkpoint: официальный `pi0_base`, **3000 updates × batch8 на D0**, без LoRA. Его D0-only normalization одинакова во всех методах и проверена по SHA256. Из одних и тех же D0 weights независимо:

- **Full SFT D1:** 1500 updates × batch8, все параметры; новый AdamW.
- **Action-expert SFT D1:** те же 1500 updates, данные, seed и LR schedule; обучались action expert и state/action/time projections. PaliGemma, vision encoder и multimodal projector frozen; trainable scope, gradients и hashes проверены.
- **Residual RL D1:** исходная D0 π₀ полностью frozen. Отдельные actor/critic/temperature; corrected V3 SAC и auxiliary full-action Cal-QL, V4 разделение ролей D0/D1. Offline initialization: 1000 updates на успешных native D1 demos ({offline['accepted_demonstrations']}/50 прошли state-based LIBERO goal check). База/replay пересчитаны. Затем 20 teacher-warmup episodes с обновлениями actor/critic/temperature и ровно 50000 active online transitions, UTD1, batch256, offline fraction .5. Всего online environment steps: {rl['environment_steps']}.

Два изменения V4 заданы **до результатов**: warmup исполняет ранний full-SFT teacher (D1 update {teacher['step']}, validation {teacher['validation']['successes']}/10), а reward содержит potential-based shaping (gamma .99, beta .1): reach нужной миски, grasp, lift/transport. Teacher не заменяет D0-базу; исполненные teacher actions и cached D0 actions записаны отдельно. Terminal Phi=0; offline/online rewards и MC returns вычисляются одним кодом. Успех всегда определяется исходным LIBERO predicate. Online OTF, probe_fraction0; evaluation deterministic actor.

Действие residual: `clip(a_D0 + 2 * residual, -1, 1)` в OSC units. xi2 обеспечивает достижимость любых bounded teacher actions, включая gripper. Это широкая correction authority, поэтому результат нельзя трактовать как проверку только малых поправок.

## Результаты

N=50 в каждой ячейке; в скобках Wilson 95% CI. Horizon220, replan5. Методы получают одинаковые сохранённые generated reset states и flow noise; hashes проверены. Последние checkpoints выбраны по заранее фиксированному бюджету; final seeds не участвовали в выборе.

| Метод | D0 SR [95% CI] | D1 SR [95% CI] |
|---|---|---|
{table}

На этом запуске наибольший измеренный D1 SR у `{best}`, наибольший D0 SR у `{retain}`. Это описательное сравнение: различия и неопределённость нужно читать вместе с [paired comparisons](results/paired_comparisons.json), где есть gained/lost successes, paired bootstrap CI, exact McNemar и Holm correction шести сравнений. [Episode CSV](results/episodes.csv) позволяет проверить каждую пару.

## Как читать выводы

SFT методы сравнивают область обновляемых параметров при одинаковом D1 бюджете. RL имеет другой объём online опыта и дополнительную информацию simulator state в shaping, поэтому это сравнение трёх рецептов адаптации, а не изоляция единственного фактора. Native D1 images/proprioception/actions в offline replay сохранены; shaping geometry и goal checks восстановлены из original XML и saved states. Offline MC labels у online эпизода доступны после его завершения; промежуточные SAC updates используют TD rewards.

Одна training seed на метод и 50 correlated scenes ограничивают точность и переносимость выводов. Teacher validation мала; наличие ≥1 успеха подтверждает ненулевой измеренный SR, а не надёжность teacher. D0/D1 различаются и target location, и distractor placement. Generated reset protocol отличается от официальных benchmark init files. Потенциал использует simulator state, недоступный обычному vision-only deployment; actor при исполнении получает только камеры, proprioception и действие базы.

## Материалы и воспроизводимость

- [Общий D0 checkpoint, teacher и три финальные модели с normalization](https://huggingface.co/{HF_REPO}/tree/main/checkpoints).
- [Все 300 двухкамерных видео с episode JSON](https://huggingface.co/{HF_REPO}/tree/main/final), terminal frame включён.
- [Generated reset states](https://huggingface.co/{HF_REPO}/tree/main/reset_states).
- [Выполненный учебный notebook](seminar.ipynb): weights/norm → scene → observations → action → проверенный rollout → video → SR; повторное чтение не добавляет episodes.
- [Протокол](PREREGISTRATION.md), [completeness check](results/completeness.json), [SR CSV](results/success_rates.csv).

Код/прогресс сохранены в GitHub; веса, optimizer/RNG/replay и артефакты — в HF с обратным чтением. Видимость существующих repositories не менялась.
'''
 (dest.parent/'REPORT.md').write_text(report)
 publish(dest,'results',full_readback=True)
 push_progress('exp: publish all 300 verified episodes, confidence intervals and paired comparisons')
 print('ALL_300_EPISODES_AND_VIDEOS_VERIFIED',flush=True)
if __name__=='__main__':main()

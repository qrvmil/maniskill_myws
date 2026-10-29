"""Report only actual new measurements; all historical outcomes are labelled."""
import json
from pathlib import Path
from common import WORK,EXPERIMENT,HF_REPO,sha,write
BASE=WORK.parent
BRANCH='exp/pi0-d0-d1-visual-alignment-20261029'
GITHUB='https://github.com/qrvmil/maniskill_myws/tree/'+BRANCH
HF='https://huggingface.co/'+HF_REPO

def render(cells,pairs,selection):
 data=json.loads((BASE/'evidence/data-preflight.json').read_text())
 smoke=json.loads((BASE/'evidence/model-smoke.json').read_text())
 gate=json.loads((BASE/'evidence/vram-gate.json').read_text())
 decision=json.loads((WORK/'alignment/lambda_decision.json').read_text())
 residual=json.loads((WORK/'rl/final/training.json').read_text())
 audit=json.loads((WORK/'evidence/online_audit_final.json').read_text());assert audit['passed'] and audit['active_transitions']==50000
 sft={s:json.loads((WORK/'sft'/s/str(n)/'training.json').read_text()) for s,n in [('d0',3000),('full',1500),('expert',1500)]}
 lines=['# Три адаптации π0 от общей D0-базы с visual alignment','',
 'Это новый последовательный D0→D1 запуск на CMP170HX. Сравниваются три рецепта адаптации общей aligned D0-базы на знакомых задачах LIBERO; новые generated scenes не являются OOD-задачами. Последние checkpoints выбраны по фиксированным бюджетам до открытия final seeds.', '',
 '| Метод | D0 SR, Wilson95% CI | D1 SR, Wilson95% CI |','|---|---|---|']
 labels={'full':'Full SFT','expert':'Action-expert SFT','residual':'Residual RL'}
 for method in ['full','expert','residual']:
  values=[]
  for task in ['D0','D1']:
   row=cells[method+'_'+task];assert row['n']==50
   low,high=row['wilson_95ci'];values.append(f"{row['successes']}/50 — {100*row['success_rate']:.0f}% [{100*low:.1f}; {100*high:.1f}]")
  lines.append('| '+labels[method]+' | '+' | '.join(values)+' |')
 lines+=['',
  f"Данные: все50native demonstrations каждой задачи, D0={data['D0']['aligned_frames']} и D1={data['D1']['aligned_frames']} aligned frames; revision f13aa24a3da8c43c7225569f28c562979fa0e35a. Сохранены obs[i]→action[i+1], обе камеры rotate180°, official resize/pad,50-action chunks и исходная D0-only normalization (SHA256 `{data['normalization_sha256']}`). D1 не участвовала в калибровке, pilot, normalization или обучении общей D0-базы.", '',
  f"Общая база: тот же официальный pi0_base, точный SHA256 исходных fp32 weights `{smoke['initial_weights_sha256']}`, затем fullSFT3000updates на D0, seed0. Из одной полученной базы независимо full/expertD1 по1500updates, seed1. Effective batch8=microbatch2×accum4; fp32 master/AdamW states,bf16 compute, gradient checkpointing; исходный AdamW(.9,.95,eps1e-8,wd1e-10), clip1, cosine3e-5→3e-6,warmup100. Full trainable={sft['full']['trainable_parameters']:,}; expert={sft['expert']['trainable_parameters']:,}; frozen scope hashes проверены.", '',
  f"Перенесён механизм visual alignment: frozen C-RADIOv3-L@9d0413465e8a91e67bbf2c1ad342815478d1b906, spatial cosine loss по visual prefix блока9 PaliGemma, две валидные камеры, grid14×14→16×16bilinear. LN+MLP2048→1024→1024 калиброван200updates только на D0 и заморожен. D0-only20update gradient-ratio pilot отброшен; выбран λ={decision['selected']:.6g}, затем зафиксирован для D0 и D1full. K4deterministic paired crops/brightness/contrast заменяют extra random train-time preprocessing; student/teacher RGB hashes проверяются. Бюджеты, модели и normalization из совместного JAX/LoRA D0+D1 эксперимента не импортировались.", '',
  'У fullD1 alignment продолжает обновлять PaliGemma. Action-expert сохраняет исходную заморозку PaliGemma/vision/projector, residual сохраняет всю π0 frozen; оба наследуют aligned D0-базу. Для этих frozen scopes дополнительный prefix loss не объявляется действующей регуляризацией; SERL encoder не получает новую alignment-цель.', '',
  f"Residual: исходный correctedV3SAC/CalQL+V4teacher warmup/shaping; offline1000updates,20warmup episodes ({audit['warmup_transitions']}transitions,{audit['warmup_successes']}/20successes), затем ровно50000active transitions; всего {residual['counters']['environment_steps']}environment steps. Batch256,UTD1,offline fraction.5,actor/temperature через2critic steps,xi2,gamma.99,beta.1,terminalPhi0. Teacher выбран на update{residual['teacher']['step']} по исходному правилу≥1/10validation successes; D0 остаётся отдельной frozen базой. Достижимость teacher actions, cache advancement и rewards/MC returns проверены.", '',
  'Оценка:3×2×50=300rollouts, seeds30000–30049,horizon220,replan5; общие сохранённые generated reset states, hashes начальных observations и общий prefix явного flow-noise stream. Residual actor deterministic. Проверены300двухкамерных MP4 с terminal frame, episodeJSON/CSV. Ни один final seed не использовался для выбора модели.', '',
  '| Задача | Сравнение (второй−первый) | ΔSR,п.п. [95%CI] | Gained/lost | McNemar p,Holm |','|---|---|---:|---:|---:|']
 for pair in pairs:
  boundary=pair['exact_boundary_95_pp'];interval=boundary if boundary is not None else pair['bootstrap_95_pp']
  low,high=interval
  lines.append(f"| {pair['task']} | {labels[pair['before']]} → {labels[pair['after']]} | {pair['gain_pp']:+.0f} [{low:+.1f}; {high:+.1f}]{'†' if boundary is not None else ''} | {pair['rescue']}/{pair['harm']} | {pair['mcnemar_holm_p']:.4g} |")
 significant=sum(p['mcnemar_holm_p']<.05 for p in pairs)
 lines+=['','Парные CI: bootstrap10000draws; † — conservative boundary interval при вырожденном bootstrap. Holm применяется ко всем6comparisons; статистически значимы '+str(significant)+' из6. SR отражает вероятность успеха на sampled scenes при фиксированной модели. Одна training seed на метод ограничивает переносимость; равные/незначимые различия не доказывают эквивалентность. RL использует дополнительный online опыт и simulator-state shaping.', '',
  'Исторический запуск без новой методики: fullD0=0/50,D1=20/50; expertD0=0/50,D1=19/50; residualD0=0/50,D1=0/50. Сравнение с этими числами описательное: меняются общая D0-база и аугментации. Изолированный причинный эффект самого alignment этим новым трёхсторонним запуском не оценивается. D0 SR после адаптации измерена; точная величина забывания относительно общей D0-базы не измерена отдельной базовой оценкой N=50.', '',
  f"Preflight: host150140,instance58231938,1×CMP170HX,x86_64; NVML65536MiB, CUDA-visible{gate['cuda_total_bytes']/(1<<30):.2f}GiB. Проверены все allocatable{gate['tested_MiB']}MiB тестами0–8и10,exit0,ECC0; driver-reserved{gate['driver_reserved_MiB']}MiB и CUDA context не являются allocatable test cells. UpstreamTest10 имел int32overflow; исправленный64bit full-volume stress1000iterations прошёл за141.3s,exit0,ECC0; source/binary/patch hashes сохранены. Optional3hour bit-fade9 отключён. JAX0.5.3/PyTorch2.7.1+cu128 GPU probes и3actual AdamW steps прошли; block9prefix relativeRMSE={smoke['middle_prefix_relative_rmse']:.6f}<.005, минимальный запас{smoke['minimum_headroom_bytes']/(1<<30):.2f}GiB. Это наблюдаемая проверка доступной памяти и стабильности, без утверждения о происхождении unlocked capacity или бесконечной безошибочности.", '',
  f"[Код/протокол/логи]({GITHUB}/experiments/pi0_d0_d1_visual_alignment_20261029), [HF checkpoints]({HF}/tree/main/checkpoints), [projector/pilot/cache metadata]({HF}/tree/main/alignment), [episode CSV и статистика]({HF}/tree/main/results), [все видео](VIDEOS.md), [preflight и runtime logs]({HF}/tree/main/preflight/evidence).", '',
  f"Общий D0: `{sha(WORK/'sft/d0/3000/model.safetensors')}`; teacher: `{sha(WORK/'teacher_policy/model.safetensors')}`.",'']
 for method in ['full','expert','residual']:lines.append(f"- {labels[method]}: `{selection[method]['weights_sha256']}` — {HF}/tree/main/checkpoints/{'residual/final' if method=='residual' else method+'/1500'}")
 return '\n'.join(lines)+'\n'

def publish_card():
 from huggingface_hub import HfApi,hf_hub_download
 report=(EXPERIMENT/'REPORT.md').read_text()
 text='---\nlanguage: [ru, en]\ntags: [openpi, pi0, robotics, libero, visual-alignment]\n---\n\n'+report
 path=WORK/'model_card/README.md';path.parent.mkdir(exist_ok=True);path.write_text(text)
 from huggingface_hub import CommitOperationAdd
 documents={'README.md':path,**{name:EXPERIMENT/name for name in ['REPORT.md','VIDEOS.md','PREREGISTRATION.md','SOURCE_AUDIT.md']}}
 commit=HfApi().create_commit(repo_id=HF_REPO,operations=[CommitOperationAdd(path_in_repo=name,path_or_fileobj=str(local)) for name,local in documents.items()],commit_message='Publish verified three-adaptation visual alignment results')
 for name,local in documents.items():
  downloaded=hf_hub_download(HF_REPO,name,revision=commit.oid);assert sha(downloaded)==sha(local)
 write(WORK/'evidence/hf_model_card.json',dict(commit=commit.oid,sha256=sha(path),completed=True,readback=True))

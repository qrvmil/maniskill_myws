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
 '| Метод | D0 SR, Wilson 95% CI | D1 SR, Wilson 95% CI |','|---|---|---|']
 labels={'full':'Full SFT','expert':'Action-expert SFT','residual':'Residual RL'}
 for method in ['full','expert','residual']:
  values=[]
  for task in ['D0','D1']:
   row=cells[method+'_'+task];assert row['n']==50
   low,high=row['wilson_95ci'];values.append(f"{row['successes']}/50 — {100*row['success_rate']:.0f}% [{100*low:.1f}; {100*high:.1f}]")
  lines.append('| '+labels[method]+' | '+' | '.join(values)+' |')
 lines+=['',
  f"Данные: все 50 исходных демонстраций каждой задачи, D0={data['D0']['aligned_frames']} и D1={data['D1']['aligned_frames']} согласованных пар наблюдение/действие; revision f13aa24a3da8c43c7225569f28c562979fa0e35a. Сохранены obs[i]→action[i+1], поворот изображений обеих камер на 180°, исходные resize/pad и последовательности из 50 действий и исходная нормализация только по D0 (SHA256 `{data['normalization_sha256']}`). D1 не участвовала в калибровке, pilot или нормализации или обучении общей D0-базы.", '',
  f"Общая база: тот же официальный pi0_base, точный SHA256 исходных fp32 weights `{smoke['initial_weights_sha256']}`, затем full SFT на D0: 3000 обновлений, seed 0. Из одной полученной базы независимо full/expert на D1: по 1500 обновлений, seed 1. Эффективный batch 8 = microbatch 2 × накопление 4; веса и состояния AdamW в fp32, вычисления в bf16, gradient checkpointing; исходный AdamW(.9,.95,eps1e-8,wd1e-10), clip 1, cosine 3e-5→3e-6, warmup 100. Обучаемых параметров full: {sft['full']['trainable_parameters']:,}; expert: {sft['expert']['trainable_parameters']:,}; хеши замороженных параметров проверены.", '',
  f"Перенесён механизм visual alignment: frozen C-RADIOv3-L@9d0413465e8a91e67bbf2c1ad342815478d1b906, spatial cosine loss по visual prefix блока 9 PaliGemma, две валидные камеры, сетка 14×14→16×16 с bilinear интерполяцией. LN+MLP2048→1024→1024 калиброван 200 обновлений только на D0 и заморожен. 20 обновлений pilot для оценки отношения норм градиентов проведены только на D0; их веса отброшены; выбран λ={decision['selected']:.6g}, затем зафиксирован для D0 и full D1. Четыре детерминированных варианта crops/brightness/contrast заменяют дополнительную случайную обработку изображений при обучении; student/teacher RGB hashes проверяются. Бюджеты, модели и normalization из совместного JAX/LoRA D0+D1 эксперимента не импортировались.", '',
  'У full D1 alignment продолжает обновлять PaliGemma. Action-expert сохраняет исходную заморозку PaliGemma/vision/projector, residual сохраняет всю π0 frozen; оба наследуют aligned D0-базу. Для этих frozen scopes дополнительный prefix loss не объявляется действующей регуляризацией; SERL encoder не получает новую alignment-цель.', '',
  f"Residual: исходный corrected V3 SAC/CalQL с V4 teacher warmup и shaping; 1000 offline обновлений, 20 warmup эпизодов ({audit['warmup_transitions']} переходов, {audit['warmup_successes']}/20 успехов), затем ровно 50000 активных переходов; всего {residual['counters']['environment_steps']} шагов среды. Batch 256, UTD 1, доля offline .5; actor/temperature через каждые 2 critic шага; xi 2, gamma .99, beta .1, terminal Phi 0. Teacher выбран на update {residual['teacher']['step']} по исходному правилу: не менее 1 успеха из 10 validation эпизодов; D0 остаётся отдельной frozen базой. Достижимость teacher actions, cache advancement и rewards/MC returns проверены.", '',
  'Оценка: 3×2×50 = 300 эпизодов, seeds 30000–30049, horizon 220, replan 5; общие сохранённые generated reset states, hashes начальных observations и общий prefix явного flow-noise stream. Residual actor deterministic. Проверены 300 двухкамерных MP4 с terminal frame, episodeJSON/CSV. Ни один final seed не использовался для выбора модели.', '',
  '| Задача | Сравнение (второй−первый) | ΔSR, п.п. [95%CI] | Gained/lost | McNemar p, Holm |','|---|---|---:|---:|---:|']
 for pair in pairs:
  boundary=pair['exact_boundary_95_pp'];interval=boundary if boundary is not None else pair['bootstrap_95_pp']
  low,high=interval
  lines.append(f"| {pair['task']} | {labels[pair['before']]} → {labels[pair['after']]} | {pair['gain_pp']:+.0f} [{low:+.1f}; {high:+.1f}]{'†' if boundary is not None else ''} | {pair['rescue']}/{pair['harm']} | {pair['mcnemar_holm_p']:.4g} |")
 significant=sum(p['mcnemar_holm_p']<.05 for p in pairs)
 lines+=['','Парные CI: bootstrap с 10000 выборок; † — conservative boundary interval при вырожденном bootstrap. Holm применяется ко всем 6 сравнениям; статистически значимы '+str(significant)+' из 6. SR отражает вероятность успеха на sampled scenes при фиксированной модели. Одна training seed на метод ограничивает переносимость; равные/незначимые различия не доказывают эквивалентность. RL использует дополнительный online опыт и simulator-state shaping.', '',
  'Исторический запуск без новой методики: full D0=0/50, D1=20/50; expert D0=0/50, D1=19/50; residual D0=0/50, D1=0/50. Сравнение с этими числами описательное: меняются общая D0-база и аугментации. Изолированный причинный эффект самого alignment этим новым трёхсторонним запуском не оценивается. D0 SR после адаптации измерена; точная величина забывания относительно общей D0-базы не измерена отдельной базовой оценкой N=50.', '',
  f"Preflight: host 150140, instance 58231938, 1× CMP170HX, x86_64; NVML 65536 MiB, CUDA-visible {gate['cuda_total_bytes']/(1<<30):.2f} GiB. Проверены все allocatable{gate['tested_MiB']} MiB тестами 0–8 и 10, exit 0, ECC 0; driver-reserved{gate['driver_reserved_MiB']} MiB и CUDA context не являются allocatable test cells. В upstream test 10 обнаружено переполнение int32; исправленный 64-bit stress всего объёма прошёл 1000 итераций за 141.3 s, exit 0, ECC 0; source/binary/patch hashes сохранены. Необязательный трёхчасовой bit-fade test 9 не запускался. GPU probes JAX 0.5.3/PyTorch 2.7.1+cu128 и 3 реальных шага AdamW прошли; block 9 prefix relative RMSE={smoke['middle_prefix_relative_rmse']:.6f}<.005, минимальный запас {smoke['minimum_headroom_bytes']/(1<<30):.2f} GiB. Это наблюдаемая проверка доступной памяти и стабильности, без утверждения о происхождении unlocked capacity или бесконечной безошибочности.", '',
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

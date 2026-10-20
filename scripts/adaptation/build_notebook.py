"""Build and execute the Russian seminar only after the 300-episode completion gate."""
from common import *
def main():
 import nbformat
 from nbclient import NotebookClient
 from jupyter_client.kernelspec import KernelSpecManager
 from storage import publish,push_progress
 complete=json.loads((ROOT/'experiments/pi0_d0_d1_three_adaptations/results/completeness.json').read_text());assert complete['passed']
 nb=nbformat.v4.new_notebook();cells=[]
 def md(s):cells.append(nbformat.v4.new_markdown_cell(s))
 def code(s):cells.append(nbformat.v4.new_code_cell(s))
 md('''# Семинар: три способа адаптации π₀ с D0 на D1

Сравним **full SFT**, **action-expert SFT** и **residual RL**. Вначале один раз обучена полная D0-модель из официального pi0_base; все методы стартуют из неё. Все 50 native demonstrations каждой задачи остаются в SFT. Новые generated evaluation scenes относятся к знакомым задачам, это не OOD.

Цель практики: проследить путь **weights + normalization → scene → observations → action → rollout → video → SR**, затем оценить сохранение D0 и освоение D1. Мы используем уже завершённые 300 финальных эпизодов. Повторный запуск проверяет и переиспользует их, не увеличивая N.

[Модели и артефакты](https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-20261020/tree/main), [код и preregistration](https://github.com/qrvmil/maniskill_myws/tree/exp/pi0-d0-d1-three-adaptations-20261020).

Среда: pinned OpenPI/LIBERO из репозитория; `scripts/adaptation/install_runtime.sh` фиксирует зависимости. На другой машине сначала скачайте checkpoint directories, normalization, final episodes и reset_states с HF в `/workspace/pi0_three` (или измените `WORK` в common.py), затем проверьте SHA256 inventories. Сохранённый notebook уже содержит выполненные outputs и три примера видео; все 300 видео доступны на HF.''')
 code("""import os, sys, json
from pathlib import Path
os.environ['MUJOCO_GL'] = 'egl'
os.environ['JAX_PLATFORMS'] = 'cpu'  # π₀ и residual используют PyTorch CUDA
repo = Path('/workspace/pi0_experiment')
sys.path[:0] = [str(repo/'scripts/adaptation'), str(repo/'src'), '/workspace/LIBERO']
from notebook_tools import SeminarPolicy
from common import WORK, ROOT, HF_REPO
selection = json.loads((WORK/'final_selection.json').read_text())
assert set(selection) == {'full', 'expert', 'residual'}
selection""")
 md('''## Данные, общий старт и честная оценка

D0: «pick up the black bowl from table center and place it on the plate»; D1: «pick up the black bowl next to the plate and place it on the plate». Native observations записаны после action; обучающая пара — `obs[i] → action[i+1]`. Камеры повёрнуты на 180°, далее официальный resize/pad. D0-only normalization фиксирована, inverse normalization применяется один раз перед OSC clipping.

3000 updates × batch8 на D0; затем два SFT по 1500 updates × batch8 с новыми optimizer states. Full обновляет всю π₀, expert — action expert и state/action/time projections, заморозив PaliGemma. Residual замораживает всю исходную D0 π₀.

Для каждой пары task/seed методы получают одинаковые сохранённые generated reset states и поток flow noise. Horizon220, replan5; финальные seeds30000–30049 не использовались в RL или validation. Наша демонстрация ниже читает seed30000 из этой фиксированной таблицы.''')
 for method,label in [('full','1. Full SFT'),('expert','2. Action-expert SFT'),('residual','3. Residual RL')]:
  md('## '+label+'\n\nЗагрузим веса и ту же D0 normalization. Создадим D1-сцену с сохранённым reset state и проверим hash. Затем вычислим первое действие и сверим его с финальным эпизодом.')
  code(f"policy = SeminarPolicy('{method}')\nprint('Checkpoint:', policy.checkpoint)\nprint('D0 normalization SHA256:', policy.norm)")
  code("observation = policy.scene('D1', seed=30000)\nprint('Prompt:', observation['prompt'])\nprint('State:', observation['state'])\nprint('Images:', observation['images'].shape, observation['images'].dtype)\nprint('Reset SHA256:', observation['reset_hash'])")
  code("import matplotlib.pyplot as plt\nfig, axes = plt.subplots(1, 2, figsize=(9, 4))\nfor ax, image, title in zip(axes, observation['images'], ['agentview', 'wrist']):\n    ax.imshow(image); ax.set_title(title); ax.axis('off')\nplt.show()")
  code("action = policy.action()\naction")
  md('''`executed_action` — семь bounded команд OSC: translation, rotation и gripper. Chunk длиной50 предсказывается заново каждые5 исполненных actions. Rollout ниже переиспользует проверенный финальный episode; видео включает обе камеры и terminal frame.''')
  code("rollout = policy.rollout()\n{k: rollout[k] for k in ['method', 'task', 'seed', 'success', 'length', 'reset_hash', 'trajectory_hash']}")
  code("policy.video(rollout)")
  code("policy.success_rates()")
  code("policy.close()\ndel policy")
  if method=='expert':md('''### Что именно заморожено?

В `sft/expert/scope.json` перечислены все tensors и trainable flags. PaliGemma (vision encoder, multimodal projector и language model) не входит в optimizer. На каждом сохранении hashes frozen weights сверяются с началом стадии; normalization тоже проверяется побайтово. Unused expert lm_head унаследован из официального PyTorch представления, не участвует в loss; это не адаптер LoRA.''')
  if method=='residual':md('''### Две модели во время warmup

В residual всегда остаётся исходная frozen D0-база. Ранний D1 full-SFT checkpoint только исполняет warmup actions. Два chunk caches независимы, D0 action вычисляется на каждом посещённом состоянии. `clip(a_D0 + 2 × residual, -1,1)` позволяет воспроизвести teacher action, что проверено в trace.

Offline Cal-QL:1000 updates; teacher warmup:20 episodes с обновлениями actor/critic/temperature; online:50000 active transitions, OTF collection, probe_fraction0. Evaluation использует deterministic actor.

Shaping: `r = r_success + .1 × (.99 × Phi(next) − Phi(state))`. Phi зависит от reach, grasp, lift/transport нужной миски. Terminal Phi равна0, включая конечный горизонт. Один код пересчитывает rewards и MC returns; исходный LIBERO success predicate сохраняется. State potential используется только при обучении; actor получает камеры, proprioception и base action.''')
 md('''## Итоговая таблица и paired comparisons

В каждой ячейке **N=50**. Wilson95% CI отражает неопределённость SR для сцен; одна training seed не измеряет вариативность самого обучения. Paired differences сопоставляют одинаковые scenes/noise, exact McNemar использует discordant pairs; Holm adjustment учитывает шесть сравнений. Не объединяем correlated episodes между методами в один размер выборки.''')
 code("import pandas as pd\nresults = ROOT/'experiments/pi0_d0_d1_three_adaptations/results'\npd.read_csv(results/'success_rates.csv')")
 code("pd.DataFrame(json.loads((results/'paired_comparisons.json').read_text()))")
 code("completeness = json.loads((results/'completeness.json').read_text())\nassert completeness['episodes'] == completeness['videos'] == 300\ncompleteness")
 md('''## Обсуждение для студентов

1. Каков tradeoff D1 acquisition и D0 retention? Какие paired differences убедительны с учётом CI?
2. Почему full/expert SFT имеют одинаковый бюджет, но RL нельзя считать тем же объёмом supervision?
3. Как широкая residual scale2 и simulator-state shaping ограничивают интерпретацию?
4. Почему эти результаты не являются OOD generalization и не измеряют variability между training seeds?

[Краткий отчёт](REPORT.md) и [все 300 видео](https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-20261020/tree/main/final).''')
 nb['cells']=cells;nb.metadata['kernelspec']=dict(name='pi0_three',display_name='π₀ three adaptations',language='python')
 # Use the exact experiment environment for execution.
 kernel_dir=WORK/'kernels/pi0_three';kernel_dir.mkdir(parents=True,exist_ok=True)
 write(kernel_dir/'kernel.json',dict(argv=['/workspace/pi0_venv/bin/python','-m','ipykernel_launcher','-f','{connection_file}'],display_name='π₀ three adaptations',language='python'))
 KernelSpecManager().install_kernel_spec(str(kernel_dir),kernel_name='pi0_three',user=True,replace=True)
 path=ROOT/'experiments/pi0_d0_d1_three_adaptations/seminar.ipynb'
 nbformat.write(nb,path)
 NotebookClient(nb,timeout=1800,kernel_name='pi0_three',resources={'metadata':{'path':str(ROOT)}}).execute()
 assert all(c['cell_type']!='code' or c['execution_count'] is not None for c in nb.cells)
 assert not any(o.get('output_type')=='error' for c in nb.cells if c.cell_type=='code' for o in c.get('outputs',[]))
 nbformat.write(nb,path)
 out=WORK/'seminar';out.mkdir(exist_ok=True);import shutil;shutil.copy2(path,out/path.name)
 shutil.copy2(path.with_name('REPORT.md'),out/'REPORT.md')
 publish(out,'seminar',full_readback=True);push_progress('exp: publish executed student seminar with three weight-to-video examples')
 print('EXECUTED_SEMINAR_PUBLISHED',flush=True)
if __name__=='__main__':main()

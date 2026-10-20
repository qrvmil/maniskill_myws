"""Readable HF model card, always explicit about incomplete versus final results."""
from common import *
def main():
 from huggingface_hub import HfApi,hf_hub_download
 results=ROOT/'experiments/pi0_d0_d1_three_adaptations/results';complete=(results/'completeness.json').exists()
 text='''---
library_name: openpi
language:
- ru
- en
tags:
- robotics
- libero
- pi0
- fine-tuning
- reinforcement-learning
---
# π₀: три способа адаптации D0 → D1

Эксперимент сравнивает full SFT, action-expert SFT и residual RL из одной общей D0-модели. Это новые episodes двух знакомых LIBERO tasks, а не OOD или held-out demonstrations.

- D0: `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`.
- D1: `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`.
- Все 50 native demonstrations каждой задачи, pinned source revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`; без filtered RLDS или новых demos.
- Официальный pi0_base → полный D0 SFT3000 updates, batch8, без LoRA. Два независимых D1 SFT по1500 updates с новыми optimizer states и одной immutable D0-only normalization.
- Residual: frozen D0 + corrected V3 SAC/Cal-QL, V4 роли D0/D1; teacher-warmup и potential shaping — явные изменения V4. `clip(a_D0 + 2*residual, -1,1)` в OSC units; scale2 даёт широкую correction authority. Online OTF; evaluation deterministic actor; probe_fraction0.

[Код и зафиксированный протокол](https://github.com/qrvmil/maniskill_myws/tree/exp/pi0-d0-d1-three-adaptations-20261020/experiments/pi0_d0_d1_three_adaptations), [текущий статус](https://github.com/qrvmil/maniskill_myws/blob/exp/pi0-d0-d1-three-adaptations-20261020/experiments/pi0_d0_d1_three_adaptations/STATUS.md).

[Normalization](normalization/d0), [checkpoints и resumable states](checkpoints), [generated reset states](reset_states), [episode/video artifacts](final).
'''
 if complete:
  stats=json.loads((results/'success_rates.json').read_text())
  text+='\n## Завершённая финальная оценка\n\n300 episodes и300 двухкамерных MP4 с terminal frame, N50 в каждой ячейке, horizon220/replan5; одинаковые reset states и flow-noise streams. Wilson95% CI:\n\n| Метод | D0 SR | D1 SR |\n|---|---|---|\n'
  for m in ('full','expert','residual'):
   values=[]
   for task in TASKS:
    s=stats[m+'_'+task];lo,hi=s['wilson_95ci'];values.append(f"{s['successes']}/50 ({100*s['success_rate']:.0f}%; {100*lo:.1f}–{100*hi:.1f}%)")
   text+='| '+m+' | '+' | '.join(values)+' |\n'
  text+='\n[Выполненный seminar notebook и краткий отчёт](seminar), [episode CSV, CI и paired comparisons](results). Одна training seed на метод; RL получает дополнительный online опыт и state-based shaping. Финальные seeds не использовались для настройки.\n'
 else:text+='\n## Статус: обучение продолжается\n\nФинальные модели, 300 episodes, SR и выполненный notebook пока не готовы. Не интерпретируйте промежуточные артефакты как завершённый результат.\n'
 path=WORK/'model_card/README.md';path.parent.mkdir(exist_ok=True);path.write_text(text)
 commit=HfApi().upload_file(path_or_fileobj=path,path_in_repo='README.md',repo_id=HF_REPO,commit_message='Update experiment status and reproducibility links')
 downloaded=hf_hub_download(HF_REPO,'README.md',revision=commit.oid,force_download=True,local_dir=WORK/'model_card_readback')
 assert sha(downloaded)==sha(path)
 write(WORK/'evidence/hf_model_card.json',dict(commit=commit.oid,sha256=sha(path),completed=complete,readback=True))
 print('HF_MODEL_CARD_VERIFIED',commit.oid,flush=True)
if __name__=='__main__':main()

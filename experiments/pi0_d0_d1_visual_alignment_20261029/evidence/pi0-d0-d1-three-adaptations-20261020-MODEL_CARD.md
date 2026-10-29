---
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

## Завершённая финальная оценка

300 episodes и300 двухкамерных MP4 с terminal frame, N50 в каждой ячейке, horizon220/replan5; одинаковые reset states и flow-noise streams. Wilson95% CI:

| Метод | D0 SR | D1 SR |
|---|---|---|
| full | 0/50 (0%; 0.0–7.1%) | 20/50 (40%; 27.6–53.8%) |
| expert | 0/50 (0%; 0.0–7.1%) | 19/50 (38%; 25.9–51.8%) |
| residual | 0/50 (0%; 0.0–7.1%) | 0/50 (0%; 0.0–7.1%) |

[Выполненный seminar notebook и краткий отчёт](seminar), [episode CSV, CI и paired comparisons](results). Одна training seed на метод; RL получает дополнительный online опыт и state-based shaping. Финальные seeds не использовались для настройки.

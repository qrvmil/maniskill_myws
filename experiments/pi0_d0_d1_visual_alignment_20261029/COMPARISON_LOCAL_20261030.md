# Локальный эвал visual alignment — 2026-10-30

| Эксперимент | D0 SR | D1 SR |
|---|---:|---:|
| D0 Full SFT aligned / 3000 | 45/50 (90%) | 0/50 (0%) |
| D1 Full SFT + alignment / 1500 | 0/50 (0%) | 31/50 (62%) |
| D1 Expert SFT / 1500 (aligned D0 base) | 0/50 (0%) | 21/50 (42%) |
| Residual RL (финальный checkpoint отсутствует) | — | — |

Все численные результаты этой таблицы получены локально на RTX 5080: 3 финальных SFT-модели × 2 задачи × 50 сцен = **300 эпизодов и 300 проверенных MP4**. Residual RL final отсутствует в HF revision `b14191f0183c19b717c6662dfeb336a23e13ab46`: опубликован только `after_teacher_warmup` с 0/50000 active steps. Промежуточный checkpoint не выдаётся за финальный, дополнительное RL-обучение не выполнялось.

## Кратко об экспериментах

- **D0 Full SFT aligned**: вся π0 обучалась 3000 updates на D0; visual alignment приближает пространственные visual features к frozen C-RADIO. Projector калиброван только на D0; λ=0.2.
- **D1 Full SFT + alignment**: из общей aligned D0-базы все параметры обучались ещё 1500 updates на D1; alignment продолжался.
- **D1 Expert SFT**: из той же aligned D0-базы на D1 обучался action expert и state/action/time projections; PaliGemma/vision заморожены. Активного visual-alignment gradient в этой ветке нет.
- **Residual RL**: планируется обучать добавочную коррекцию действий поверх frozen aligned D0. Опубликованный warmup не завершает зарегистрированный бюджет 50000 active transitions.

## Неопределённость SR

| Model | Task | Successes | SR | Wilson 95% CI |
|---|---|---:|---:|---|
| d0 | D0 | 45/50 | 90% | 78.6–95.7% |
| d0 | D1 | 0/50 | 0% | 0.0–7.1% |
| full | D0 | 0/50 | 0% | 0.0–7.1% |
| full | D1 | 31/50 | 62% | 48.2–74.1% |
| expert | D0 | 0/50 | 0% | 0.0–7.1% |
| expert | D1 | 21/50 | 42% | 29.4–55.8% |

## Историческое сравнение без alignment

| Без alignment, исторические результаты | D0 SR | D1 SR |
|---|---:|---:|
| D0 Full SFT / 3000 | 43/50 (86%) | 0/50 (0%) |
| D1 Full SFT | 0/50 (0%) | 20/50 (40%) |
| D1 Expert SFT | 0/50 (0%) | 19/50 (38%) |
| Residual RL | 0/50 (0%) | 0/50 (0%) |

Исторические строки взяты из эвала D0 от 2026-10-21 и опубликованных результатов эксперимента 20261020; в текущем запуске они повторно не оценивались. Помимо alignment в новой ветке изменены training augmentations (детерминированные K4 crops/brightness/contrast вместо прежнего потока). Поэтому разницу SR нельзя приписывать только alignment без отдельного нового no-alignment control. По одной training seed на рецепт.

## Протокол и достоверность

- D0: `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`; D1: `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`.
- По 50 episodes, seeds 30000–30049; horizon220, replan5, flow-noise50×32, LIBERO actions7D. Финальные checkpoints выбраны по фиксированному бюджету, без настройки по final seeds.
- Новая HF revision ещё не содержит final reset bank. Для всех новых моделей использован один сохранённый generated scene bank эксперимента 20261020, HF revision `57b978774e8e177d8d721b71f15542751e0610d9`: те же задачи, seeds, BDDL и reset protocol; не benchmark init files. Исходные sampled states совпадают побайтово.
- Между всеми тремя новыми локальными моделями reset hashes и initial observation hashes совпадают **побайтово на всех 100 сценах**; flow-noise prefixes также идентичны. По сравнению с историческим удалённым запуском max post-settle physics drift `1.02490055915e-07` при atol1e-6/rtol0; совпали 246/300 initial observation hashes. Межмашинная побайтовая идентичность траекторий не заявляется.
- Веса float32, bf16 autocast, без compile; отключён cache копий bf16 weights для 16 GiB GPU, без изменения арифметического рецепта. C-RADIO/projector не требуются при inference.
- SHA256 weights и norm проверены по checkpoint inventories. Для каждого из 300 видео проверены SHA256, полное декодирование, размер512×256, length+1 кадров с terminal frame. Сохранены JSON каждого эпизода, агрегированные JSON/CSV, video index и completion markers.
- Парные различия SR, bootstrap95% (10000 draws, seed0), exact McNemar и Holm для 6 сравнений сохранены в `paired_comparisons.json`; пары используют строго совпадающие локальные resets. Все записи уникальны по method/task/seed.

Weights SHA256:
- `d0/3000`: `4b50673559a5dcc55281d0793eab2013011503663976d937fd1bec5b626c8356`.
- `full/1500`: `c7b6bb255d68db1444edfa8ae187f9e03ed9412ff7966d19655606ad2237e738`.
- `expert/1500`: `444b243d57f2dd760fa0540e8ff356c681ef07f403ca4e744ff956028f700bdd`.

Normalization SHA256: `a06c2e745bf91d8dd6a3fa6377560d3104e6cad621da4ecebe1372c74bbb388e`.

## Артефакты и воспроизведение

Локальная папка всех новых видео: `pi0-eval-libero/visual_alignment_eval_20261030/final/`, структура `<method>/<task>/<seed>/*.mp4`. Индекс и episode JSON лежат рядом; общие CSV/JSON и `complete.json` — в корне `visual_alignment_eval_20261030`.

Используется прежний проект и окружение `runtime/maniskill_myws/third_party/openpi/.venv/bin/python`. Запуск/продолжение — `run_visual_alignment_eval.py`, проверка и сборка таблицы — `report_visual_alignment_eval.py`. Checkpoints загружаются по очереди из закреплённой revision; после успешной проверки удаляется только новая скачанная копия model weights, inventories/метаданные/видео сохраняются.

## Источники

- [GitHub: qrvmil/maniskill_myws](https://github.com/qrvmil/maniskill_myws), ветка `exp/pi0-d0-d1-visual-alignment-20261029`.
- [Протокол и инструкции](https://github.com/qrvmil/maniskill_myws/tree/c79044f240332b13af868479d358a21d5a15030b/experiments/pi0_d0_d1_visual_alignment_20261029), source commit `c79044f240332b13af868479d358a21d5a15030b`.
- [Новые HF checkpoints](https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-visual-alignment-20261029/tree/b14191f0183c19b717c6662dfeb336a23e13ab46/checkpoints), revision `b14191f0183c19b717c6662dfeb336a23e13ab46`.
- [Исторические checkpoints/results и generated scenes](https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-20261020/tree/57b978774e8e177d8d721b71f15542751e0610d9).
- OpenPI `981483dca0fd9acba698fea00aa6e52d56a66c58`; LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`.

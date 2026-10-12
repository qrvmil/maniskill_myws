# TurboVLA: эвал на D0, D1, D2, H1, H2

Выполненный эвал опубликованного TurboVLA без нового дообучения на RTX 4090: по 50 эпизодов на каждую из пяти задач в двух протоколах, всего 500 эпизодов. Все видео сохранены отдельно от ноутбуков.

- [Поэтапный семинар](01_turbovla_step_by_step.ipynb): окружение, загрузка полного checkpoint, подготовка изображений и состояния, LIBERO API, цикл действий и выбор новой задачи.
- [Результаты и VRAM](02_results_and_vram.ipynb): оба полных прогона, исходные результаты π₀ A/B/C, доверительные интервалы и измерения памяти.
- [CSV результатов](results/success_rates.csv), [аудит](results/completion_audit.json).

Оба ноутбука выполнены сверху вниз и содержат сохранённые outputs. Видеофайлы находятся в `results/reference_n50/videos/<task>/` и `results/official_n50/videos/<task>/`. JSON рядом с каждым видео связывает его с эпизодом, состоянием сброса и исходом.

| Задача | Reset-состояния π₀ | Стандартные initial states |
|---|---:|---:|
| D0 | 50/50 | 50/50 |
| D1 | 45/50 | 50/50 |
| D2 | 50/50 | 49/50 |
| H1 | 50/50 | 50/50 |
| H2 | 50/50 | 49/50 |
| Среднее по пяти задачам | 98% | 99,2% |

Reference-mode использует seed 10000–10049, 10 settling steps, MSAA off, горизонты из исходного эксперимента (D2=280). Все 250 физических reset-хэшей совпали с исходными записями π₀. Official-mode использует benchmark initial states 0–49, seed 7 один раз на задачу и штатные горизонты (D2=300). Два контрольных эпизода D0 дали побайтовое совпадение действий с неизменённым rollout авторов. TurboVLA исполняет чанки по 12 действий; исходный π₀ использовал replan=5.

**H1/H2 входят в обучение опубликованного TurboVLA.** Для него эти результаты не являются OOD-оценкой. Сравнение с π₀ имеет разное обучающее покрытие, preprocessing и replan. Среднее по нашим пяти задачам не заменяет полный 40-task benchmark статьи.

## Память

Policy-only BF16: **0,921 GiB / 0,989 GB**, подтверждено повторным замером; median forward **31,8 ms**. FP32: **1,388 GiB / 1,490 GB**. Полный официальный эвал с LIBERO/EGL и видеозаписью: **1,454 GiB / 1,561 GB**.

Измерение политики идёт в отдельном процессе: idle GPU → `nvidia-smi` sampling каждые 50 ms → загрузка всей модели → 10 warmup → 50 synchronized forwards. Симулятор не входит в policy-only измерение. Raw samples и методика находятся в ноутбуке и `results/vram_*`. Файлы с `concurrent` в имени содержат диагностический замер с кратким перекрытием подготовки наблюдения; они исключены из изолированной оценки памяти.

## Повторить

Из корня репозитория:

```bash
cd turbovla_eval
bash setup.sh
```

Затем открыть ноутбуки и выбрать kernel **TurboVLA LIBERO (Python 3.10)**. Скрипт загрузит закреплённые версии TurboVLA и LIBERO, полный публичный checkpoint и tokenizer. Локальные BERT/DINOv3 weights извлекаются из полного checkpoint и проверяются строгой загрузкой; отдельное скачивание gated DINOv3 weights не требуется. Веса, симулятор и `.venv` исключены из Git.

```bash
.venv/bin/python run_eval.py --protocol reference --episodes 50 --tasks D0,D1,D2,H1,H2 --output results/my_reference_run
.venv/bin/python run_eval.py --protocol official --episodes 50 --tasks D0,D1,D2,H1,H2 --output results/my_official_run
.venv/bin/python benchmark_vram.py --precision bf16 --output results/my_vram_run
.venv/bin/python -m pytest -q
```

Перед VRAM-тестом завершите GPU kernels и другие эвал-процессы. Повторные запуски требуют нового имени выходной папки. `setup.sh` рассчитан на Linux/NVIDIA и использованный RTX 4090.

Источники и версии: [статья v3](https://arxiv.org/pdf/2607.27205v3), [код TurboVLA](https://github.com/H-EmbodVis/TurboVLA), [checkpoint](https://huggingface.co/H-EmbodVis/TurboVLA), `provenance/source_revisions.json`, `assets/hf_revision.json`, `provenance/environment.txt`. Лицензии стороннего кода и DINOv3 сохранены в `provenance/` и `assets/`.

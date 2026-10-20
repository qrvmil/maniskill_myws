# Запуск и восстановление

Рабочий код: `/workspace/pi0_experiment`, ветка `exp/pi0-d0-d1-three-adaptations-20261020`. Артефакты: `/workspace/pi0_three`; Python: `/workspace/pi0_venv/bin/python`. Эта машина не имеет persistent workspace volume; stop/start сохраняет диск, recycle/destroy — нет. HF: `qrvmil/pi0-d0-d1-three-adaptations-20261020`. Credentials находятся в стандартном для этой сессии cache, токены не входят в код/логи.

## Состояние на 2026-10-20, 14:20 UTC

Основной D0 full SFT работает, пройдено более500/3000 updates. Никаких финальных SR пока нет. Pipeline ждёт завершения D0 и затем запускает остальные стадии последовательно. Текущее состояние всегда проверять по supervisor и метрикам, а не по этой датированной заметке.

```bash
supervisorctl status pi0_three_d0 pi0_three_pipeline
tail -20 /var/log/portal/pi0_three_d0.log
tail -20 /var/log/portal/pi0_three_pipeline.log
```

64 contract tests passed, 2 optional skipped; дополнительные5 statistical tests passed. Реальные MuJoCo resets, explicit state replay, shaping и обе камеры прошли smoke. На CPU подтверждён native LIBERO success для50/50 D1 demos. Официальный pi0_base проверен по33 GCS objects/MD5; обе native HDF5 — по HF LFS SHA256. D0 normalization опубликована/прочитана: SHA256 `a06c2e745bf91d8dd6a3fa6377560d3104e6cad621da4ecebe1372c74bbb388e`.

Основной D0 run использует float32 master weights и bf16 autocast, microbatch2×accumulation4. Profile update отброшен; его loss совпал с первым update основного run. Весь scope открыт optimizer (777 tensors); семь официальных PyTorch outputs не участвуют в action loss, поэтому770 имеют gradients. Список в evidence. Нет LoRA.

## Resume

`latest.json` внутри `sft/<stage>` или `rl` указывает на целый checkpoint. Файлы сначала сохраняются атомарно, затем загружаются и проверяются. Не прерывать работающий GPU run ради косметических правок кода; текущий процесс уже загрузил свои Python modules. Новые child stages читают обновлённые scripts при старте.

При незавершённом D0, после фактического исправления ошибки/восстановления зависимостей:

```bash
supervisorctl restart pi0_three_d0
supervisorctl restart pi0_three_pipeline
```

Если D0 уже завершён, перезапускайте только `pi0_three_pipeline`. D0 wrapper в таком случае проверит hashes и завершится без нового profile/training.

D0 wrapper проверяет существующую конверсию/data/norm и возобновляет optimizer/RNG из последнего checkpoint. Пока первый checkpoint1000 не сохранён, прерывание теряет текущие updates. Pipeline сам переиспользует завершённые SFT stages. RL сохраняет actor/critic/temperature optimizers, replay, RNG и counters на границах episodes. Удалённые receipts и hashes лежат в `evidence/`.

При необходимости восстановить с HF соответствующую checkpoint directory в тот же локальный путь, проверить `inventory.json`/SHA256 и восстановить `latest.json`. Не менять видимость существующих repositories.

Финальная оценка открывается после `final_selection.json` с hashes трёх последних checkpoints. Seeds30000–30049 до этого не используются. Завершённый final episode с `episode.json` и проверенным MP4 переиспользуется; отсутствие upload receipt вызывает повторную публикацию, а не новый rollout. При повреждении опубликованного видео восстановить исходные bytes из HF. Неполный encode без episode JSON сохраняется отдельно в `incomplete_final`.

## Окончание

Нужны все300 episodes/MP4 (6 ячеек по50), общий D0 checkpoint, ранний teacher, три финальные модели, проверенные hashes frozen scope, CI/paired comparisons, выполненный seminar notebook и5-minute report. `report.py` проверяет полноту; `build_notebook.py` выполняет notebook, используя проверенные episodes. После фактических результатов ещё проверить интерпретацию выводов в REPORT.md и визуально несколько финальных видео. Goal считается выполненным только после проверенной публикации всего этого; промежуточный запуск не является результатом.

На update1000 сохранён целый checkpoint для восстановления при смене только способа передачи файлов. Последовательный multipart HF uploader оказался медленным; включён поддерживаемый установленной библиотекой `HF_HUB_ENABLE_HF_TRANSFER=1`. Это изменение I/O, без изменения model/data/optimizer/schedule. Уже успешный memory/inference profile переиспользуется с проверкой batch/microbatch/fp32-master settings.

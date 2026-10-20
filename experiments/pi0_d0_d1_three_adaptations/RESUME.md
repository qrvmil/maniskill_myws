# Запуск и восстановление

Рабочий код: `/workspace/pi0_experiment`, ветка `exp/pi0-d0-d1-three-adaptations-20261020`. Артефакты: `/workspace/pi0_three`; Python: `/workspace/pi0_venv/bin/python`. Эта машина не имеет persistent workspace volume; stop/start сохраняет диск, recycle/destroy — нет. HF: `qrvmil/pi0-d0-d1-three-adaptations-20261020`. Credentials находятся в стандартном для этой сессии cache, токены не входят в код/логи.

## Состояние на 2026-10-20, 19:30 UTC

D0 full SFT3000 и оба D1 SFT1500 завершены. Общий D0 (`9ba40f3f7c1b7eb40805df49901109f9fe00089aae10b04eab412d62694cb451`), ранний teacher250 и финальный full D1 опубликованы и прошли полное чтение policy weights из HF. Expert1500 сейчас публикуется; независимо прочитаны все603 frozen tensors из сохранённого safetensors — hashes и float32 dtype совпали. Оба D1 optimizer имеют ровно1500 steps; все777 initial hashes, sampler и первый loss совпали между full/expert.

Teacher250 имел3/10 на validation22000–22009; повторная проверка после reset fix также3/10, weights прежние. До RL zero gate обнаружил разные sensor caches у fresh/restored resets с одинаковым physics hash. Исправлено одинаковым `set_init_state` refresh до10 settle steps, добавлены initial-observation hashes в reset NPZ и final protocol. Реальные12 rollout (6 base/zero pairs) совпали побайтово по images, actions, physics trajectories, lengths/success и noise. Бюджеты и weights не менялись, final seeds не использовались.

Native D1 replay пересчитан:50/50 demos,5055 transitions, SHA256 `12d788d4ff17ad2dbd76fa13a37427a718725f57b5dede280140a86dcdf629ee`. Независимый `audit_offline.py` подтвердил исходные actions и обе камеры, next-base cache и absorbing zero sentinel, shared shaped rewards/returns (ошибки <3e-8). Residual сейчас выполняет1000 Cal-QL updates, затем20 teacher warmup episodes и50000 active transitions. `pi0_three_pipeline` и `pi0_three_publish` работают; `pi0_three_d0` штатно EXITED. В `pipeline.py` исправлена обработка nonzero CLI exit supervisor status для штатно завершённой зависимости.

300 финальных episodes/videos, SR и выполненного notebook ещё нет. Следить за finite RL metrics, warmup actor/critic/temperature hash changes, base freeze, publication receipts. После обучения pipeline сам фиксирует final selection, запускает300 episodes и notebook. Требуются итоговая проверка интерпретации отчёта и визуальный просмотр нескольких final videos.

## Ранняя датированная заметка, 14:20 UTC

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

## Асинхронная публикация

`pi0_three_publish` (supervisor) обрабатывает immutable snapshots из `/workspace/pi0_three/publication_queue`. GPU обучение не ждёт окончания больших transfers. Сохраняются полные optimizer/RNG/replay snapshots; snapshots, которые ещё загружаются, не удаляются. Непубликовавшиеся промежуточные SFT snapshots могут быть помечены `superseded`, когда сохранён более новый целый model + optimizer/RNG snapshot или стадия завершена; они не входят в запрошенный набор моделей. Изменения очереди защищены общей блокировкой, чтобы cleanup не пересекался с началом upload. Финальные D0/full/expert/residual и выбранный teacher защищены и публикуются обязательно. Teacher хранится отдельным hardlink к точным ранним full-SFT weights с normalization и validation evidence, без ненужного teacher optimizer.

До финального отчёта pipeline выполняет publication barrier: все требуемые models, resets и300 episodes/videos должны иметь проверенный readback. Статус очереди и ошибки — `/var/log/portal/pi0_three_publish.log`. Новое окружение uploader изолировано от OpenPI: HF Hub2.2/Xet1.7, обучение продолжает использовать pinned зависимости.

На этой машине adaptive Xet controller снизил upload concurrency до4 после timeout отдельных соединений. Для изолированного publisher задано32 параллельных upload streams и read timeout600s ([официальные настройки](https://huggingface.co/docs/hub/xet/using-xet-storage)). Повторный upload использует chunk deduplication; изменения касаются только передачи bytes, не обучения.

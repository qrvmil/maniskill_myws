# Состояние и продолжение запуска

Этот файл описывает управление запуском, а не финальные результаты. Текущие updates: `LIVE_STATUS.json`; протокол и неизменяемые источники: `PREREGISTRATION.md` и `SOURCE_AUDIT.md`.

Vast host 150140, instance 58231938; одна CMP170HX. Файловая система не является persistent volume: recycle/destroy удалит локальные данные. Не выполнять эти действия до проверенной внешней публикации.

Код: `/workspace/visual-alignment-repro-20261029/maniskill_myws`.
Рабочие артефакты: `/workspace/visual-alignment-repro-20261029/run`.
Аппаратные логи: `/workspace/visual-alignment-repro-20261029/evidence`.
GitHub: `qrvmil/maniskill_myws`, ветка `exp/pi0-d0-d1-visual-alignment-20261029`.
Публичный HF: `qrvmil/pi0-d0-d1-three-adaptations-visual-alignment-20261029`.

Все команды запускают из корня репозитория после:

```bash
source experiments/pi0_d0_d1_visual_alignment_20261029/run-config.sh
supervisorctl status pi0_va_d0 pi0_va_pipeline pi0_va_publish pi0_va_logs
```

HF_HOME в этом конфиге обязателен: default supervisor окружение указывает другой каталог. Токены не копировать в конфиги, логи или чат. Аутентификация GitHub и машинного HF — `qrvmil`; read-only HF app connector — отдельный аккаунт `dpaleyev`.

Завершено перед длинным обучением: источники/датасеты/normalization проверены; CUDA/JAX/PyTorch probes; тесты0–8 всей allocatable VRAM и исправленный64bit test10 на64606MiB×1000iterations; ECC0; повторный реальный model smoke с3AdamW updates и запасом6.79GiB. Исходный32bit test10 не считался доказательством полного stress: дефект, patch и повторный запуск сохранены. Optional3hour bit-fade9 не запускался.

Завершены D0K4teacher cache,200projector updates и discarded20update pilot. λ=.2; projector frozen. Общая D0SFT3000updates выполняется. Первый update с optimizer/RNG опубликован и проверен чтением до продолжения: HF commit `866f778afb3494794429565d34501ba7168cec7a`. Остальные этапы выполняет `pi0_va_pipeline`, дожидаясь успешного окончания D0.

Порядок pipeline: D1cache → full1500 → expert1500 → проверка early teacher → zero residual equivalence → native offline replay → CalQL1000/warmup20/active50000 → online audit → фиксация final checkpoints →300paired evaluations → статистика/отчёт → HF delivery audit → сохранение completion. Новые данные не подменять историческими checkpoints/results. Если ни один зарегистрированный teacher candidate не пройдёт≥1/10validation successes, pipeline остановится с объяснением; менять правило выбора нельзя.

`pi0_va_publish` публикует очередь snapshots, `pi0_va_logs` сохраняет Git/HF логи каждые5минут; `pi0_preflight_telemetry` пишет GPU/ECC каждые5секунд. Перед крупными этапами pipeline выполняет drain и отдельную проверенную публикацию логов. Не удалять неопубликованные snapshots. Опубликованные старые локальные snapshots может очищать штатный training controller; история остаётся в HF.

При остановке сначала изучить `/var/log/portal/pi0_va_*.log`, telemetry и receipts. Не перезапускать длинное обучение после новых ошибок VRAM/ECC или нестабильного model smoke. Сохранить доказательства и объяснить проблему. После программного сбоя контроллеры восстанавливают максимальный committed checkpoint и проверяют lineage, optimizer/RNG и внешнюю публикацию; `.partial` не является checkpoint. Уже завершённые final episodes проверяются и переиспользуются.

После проверки причины можно запустить остановившиеся собственные сервисы через `supervisorctl start <name>`; не перезапускать работающую D0-модель ради косметических изменений. Не трогать caddy/instance_portal/tunnel_manager. Не запускать второй GPU learner параллельно. Для ручной публикации:

```bash
/venv/main/bin/python -u experiments/pi0_d0_d1_visual_alignment_20261029/tools/save_progress.py 'exp: preserve inspected experiment state'
```

Завершение подтверждается только `run/complete.json` после успешных audit и публикации, а не наличием фонового процесса. До этого финальных SR нет. Итоговые ссылки: `REPORT.md`, `results/`, `VIDEOS.md`, HF `checkpoints/`, `preflight/evidence/` и `completion/`.

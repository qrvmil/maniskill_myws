# Статус

Обновлено: 2026-10-20, до первого gradient run.

- GitHub и HF: реальные write/read probes прошли, evidence в `access_results.json`.
- Native D0/D1: оба HDF5 из pinned revision скачаны и проверены по HF LFS SHA256; все 50 demos проверяются перед SFT.
- Официальный pi0_base: 33/33 GCS objects проверены по размерам и MD5, 11.19 GiB.
- GPU: NVIDIA RTX 6000D, 85651 MiB, compute capability12.0; CUDA12.8 wheel обязателен. Устанавливается PyTorch2.7.1+cu128; JAX0.5.3 используется только для conversion/helpers на CPU.
- Реализованы отдельные стадии SFT, scope/hash audits, native offline replay, V3 SAC + teacher-warmup/shaping, paired evaluation, статистика и генератор выполненного notebook.
- Pure NumPy invariants прошли; полный набор contract tests, memory/inference profile и фактическое обучение ещё не завершены. SR и финальных результатов пока нет.

Supervisor: `pi0_three_assets`, `pi0_three_runtime`, `pi0_three_d0`, `pi0_three_pipeline`. Логи: `/var/log/portal/<name>.log`. Wrapper/config snapshots рядом в `runtime/`. `start_d0.sh` ждёт проверенных assets и runtime, затем выполняет tests → conversion → data audit/norm → discarded memory/inference profile → D0 training. Pipeline ждёт завершения D0 и последовательно запускает оставшиеся GPU стадии. Финальная оценка открывается только после сохранения `final_selection.json`.

Данные и артефакты: `/workspace/pi0_three`; окружение: `/workspace/pi0_venv`; код: `/workspace/pi0_experiment`. Основные стадии имеют checkpoints с optimizer/RNG/replay для resume и проверенную публикацию в HF. Повторная финальная evaluation проверяет существующие episode/video artifacts и не добавляет duplicates.

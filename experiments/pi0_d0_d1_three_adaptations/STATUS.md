# Статус эксперимента

Обновлено UTC: 2026-10-20T18:20:32.044776+00:00

GitHub/HF write/read probes прошли. Бюджеты и правила выбора: [PREREGISTRATION.md](PREREGISTRATION.md).

- SFT d0: 3000 выполненных updates; 3000 updates; checkpoint /workspace/pi0_three/sft/d0/3000
- SFT full: 1453 выполненных updates; 1000 updates; checkpoint /workspace/pi0_three/sft/full/1000
- SFT expert: ещё нет сохранённого checkpoint
- Teacher: 250 D1 full-SFT updates, validation passed
- Residual RL: ещё нет сохранённого checkpoint
- Финальные episodes: 0/300. Каждому завершённому episode соответствует проверенное двухкамерное видео с terminal frame.

Final seeds не используются для настройки или выбора checkpoints. Завершённые episodes при resume проверяются и переиспользуются без увеличения N.

Код: `/workspace/pi0_experiment`; артефакты: `/workspace/pi0_three`; среда: `/workspace/pi0_venv`. Supervisor и логи: `pi0_three_d0`, `pi0_three_pipeline`, `/var/log/portal/<name>.log`.

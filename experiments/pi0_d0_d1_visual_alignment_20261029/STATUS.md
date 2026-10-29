# Статус эксперимента

Обновлено UTC: 2026-10-29T20:27:49.882926+00:00

GitHub/HF write/read probes прошли. Бюджеты и правила выбора: [PREREGISTRATION.md](PREREGISTRATION.md).

- SFT d0: 3000 выполненных updates; 3000 updates; checkpoint /workspace/visual-alignment-repro-20261029/run/sft/d0/3000
- SFT full: 1500 выполненных updates; 1500 updates; checkpoint /workspace/visual-alignment-repro-20261029/run/sft/full/1500
- SFT expert: 500 выполненных updates; 500 updates; checkpoint /workspace/visual-alignment-repro-20261029/run/sft/expert/500
- Teacher: 250 D1 full-SFT updates, validation passed
- Residual RL: ещё нет сохранённого checkpoint
- Финальные episodes: 0/300. Каждому завершённому episode соответствует проверенное двухкамерное видео с terminal frame.

Final seeds не используются для настройки или выбора checkpoints. Завершённые episodes при resume проверяются и переиспользуются без увеличения N.


Новый запуск: alignment общей D0-базы; full D1 продолжает alignment; expert и residual сохраняют исходные frozen scopes и наследуют aligned D0. Протокол: PREREGISTRATION.md.
Рабочий код: `/workspace/visual-alignment-repro-20261029/maniskill_myws`; артефакты: `/workspace/visual-alignment-repro-20261029/run`; HF: https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-visual-alignment-20261029.
Аппаратный и модельный preflight прошли; исходные historical results не являются результатами этого запуска.

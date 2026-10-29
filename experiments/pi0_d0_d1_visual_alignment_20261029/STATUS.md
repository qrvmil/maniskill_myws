# Статус эксперимента

Обновлено UTC: 2026-10-29T11:49:29.446583+00:00

GitHub/HF write/read probes прошли. Бюджеты и правила выбора: [PREREGISTRATION.md](PREREGISTRATION.md).

- SFT d0: 1 выполненных updates; 1 updates; checkpoint /workspace/visual-alignment-repro-20261029/run/sft/d0/1
- SFT full: ещё нет сохранённого checkpoint
- SFT expert: ещё нет сохранённого checkpoint
- Teacher: ещё не выбран по отдельной validation
- Residual RL: ещё нет сохранённого checkpoint
- Финальные episodes: 0/300. Каждому завершённому episode соответствует проверенное двухкамерное видео с terminal frame.

Final seeds не используются для настройки или выбора checkpoints. Завершённые episodes при resume проверяются и переиспользуются без увеличения N.


Новый запуск: alignment общей D0-базы; full D1 продолжает alignment; expert и residual сохраняют исходные frozen scopes и наследуют aligned D0. Протокол: PREREGISTRATION.md.
Рабочий код: `/workspace/visual-alignment-repro-20261029/maniskill_myws`; артефакты: `/workspace/visual-alignment-repro-20261029/run`; HF: https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-visual-alignment-20261029.
Аппаратный и модельный preflight прошли; исходные historical results не являются результатами этого запуска.

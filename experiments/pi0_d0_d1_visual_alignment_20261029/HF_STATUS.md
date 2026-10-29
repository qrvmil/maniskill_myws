---
language: [ru, en]
tags: [pi0, openpi, libero, robotics, visual-alignment]
---
# π0 D0→D1: три адаптации общей aligned D0-базы

Запуск продолжается. Финальных D0/D1 success rates пока нет; файлы из исходных экспериментов в source audit явно обозначены как historical reference.

Общая база обучается заново из точных исходных π0 weights: D0 full SFT3000updates с frozenC-RADIO spatial alignment. Из этой базы независимо D1full/expert по1500updates; residual сохраняет исходный SAC/CalQL+teacher warmup и50000active transitions. Expert/residual сохраняют исходные frozen scopes и наследуют aligned D0; fullD1 продолжает alignment.

Все50native demonstrations каждой задачи; исходная D0-only normalization; effective batch8,micro2accum4,fp32master/AdamW,bf16compute и исходные optimizer/LR budgets. Projector200D0-only updates и discarded20step D0pilot завершены; λ=0.2. Final evaluation:300paired rollouts, seeds30000–30049,horizon220,replan5.

На выбранном Vast host150140/instance58231938 проверены64606MiB allocatableVRAM. Upstream32bitstress overflow исправлен: дополнительный64bit full-volume test1000iterations,141.3s,exit0,ECC0. После него повторно прошли загрузка π0/C-RADIO, alignment loss и3AdamW updates с минимальным запасом6.79GiB. Полные логи, версии, patch/source/binary hashes сохранены.

- [Протокол](https://github.com/qrvmil/maniskill_myws/blob/exp/pi0-d0-d1-visual-alignment-20261029/experiments/pi0_d0_d1_visual_alignment_20261029/PREREGISTRATION.md)
- [Live status](https://github.com/qrvmil/maniskill_myws/blob/exp/pi0-d0-d1-visual-alignment-20261029/experiments/pi0_d0_d1_visual_alignment_20261029/LIVE_STATUS.json)
- [Код и source audit](https://github.com/qrvmil/maniskill_myws/tree/exp/pi0-d0-d1-visual-alignment-20261029/experiments/pi0_d0_d1_visual_alignment_20261029)
- [Preflight и runtime logs](preflight/evidence)
- [Созданные checkpoints](checkpoints)
- [Projector, pilot и cache metadata](alignment)

Исходные datasets, teacher cache arrays, upstream weights, credentials и временные файлы не публикуются. Этот новый репозиторий сделан публичным по явному указанию пользователя; остальные репозитории не изменялись.

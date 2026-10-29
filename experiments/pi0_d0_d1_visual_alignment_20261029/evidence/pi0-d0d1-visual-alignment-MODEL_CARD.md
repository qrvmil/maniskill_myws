---
language:
- ru
library_name: openpi
tags:
- robotics
- pi0
- libero
- visual-alignment
---
# π0: visual alignment на D0+D1

Код и воспроизведение: https://github.com/qrvmil/pi0-d0d1-visual-alignment
Статья: https://arxiv.org/html/2510.25616
Результаты: [REPORT_RU.md](REPORT_RU.md).

Новые B-control и B-align из одной исходной π0, одинаковые seed0, data order, K=4 deterministic augmentations, optimizer/schedule и effectivebatch8. Отличие B-align — cosine middle PaliGemma visual features к frozen C-RADIOv3-L через предварительно обученный, затем frozen LN+MLP.

D0: libero_spatial / pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.
D1: libero_spatial / pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate.
D2: libero_goal / put_the_bowl_on_the_stove — held-out OOD.

**D2 не использовалась для training, calibration, normalization, настройки λ, debugging или выбора модели.** Финальный checkpoint3001 выбран заранее. Все scheduled checkpoints сохранены: B-control/0, B-control/500, B-control/1000, B-control/2000, B-control/3001, B-align/0, B-align/500, B-align/1000, B-align/2000, B-align/3001. Доказанный успех заявляется только при выполнении зарегистрированных статистических критериев; см. численные результаты в отчёте.

Веса в checkpoints/<arm>/<step>/params — Orbax/OpenPI format; train_state содержит optimizer state для resume; assets/local/d0d1/norm_stats.json — D0+D1-only normalization. Projector — projector.npz (LN scale/bias, w1/b1, w2/b2); не нужен при inference. Конфиги/metadata доступны рядом. Используйте pinned OpenPI и LoRA32 из GitHub scripts/setup.sh и alignment/training.py. Исходная π0: gs://openpi-assets/checkpoints/pi0_base; teacher revision и source hashes в configs/pins.json.

Данные, teacher feature cache, исходные upstream checkpoints и credentials в этот репозиторий не загружены. Фактическая машина A10080GB; Blackwell compatibility не проверена. Это один seed обучения; внешние задачи за пределами D2 не оценивались.

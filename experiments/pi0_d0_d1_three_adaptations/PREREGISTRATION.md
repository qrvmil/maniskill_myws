# π₀: D0 → D1, три способа адаптации

Дата регистрации: 2026-10-20. Результатов нового обучения и финальных rollout пока нет.

## Неизменяемые условия

- Ветка: `exp/pi0-d0-d1-three-adaptations-20261020`, исходный commit `c6b77f51e3e5bd8873b812d258744c43380fdfeb`.
- OpenPI `981483dca0fd9acba698fea00aa6e52d56a66c58`; LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`.
- Только две native HDF5 задачи, revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`, все 50 demonstrations каждой. Никаких filtered RLDS или новых demonstrations.
- D0: `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`.
- D1: `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`.
- Сохраняем исправленное temporal alignment: native `obs[i] → action[i+1]`; поворот обеих камер на 180°, официальный resize/padding и LIBERO transforms, без повторного inverse normalization.
- Одна D0-only normalization; вычисляется один раз и копируется побайтово во все checkpoints, включая D1. Все 50 demos остаются в обучении.

## Бюджеты и выбор

- Официальный `gs://openpi-assets/checkpoints/pi0_base` → полный D0 SFT, 3000 optimizer updates, seed 0, effective batch 8. Никаких LoRA. Последний checkpoint становится единственной общей D0-базой.
- Два независимых D1 SFT: по 1500 updates, одинаковый effective batch 8, seed 1, новый AdamW и schedule. Инициализация — одни и те же веса D0. Фиксированный последний checkpoint; метрики финальных seeds не участвуют в выборе.
- Full: все параметры π₀ доступны оптимизатору. Expert: `gemma_expert` и state/action/time projections; вся PaliGemma, включая vision encoder и multimodal projector, заморожена. Проверяем список trainable tensors, gradients и побайтовые hashes frozen weights.
- SFT: официальный PyTorch forward/loss и preprocessing, gradient checkpointing, AdamW b1=.9 b2=.95 eps=1e-8 wd=1e-10, grad clip 1; LR cosine 3e-5 → 3e-6, warmup 100. Microbatch/accumulation фиксируются после проверки памяти, до длительного обучения; effective batch 8 одинаков для методов.
- Teacher: самый ранний D1 full-SFT checkpoint из 250/500/750/1000/1250 updates с ≥1 успешным эпизодом на validation seeds 22000–22009. Если ни один не проходит, residual warmup останавливается с явным failure; D0-база не заменяется.
- Residual: corrected V3 SAC/Cal-QL, shared SERL visual encoder, uniform learned std, actor Q mean, SERL softplus temperature, 2000-step optimizer warmup; D0 π₀ полностью frozen. Offline Cal-QL 1000 updates; затем 20 teacher-warmup episodes с actor/critic/temperature updates; затем ровно 50000 active online transitions, UTD=1, offline fraction .5, batch 256 (microbatch при необходимости фиксируется до запуска). Фиксированный последний checkpoint, deterministic actor evaluation; online OTF, probe_fraction=0.
- Teacher-warmup и potential shaping — явные изменения V4. Teacher и D0 имеют независимые chunk caches; D0 cache продвигается на каждом посещённом состоянии, даже когда действие исполняет teacher.
- `clip(a_D0 + xi * residual, -1, 1)` в OSC units. Предварительно xi=2: так любое bounded teacher action достижимо из любого bounded D0 action, включая противоположные gripper commands. Эта широкая область коррекции — ограничение интерпретации как «малой» residual adaptation. Фактическую достижимость проверяем до RL.
- Shaping: gamma=.99, beta=.1, state potential отражает reach, grasp, lift/transport целевой миски к тарелке. Для absorbing success и конечного горизонта Phi(next)=0. Offline/online reward и Monte Carlo returns пересчитываются одним кодом. Success всегда оригинальный LIBERO check_success.
- RL seeds: 20000–20999, циклически; offline flow seeds 1000000+demo index. Validation 22000–22009; sanity/zero-residual 23000–23002. Final 30000–30049. Наборы не пересекаются.

## Финальная оценка

Только после фиксации всех трёх финальных checkpoints: 3 метода × 2 задачи × 50 scenes = 300 rollout, horizon 220, replan 5. Для каждой задачи сохраняются generated reset states, hashes проверяются перед эпизодами. Noise определяется seed и номером replan, одинаков для всех методов. Не используем demo-associated benchmark init files.

Каждый эпизод: двухкамерное видео с terminal frame, method/task/seed/outcome, JSON и CSV. Возобновление проверяет завершённые episodes и не добавляет повтор как новое наблюдение. SR и Wilson 95% CI на шесть ячеек, paired differences и exact McNemar; N=50 на ячейку, не объединяем correlated measurements. Одна training seed на метод — ограничение.

Код, preregistration, логи и отчёты публикуются постепенно в GitHub; checkpoints, norm, replay, reset states и 300 видео — в новом HF repo `qrvmil/pi0-d0-d1-three-adaptations-20261020`, с проверкой скачивания. Видимость существующих repo не меняем.

## Уточнение реализации до первого optimizer update

Фиксируем microbatch2 × accumulation4 для всех SFT (effective batch8), float32 master weights и AdamW states, bf16 autocast только вычислений. Короткий одношаговый memory/inference profile отбрасывается; он не является общим D0 checkpoint и не входит в результаты. При OOM меняем только microbatch/accumulation при сохранении batch8, публикуя изменение до длительного обучения.

Новый путь использует официальный PyTorch forward/loss и inference transforms; старый JAX numerical contract не применяется к этим новым PyTorch weights. Все методы используют одинаковый backend и explicit NumPy-generated flow noise. Нормализация считается official RunningStats по всем aligned D0 frames и их 50-action chunks (последнее действие повторяется при padding), без отбрасывания неполного последнего batch.

Offline RL сохраняет оригинальные native images/proprioception/actions. Для state potential и подтверждения success восстанавливаются соответствующие original XML и saved simulator states (`obs[i]` соответствует `states[i+1]`; последнее post-action state получается исполнением последнего записанного action из его pre-action state). Это восстановление для диагностики и наград, не новые demonstrations. В replay входят native траектории с исходной положительной terminal reward и подтверждённым LIBERO goal после начального observation; число прошедших проверку публикуется. SFT продолжает использовать все 50/50 demonstrations.

Сохранения SFT (до начала обучения): D0 через1000 updates; D1 expert через500; D1 full через500 и на ранних 250-step teacher candidates до первого прошедшего validation checkpoint. Общие optimizer budgets не меняются. Все сохранения содержат optimizer/RNG для resume. У финальных policy weights выполняется полное скачивание с SHA256; у больших optimizer state files — точный remote LFS SHA256, размер и фактическое чтение первых/последних64KiB, чтобы не скачивать десятки GiB повторно. Receipt явно различает способы проверки.

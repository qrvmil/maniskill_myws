# Три способа адаптации π₀: D0 → D1

Это сравнение освоения D1 и сохранения D0 на двух **знакомых задачах LIBERO**. D0 — миска в центре стола; D1 — миска рядом с тарелкой. Во всех SFT использованы все 50 исходных native demonstrations соответствующей задачи. Финальная оценка — новые generated scenes этих задач; это не held-out demonstrations и не OOD.

## Что обучали

Один общий full-SFT checkpoint: официальный `pi0_base`, **3000 updates × batch8 на D0**, без LoRA. Весь model scope доступен optimizer: 777 tensors; 770 получают action-loss gradients, а семь официальных PyTorch tensors не участвуют в action loss (unused expert LM head и последние VLM-only outputs; полный список в evidence/sft_d0_gradient_scope.json). Это архитектурно неактивные outputs. Его D0-only normalization одинакова во всех методах и проверена по SHA256. Из одних и тех же D0 weights независимо:

- **Full SFT D1:** 1500 updates × batch8, все параметры; новый AdamW.
- **Action-expert SFT D1:** те же 1500 updates, данные, seed и LR schedule; обучались action expert и state/action/time projections. PaliGemma, vision encoder и multimodal projector frozen; trainable scope, gradients и hashes проверены.
- **Residual RL D1:** исходная D0 π₀ полностью frozen. Отдельные actor/critic/temperature; corrected V3 SAC и auxiliary full-action Cal-QL, V4 разделение ролей D0/D1. Offline initialization: 1000 updates на успешных native D1 demos (50/50 прошли state-based LIBERO goal check). База/replay пересчитаны. Затем 20 teacher-warmup episodes (2/20 successes, 4127 transitions сбора) с обновлениями actor/critic/temperature и ровно 50000 active online transitions. Critic UTD1, actor/temperature через2 critic steps, batch256, offline fraction .5. Всего online environment steps: 54127.

Два изменения V4 заданы **до результатов**: warmup исполняет ранний full-SFT teacher (D1 update 250, повторная validation после унификации reset 3/10), а reward содержит potential-based shaping (gamma .99, beta .1): reach нужной миски, grasp, lift/transport. Teacher не заменяет D0-базу; исполненные teacher actions и cached D0 actions записаны отдельно. Terminal Phi=0; offline/online rewards и MC returns вычисляются одним кодом. Успех всегда определяется исходным LIBERO predicate. Online OTF, probe_fraction0; evaluation deterministic actor.

Действие residual: `clip(a_D0 + 2 * residual, -1, 1)` в OSC units. xi2 обеспечивает достижимость любых bounded teacher actions, включая gripper. Это широкая correction authority, поэтому результат нельзя трактовать как проверку только малых поправок.

## Результаты

N=50 в каждой ячейке; в скобках Wilson 95% CI. Horizon220, replan5. Методы получают одинаковые сохранённые generated reset states и flow noise; hashes physics state и начальных observations проверены. До RL обнаружен различный refresh sensor caches у fresh/restored resets; оба пути унифицированы и прошли точную zero-residual проверку. Последние checkpoints выбраны по заранее фиксированному бюджету; final seeds не участвовали в выборе.

| Метод | D0 SR [95% CI] | D1 SR [95% CI] |
|---|---|---|
| Full SFT | 0/50 — 0% [0.0; 7.1] | 20/50 — 40% [27.6; 53.8] |
| Action-expert SFT | 0/50 — 0% [0.0; 7.1] | 19/50 — 38% [25.9; 51.8] |
| Residual RL | 0/50 — 0% [0.0; 7.1] | 0/50 — 0% [0.0; 7.1] |

На этом запуске наибольший измеренный D1 SR: `full`. На D0 все три политики дали 0/50; лидера по сохранению D0 в этих данных нет. Заморозка PaliGemma относительно full SFT изменила D1 SR на -2 процентных пунктов, D0 SR — на +0 п.п. После Holm correction обнаружены различия в 2 из шести парных сравнений.

Оба SFT дали 0/50 на D0: в этом запуске заморозки PaliGemma оказалось недостаточно для успешного выполнения старой задачи. Residual RL получил 0/50 на D0 и 0/50 на D1. В зафиксированном бюджете этот рецепт не показал ни одного успеха D1; ненулевого SR teacher на validation оказалось недостаточно для успешной RL-адаптации. Неизменность D0 weights сама по себе не гарантирует сохранение поведения: исполняемое действие изменяется residual actor.

Парное изменение ниже означает **второй метод минус первый**. Gained/lost — число сцен, где второй метод приобрёл/потерял успех. Интервалы — paired bootstrap 95% (10000 draws); † — при вырожденном bootstrap показана консервативная 95% граница, чтобы одинаковые исходы не создавали иллюзии точной эквивалентности. p — exact McNemar после Holm correction всех шести сравнений.

| Задача | Сравнение | Δ SR, п.п. [95% CI] | Gained/lost | p Holm |
|---|---|---:|---:|---:|
| D0 | Full SFT → Action-expert SFT | +0 [-5.8; +5.8]† | 0/0 | 1 |
| D0 | Full SFT → Residual RL | +0 [-5.8; +5.8]† | 0/0 | 1 |
| D0 | Action-expert SFT → Residual RL | +0 [-5.8; +5.8]† | 0/0 | 1 |
| D1 | Full SFT → Action-expert SFT | -2 [-10.0; +4.0] | 1/2 | 1 |
| D1 | Full SFT → Residual RL | -40 [-54.0; -26.0] | 0/20 | 1.144e-05 |
| D1 | Action-expert SFT → Residual RL | -38 [-52.0; -24.0] | 0/19 | 1.907e-05 |

[Полные paired comparisons](results/paired_comparisons.json) содержат также нескорректированные p-values и conservative bounds для вырожденного bootstrap на границе; [episode CSV](results/episodes.csv) позволяет проверить каждую пару. Эти интервалы описывают вариативность сцен при фиксированных обученных моделях.

## Как читать выводы

SFT методы сравнивают область обновляемых параметров при одинаковом D1 бюджете. RL имеет другой объём online опыта и дополнительную информацию simulator state в shaping, поэтому это сравнение трёх рецептов адаптации, а не изоляция единственного фактора. Native D1 images/proprioception/actions в offline replay сохранены; shaping geometry и goal checks восстановлены из original XML и saved states. Offline MC labels у online эпизода доступны после его завершения; промежуточные SAC updates используют TD rewards.

Одна training seed на метод и только 50 сцен на задачу ограничивают точность и переносимость выводов. Сцены общие для методов: их исходы нельзя считать независимыми между методами. D0 retention здесь — абсолютный SR трёх адаптированных политик; общий D0 checkpoint имеет только короткую отдельную zero-residual sanity-проверку, поэтому точный спад относительно базового SR на N=50 этим протоколом не оценивается. Teacher validation мала; наличие ≥1 успеха подтверждает ненулевой измеренный SR, а не надёжность teacher. D0/D1 различаются и target location, и distractor placement. Generated reset protocol отличается от официальных benchmark init files. Потенциал использует simulator state, недоступный обычному vision-only deployment; actor при исполнении получает только камеры, proprioception и действие базы.

## Материалы и воспроизводимость

- [Общий D0 checkpoint3000](https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-20261020/tree/main/checkpoints/d0/3000) и [ранний teacher250](https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-20261020/tree/main/checkpoints/teacher).
- Финальные модели с normalization: [full SFT1500](https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-20261020/tree/main/checkpoints/full/1500), [action-expert SFT1500](https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-20261020/tree/main/checkpoints/expert/1500), [residual RL50000](https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-20261020/tree/main/checkpoints/residual/final).
- [Индекс всех 300 двухкамерных видео и episode JSON](VIDEOS.md), terminal frame включён; [HF artifacts](https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-20261020/tree/main/final).
- [Generated final reset states](https://huggingface.co/qrvmil/pi0-d0-d1-three-adaptations-20261020/tree/main/reset_states/final).
- [Выполненный учебный notebook](seminar.ipynb): weights/norm → scene → observations → action → проверенный rollout → video → SR; повторное чтение не добавляет episodes.
- [Протокол](PREREGISTRATION.md), [completeness check](results/completeness.json), [SR CSV](results/success_rates.csv).

Код/прогресс сохранены в GitHub; веса, optimizer/RNG/replay и артефакты — в HF с обратным чтением. Видимость существующих repositories не менялась.

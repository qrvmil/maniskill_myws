"""Generate the readable evaluation tutorial; execute after real models exist."""
import textwrap
import nbformat as nbf
from data import ROOT


def build():
    cells=[]
    def clean(text):
        first,*rest=text.strip('\n').splitlines()
        return first.strip()+'\n'+textwrap.dedent('\n'.join(rest)).rstrip()
    def md(text): cells.append(nbf.v4.new_markdown_cell(clean(text)))
    def code(text): cells.append(nbf.v4.new_code_cell(clean(text)))
    completed=(ROOT/'docs/turbovla_multitask/results.json').exists()
    status='Результаты трёх моделей доступны ниже.' if completed else '**Черновик: обучение продолжается. Полный запуск notebook будет проверен после появления всех трёх моделей.**'
    md(f'''# TurboVLA: загружаем модель и оцениваем её в LIBERO

    {status}

    Три независимых запуска: **A → D0**, **B → D0+D1**, **C → D0+D1+D2**.
    Каждый начинает обучение из одних и тех же исходных весов DINOv3/BERT/GroundingDINO.
    H1/H2 не участвуют ни в обучении, ни в расчёте статистик. В готовом чекпоинте
    сохранена вся TurboVLA; для оценки заранее выбран EMA-вариант весов.

    Здесь мы выбираем модель, загружаем её, подключаем LIBERO, выполняем
    **15 эпизодов одной выбранной задачи** и считаем успехи. Затем показываем
    полную матрицу A/B/C × D0/D1/D2/H1/H2 из основного эксперимента.''')
    md('''## 1. Подготовить окружение

    Используйте ядро **TurboVLA multitask (Python 3.10)**. Команды ниже выполняются
    в терминале из корня репозитория, если окружение ещё не подготовлено:

    ```bash
    export HF_HOME=/workspace/.hf_home
    bash turbovla_multitask/setup.sh
    ```

    Полный чекпоинт уже содержит обученные DINOv3 и BERT. Ниже восстановим
    их локальные файлы прямо из него; отдельно скачиваются только файлы
    токенизатора BERT. Доступ к gated-репозиторию Meta не требуется.
    Для скачивания **наших** моделей из приватного HF-репозитория нужен ваш
    действующий HF-токен. Токен в notebook не записываем.''')
    code('''import os
    import sys
    import json
    from pathlib import Path

    os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    # Notebook можно открыть из корня репозитория или из каталога notebooks.
    here = Path.cwd().resolve()
    REPO = next(p for p in [here, *here.parents] if (p / "turbovla_multitask").is_dir())
    sys.path.insert(0, str(REPO / "turbovla_multitask"))
    sys.path.insert(0, str(REPO / "turbovla_eval"))

    import torch
    torch.set_num_threads(4)
    import pandas as pd
    import numpy as np
    from IPython.display import display, Image, Video
    from data import WORK, TASKS, TRAIN_SETS, HF_REPO
    from prepare import sha256, write_json
    from eval_utils import configure, make_env, reset_episode, run_episode
    from report import summarize_cell

    print("PyTorch:", torch.__version__)
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")''')
    md('''## 2. Выбрать модель и задачу

    D0/D1 — две пространственные версии переноса чёрной миски на тарелку;
    D2 — миска на плиту. H1 — перенос миски рядом с ramekin; H2 — суп в корзину.
    Сравниваются одинаковые seeds 10000–10014. Для Spatial горизонт 220, для
    Goal/Object 280. Это протокол предыдущего pi0-эксперимента, а не официальный
    LIBERO-протокол с готовыми initial states и Goal-горизонтом 300.''')
    code('''VARIANT = "A"             # "A", "B" или "C"
    TASK = "D0"               # "D0", "D1", "D2", "H1" или "H2"
    EPISODES = 15
    SEED_START = 10000
    DOWNLOAD_FROM_HF = True
    RUN_NEW_ROLLOUTS = True    # False: прочитать 15 эпизодов из основного эксперимента

    display(pd.DataFrame([
        {"Модель": v, "Обучение": "+".join(tasks), "Эпизодов на eval-задачу": 15}
        for v, tasks in TRAIN_SETS.items()
    ]))
    print("Выбрано:", VARIANT, TASK)
    print("Инструкция:", TASKS[TASK]["name"].replace("_", " "))
    print("Задача была в обучении:", TASK in TRAIN_SETS[VARIANT])''')
    md('''## 3. Получить чекпоинт и его статистики

    Модель и нормализация должны относиться к **одному запуску**. Мы используем
    точную опубликованную HF-ревизию, проверяем SHA256 файлов и не подставляем
    статистики от готовой модели авторов, обученной на всём LIBERO.''')
    code('''from artifacts import require_final

    revision = None
    if DOWNLOAD_FROM_HF:
        publication_path = REPO / "docs/turbovla_multitask/publications" / f"{VARIANT}.json"
        publication = json.loads(publication_path.read_text())
        revision = publication["revision"]
        from huggingface_hub import hf_hub_download
        checkpoint = Path(hf_hub_download(HF_REPO, f"{VARIANT}/final.pth", revision=revision))
        stats_path = Path(hf_hub_download(HF_REPO, f"{VARIANT}/stats.json", revision=revision))
        complete_path = Path(hf_hub_download(HF_REPO, f"{VARIANT}/training_complete.json", revision=revision))
    else:
        checkpoint = WORK / VARIANT / "train/final.pth"
        stats_path = WORK / VARIANT / "stats.json"
        complete_path = WORK / VARIANT / "train/complete.json"

    metadata = json.loads(complete_path.read_text())
    require_final(metadata, VARIANT)
    assert sha256(checkpoint) == metadata["checkpoint_sha256"]
    assert sha256(stats_path) == metadata["stats_sha256"]
    print("Источник:", f"HF {HF_REPO}@{revision}" if DOWNLOAD_FROM_HF else "Локальные проверенные веса")
    print("Обновлений:", metadata["step"], "Веса:", metadata["primary_weights"])''')
    md('''### Восстановить энкодеры из полного чекпоинта

    Это энкодеры выбранной обученной модели. Исходные веса Meta и GroundingDINO
    для оценки не скачиваем. Проверка strict load и хешей сохраняет соответствие
    выбранному EMA-чекпоинту; отдельный каталог предотвращает смешение моделей.''')
    code('''from prepare_eval_assets import prepare_policy_assets

    dinov3_directory, bert_directory = prepare_policy_assets(checkpoint, WORK / "eval_assets")
    print("Энкодеры восстановлены из выбранного чекпоинта.")''')
    md('''## 4. Загрузить полную TurboVLA

    Внутри модели — DINOv3 ViT-B/16, BERT, модуль взаимодействия vision-language
    и ACT-декодер. Класс ниже загружает весь сохранённый state dict строго,
    выбирает EMA, применяет статистики именно этой модели. Параметры FP32;
    вычисления DINOv3 используют BF16 autocast, как в обучении.''')
    code('''configure()
    from turbovla.evaluation.suite_policy import TurboVLAPolicy

    policy = TurboVLAPolicy(
        ckpt_path=str(checkpoint),
        dinov3_path=str(dinov3_directory),
        bert_path=str(bert_directory),
        stats_path=str(stats_path),
        stats_key="train",
        precision="fp32",
        allow_hf_download=False,
        verbose=False,
    )
    print("Модель загружена:", VARIANT, "action chunk:", policy.chunk_size)''')
    md('''## 5. Подключить LIBERO и посмотреть наблюдение

    `make_env` находит задачу по точному имени, проверяет её BDDL и создаёт
    256×256 камеры. После reset выполняем10 нейтральных действий для стабилизации.
    Для модели изображения поворачиваются180°; ниже показываем тот же поворот.''')
    code('''import matplotlib.pyplot as plt
    from turbovla.evaluation.policy import rotate_libero_image

    env, task, task_id, suite, horizon = make_env(TASK, protocol="reference")
    try:
        observation, reset_info = reset_episode(env, SEED_START)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        for ax, camera, title in zip(axes,
            ["agentview_image", "robot0_eye_in_hand_image"], ["Внешняя камера", "Камера на руке"]):
            ax.imshow(rotate_libero_image(observation[camera]))
            ax.set_title(title)
            ax.axis("off")
        plt.show()
        chunk = policy.predict_env_action_chunk_from_obs(observation, task.language, execute_steps=12)
        assert chunk.shape == (12, 7) and np.isfinite(chunk).all()
        print("Инструкция:", task.language)
        print("Chunk:", chunk.shape, "горизонт:", horizon)
    finally:
        env.close()''')
    md('''## 6. Выполнить15 эпизодов

    Один эпизод: reset →10 settling steps → наблюдение → прогноз12 действий →
    исполнение chunk → новое наблюдение. Успех берём из `env.check_success()`.
    `run_episode` содержит именно этот цикл; его код находится в
    `turbovla_eval/eval_utils.py`. Видео сохраняет обе камеры.

    При новом запуске создаётся отдельный каталог: предыдущие результаты
    основного эксперимента сохраняются.''')
    code('''from datetime import datetime, timezone

    if RUN_NEW_ROLLOUTS:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = WORK / VARIANT / "notebook_evaluations" / run_name
        output.mkdir(parents=True)
        env, task, task_id, suite, horizon = make_env(TASK, protocol="reference")
        rows = []
        try:
            for episode in range(EPISODES):
                result = run_episode(
                    env, policy, task.language, seed=SEED_START + episode,
                    horizon=horizon, video_path=output / "videos" / f"episode_{episode:03d}.mp4",
                )
                result.update(variant=VARIANT, task=TASK, task_id=task_id, episode=episode,
                              seen_during_training=TASK in TRAIN_SETS[VARIANT])
                rows.append(result)
                write_json(output / "episodes.json", rows)
                print(f"{episode + 1}/{EPISODES}: success={result['success']}, steps={result['steps']}")
        finally:
            env.close()
        write_json(output / "provenance.json", {
            "checkpoint_sha256": metadata["checkpoint_sha256"],
            "stats_sha256": metadata["stats_sha256"], "hf_revision": revision,
            "variant": VARIANT, "task": TASK, "protocol": "reference",
        })
    else:
        output = WORK / VARIANT / "eval15"
        rows = json.loads((output / TASK / "episodes.json").read_text())''')
    md('''## 7. Посчитать успехи и неопределённость

    SR = число успешных эпизодов /15. Wilson95% CI показывает неопределённость
    оценки при небольшом числе запусков; один успех меняет SR на6.67 процентных
    пункта. Проверяем состав эпизодов, seeds, горизонты и действия перед расчётом.''')
    code('''summary = summarize_cell(rows, VARIANT, TASK)
    low, high = summary["ci95"]
    print(f"{summary['successes']}/15 = {summary['success_rate']:.1%}")
    print(f"Wilson95% CI: {low:.1%}–{high:.1%}")
    display(pd.DataFrame([
        {"Seed": r["seed"], "Успех": r["success"], "Шагов": r["steps"]} for r in rows
    ]))''')
    md('''## 8. Посмотреть видео эпизода

    Выберите индекс 0–14. Видео и JSON сохраняются локально рядом с запуском.
    Это наблюдение конкретного эпизода, а не доказательство качества всей модели.''')
    code('''EPISODE_TO_WATCH = 0
    video_path = Path(rows[EPISODE_TO_WATCH]["video"])
    if not video_path.is_absolute():
        video_path = REPO / "turbovla_eval" / video_path
    display(Video(str(video_path), embed=True, width=640))''')
    md('''## 9. Полная матрица основного эксперимента

    Ниже 3 модели × 5 задач × 15 эпизодов = 225 эпизодов. Чекпоинты заранее
    фиксированы на 3001 обновлении; H1/H2 не использовались для их выбора.
    Матрица пересчитана из episode-level JSON. `seen` означает участие задачи
    в обучении данной модели; H1/H2 всегда `unseen`.''')
    code('''matrix_path = REPO / "docs/turbovla_multitask/generalization_matrix.csv"
    matrix = pd.read_csv(matrix_path)
    assert len(matrix) == 15 and (matrix["episodes"] == 15).all()
    display(matrix.assign(
        SR=matrix["success_rate"].map(lambda x: f"{x:.1%}"),
        CI=matrix.apply(lambda r: f"{r.ci95_low:.1%}–{r.ci95_high:.1%}", axis=1),
    )[["train_set", "task", "seen", "successes", "episodes", "SR", "CI"]])
    display(Image(filename=str(REPO / "docs/turbovla_multitask/generalization_matrix.png")))''')
    md('''## 10. Как повторить оценку другой модели

    Измените `VARIANT` и `TASK` в разделе 2, затем выполните notebook сверху вниз.
    Для пакетной оценки всех пяти задач:

    ```bash
    /workspace/turbovla-venv/bin/python turbovla_multitask/evaluate.py \
      --variant B --episodes 15 --output /workspace/turbovla-multitask/B/new_eval15
    ```

    При сравнении с pi0 учитывайте: здесь официальные 256px no-noops данные
    (46/47/48 демонстраций), у pi0 — 50 исходных демонстраций и другая обработка
    изображений. Наш бюджет 3001 обновление × batch 8 также отличается от
    авторского обучения TurboVLA на всех четырёх LIBERO suites (40k × batch 128).
    Эти результаты отвечают на вопрос о наших вложенных обучающих наборах;
    они не воспроизводят заявленные авторами 97.6%.''')
    notebook=nbf.v4.new_notebook(cells=cells,metadata={
        'kernelspec':{'display_name':'TurboVLA multitask (Python 3.10)','language':'python','name':'turbovla-multitask'},
        'language_info':{'name':'python','version':'3.10'},
    })
    nbf.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code': compile(cell.source,'notebook_cell','exec')
    path=ROOT/'notebooks'/'02_turbovla_multitask_eval.ipynb'
    nbf.write(notebook,path)
    print('NOTEBOOK_WRITTEN',path,'completed_results=',completed)


if __name__=='__main__': build()

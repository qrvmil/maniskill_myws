"""Build Russian seminar notebooks using nbformat; executed outputs saved separately."""
from pathlib import Path
import ast
import nbformat as nbf
ROOT=Path(__file__).resolve().parent

def source_function(name):
    source=(ROOT/'eval_utils.py').read_text()
    tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    return ast.get_source_segment(source,node)

def notebook(cells,path):
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'turbovla-libero','display_name':'TurboVLA LIBERO (Python 3.10)','language':'python'},'language_info':{'name':'python','version':'3.10'}})
    nbf.validate(nb);nbf.write(nb,ROOT/path)

md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
intro='''# TurboVLA на пяти задачах LIBERO: практический семинар

## Цель
Пошагово загрузить **опубликованный TurboVLA без нового дообучения**, получить наблюдение симулятора, предсказать действия и измерить успех на D0, D1, D2, H1, H2. Видео лежат отдельно в `results/<run>/videos/<task>/`, JSON — в `results/<run>/<task>/`.

**Важное различие:** H1/H2 были исключены из SFT ваших вариантов π₀. TurboVLA обучен совместно на 40 задачах четырёх наборов LIBERO, включая все наши пять. Его результаты на H1/H2 показывают качество опубликованной модели на этих задачах, **не zero-shot OOD-обобщение**.

Первый ноутбук — лабораторная работа с настоящим коротким прогоном. Второй (`02_results_and_vram.ipynb`) содержит полный эвал и измерения RTX 4090. Выполняйте ячейки сверху вниз; после изменения параметров можно перезапустить kernel.

Источники: [статья, версия v3](https://arxiv.org/pdf/2607.27205v3), [официальный код](https://github.com/H-EmbodVis/TurboVLA), [checkpoint](https://huggingface.co/H-EmbodVis/TurboVLA), [исходный π₀-ноутбук](https://github.com/qrvmil/maniskill_myws/blob/exp/pi0-multitask-sft-generalization/notebooks/01_multitask_sft_eval.ipynb). Точные имена/горизонты: `src/maniskill_myws/pld/multitask_protocol.py` в указанной ветке. Ревизии и SHA256 сохраняются рядом с результатами.
'''
cells=[md(intro),md('''## Подготовка

### 1. Окружение и kernel
На этой машине окружение уже подготовлено. На новой Linux-машине с NVIDIA GPU выполните `bash setup.sh` из папки этого ноутбука, затем выберите kernel **TurboVLA LIBERO (Python 3.10)**. Скрипт создаёт `.venv`, ставит PyTorch 2.3.1/cu121 и фиксированные версии симулятора, скачивает исходники и веса, регистрирует kernel. Системный драйвер NVIDIA не меняется.

Мы используем MuJoCo 3.2.7 / robosuite 1.4.1 и LIBERO revision `8f1084e…`, как в исходном эксперименте. TensorFlow/RLDS для инференса **не нужны**: они нужны для обучения на демонстрациях. Для эвала достаточно BDDL-сцен, ассетов симулятора, весов, tokenizer и статистик нормализации.

Не запускайте два GPU-ноутбука одновременно во время измерения VRAM. Эвал работает локально; поднимать сервер или открывать порт не требуется.
'''),code('''from pathlib import Path
import os, sys, json, time, hashlib
ROOT = Path.cwd()
assert (ROOT / "eval_utils.py").exists(), "Откройте notebook из папки turbovla_eval"
sys.path.insert(0, str(ROOT))
import eval_utils
from eval_utils import TASKS, ASSETS, configure, write_json
configure()  # MUJOCO_GL=egl, локальные пути LIBERO, отключение TensorFlow
import numpy as np
import torch
import pandas as pd
from IPython.display import display, Video
assert torch.cuda.is_available(), "Нужен CUDA kernel из .venv"
torch.set_num_threads(2)
print("Python:", sys.version.split()[0], "| torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name())
'''),md('''### 2. Параметры лабораторной работы

Для знакомства запускаем **один эпизод D0**. Для полноценного нового эвала задайте все пять задач и `N_EPISODES=50`. Результаты уже выполненного полного прогона читаются во втором ноутбуке. Каждое выполнение создаёт новую папку, чтобы не переписать предыдущий эксперимент.

`reference`: сброс по seed 10000–10049, 10 settling-шагов, MSAA выключен, горизонт D2=280 — как в π₀-эксперименте. Изображения TurboVLA остаются 256×256, chunk/replan=12; у π₀ replan был 5, поскольку это другая политика.

`official`: benchmark initial states 0–49, seed 7, D2=300, штатный рендеринг. Это вариант для сопоставления с протоколом статьи. Имена пяти задач одинаковы в обоих режимах; номера состояний и seed — разные сущности.
'''),code('''SELECTED_TASKS = ["D0"]  # полный набор: ["D0", "D1", "D2", "H1", "H2"]
N_EPISODES = 1             # основной эвал: 50
SEED_START = 10000
PROTOCOL = "reference"    # или "official"
SAVE_VIDEOS = True
RUN_NAME = time.strftime("tutorial_%Y%m%d_%H%M%S")
OUTPUT = ROOT / "results" / RUN_NAME
PRECISION = "bf16"
display(pd.DataFrame(TASKS).T[["suite", "name", "horizon"]])
'''),md('''### 3. Подготовка весов и данных

Checkpoint `turbovla_libero.pth` содержит **полную** сеть: BERT, DINOv3 ViT-B, cross-attention и action head. Отдельный DINOv3 Hub требует авторизации, поэтому `prepare_assets.py` извлекает уже опубликованные тензоры BERT/DINOv3 из этого полного checkpoint в локальные папки. В конечной политике все тензоры снова загружаются с `strict=True`; случайные начальные веса не остаются.

Архитектура DINOv3 ViT-B/16 берётся из открытого определения Meta: 768 признаков, 12 слоёв/голов, четыре register token, RoPE base=100. Размеры сверяются с тензорами. Препроцессинг использует штатные defaults TurboVLA (ImageNet mean/std, /255, **без resize/crop**); это записано в `assets/preparation.json`. BERT tokenizer — публичный `google-bert/bert-base-uncased`.

Расположение файлов: `assets/dinov3`, `assets/bert`, `assets/turbovla_libero.pth`, `assets/libero_all4_stats.json`. Нормализация берётся от TurboVLA, а не от π₀. Демонстрации, дообучение и пересчёт статистик для этих пяти задач не требуются.

На новой машине `setup.sh` вызывает `download_assets.py`, `prepare_assets.py` и `prepare_observation.py` (сохраняет настоящее D0-наблюдение для VRAM-теста в отдельном процессе). Ниже проверяем подготовленный набор.
'''),code('''required = ["turbovla_libero.pth", "libero_all4_stats.json", "dinov3/config.json", "dinov3/model.safetensors", "bert/config.json", "bert/model.safetensors", "bert/tokenizer.json"]
assert all((ASSETS / name).is_file() for name in required)
preparation = json.loads((ASSETS / "preparation.json").read_text())
print("Checkpoint SHA256:", preparation["checkpoint_sha256"])
display(pd.DataFrame(preparation["backbones"]).T)
statistics = json.loads((ASSETS / "libero_all4_stats.json").read_text())["libero_all4_no_noops"]
display(pd.DataFrame({"action_min": statistics["action"]["min"], "action_max": statistics["action"]["max"]}))
'''),md('''## Шаги

### 4. Загружаем модель
Политика строит архитектуру из `model_config` checkpoint, проверяет полное совпадение ключей и размеров, переводит **все параметры в BF16**, включает `eval()` и отключает градиенты. Никакого обучения в этом ноутбуке нет. CUDA память для весов и активаций выделяется здесь.
'''),code(source_function('load_policy')+'\n\npolicy = load_policy(PRECISION)\nprint("Параметров:", f"{sum(p.numel() for p in policy.model.parameters()):,}")\nprint("Память тензоров сейчас, GiB:", round(torch.cuda.memory_allocated()/1024**3,3))'),md('''### 5. Подключаемся к LIBERO и проверяем выбранную задачу
Это прямой локальный вызов Python API. Никакой REST endpoint не нужен. По имени выбираем задачу в suite, проверяем BDDL SHA256 и создаём `OffScreenRenderEnv`. BDDL определяет объекты, размещение и условие успеха. Номер task может зависеть от порядка задач, поэтому фиксируем `task_order_index=0` и связываем всё по имени.

Физика и CUDA-инференс — разные компоненты: симулятор возвращает два RGB-изображения и состояние робота; policy возвращает действия для его контроллера.
'''),code('''from eval_utils import make_env, reset_episode, video_frame
TASK = SELECTED_TASKS[0]
env, task, task_id, suite, horizon = make_env(TASK, PROTOCOL)
initial_state = suite.get_task_init_states(task_id)[0] if PROTOCOL == "official" else None
obs, reset_info = reset_episode(env, SEED_START if PROTOCOL == "reference" else 7, initial_state)
print("Task ID:", task_id, "| instruction:", task.language, "| horizon:", horizon)
print("Settled reset hash:", reset_info["reset_hash"])
print({key: np.asarray(obs[key]).shape for key in ["agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]})
'''),md('''### 6. Наблюдение → входные тензоры

Кадры симулятора поворачиваются на 180° (`[::-1, ::-1]`), как в официальном адаптере. Состояние: позиция end effector (3), axis-angle ориентация (3), позиции пальцев (2) = **8 чисел**. Quaternion нельзя просто передать вместо axis-angle.

Два RGB-кадра нормализуются ImageNet mean/std; state стандартизуется `(state-mean)/(std+1e-6)` по статистикам всех четырёх наборов. Инструкция идёт в BERT онлайн. Не нужен кэш текстовых эмбеддингов.
'''),code('''from turbovla.evaluation.policy import rotate_libero_image, state_from_libero_obs
primary = rotate_libero_image(obs["agentview_image"])
wrist = rotate_libero_image(obs["robot0_eye_in_hand_image"])
state = state_from_libero_obs(obs)
samples, states = policy._build_batch([primary], [wrist], [state])
samples, states = policy._prepare_model_inputs(samples, states)
print("images:", samples["dinov3"].shape, samples["dinov3"].dtype)
print("state:", states.shape, states.dtype)
from PIL import Image
display(Image.fromarray(np.concatenate([primary,wrist],axis=1)))
'''),md('''### 7. Один forward и обратная нормализация действий

Сеть выдаёт `[batch=1, chunk=12, action_dim=7]`. Первые 6 координат — нормированные смещения/вращения OSC-контроллера; последняя — команда gripper. Адаптер один раз выполняет min–max inverse transform и sign rule для gripper. Поэтому передавать нормированный output сети прямо в env или повторно денормализовать — ошибка.

Проверяем настоящий CUDA forward и конечность значений. Пока это только предсказание: робот ещё не исполнял chunk.
'''),code('''with torch.inference_mode():
    predicted_normalized = policy.model([task.language], samples, states)
assert predicted_normalized.shape == (1,12,7)
assert torch.isfinite(predicted_normalized).all()
action_chunk = policy.predict_env_action_chunk_from_obs(obs, task.language, execute_steps=12)
assert action_chunk.shape == (12,7) and np.isfinite(action_chunk).all()
display(pd.DataFrame(action_chunk, columns=["dx","dy","dz","rx","ry","rz","gripper"]).head())
env.close()  # освободим rendering context перед новым эпизодом
'''),md('''### 8. Понятный цикл одного эпизода

Ниже весь rollout-код открыт для редактирования. После reset даём физике 10 шагов успокоиться. Затем кладём 12 предсказанных действий в очередь и исполняем по одному. Новое наблюдение используется для нового forward, когда очередь пуста. `env.check_success()` задаёт успех; истечение horizon означает failure. Ошибка модели/симулятора прерывает прогон и не превращается в искусственный failure.

Видео записывается потоково: начальный кадр + кадр после каждого действия, включая терминальный. Слева внешняя камера, справа wrist; 20 fps. Очередь создаётся заново при каждом reset.
'''),code('''from collections import deque
'''+source_function('run_episode')+'''
# evaluate() ниже использует именно эту редактируемую функцию.
eval_utils.run_episode = run_episode
'''),md('''### 9. Запускаем выбранные задачи

`evaluate` создаёт env для каждой задачи, запускает эпизоды, сохраняет `episodes.json` после каждого шага эксперимента и закрывает env в `finally`. Политика загружается один раз. JSON хранит действия, success, seed, reset hash, длину и связь с видео; `summary.json` — число успехов и Wilson 95% CI.

Для эксперимента с новыми параметрами меняйте конфигурационную ячейку, затем Run All. После аварии частичные записи остаются; используйте новую папку для повторного запуска.
'''),code('''summaries = eval_utils.evaluate(policy, OUTPUT, SELECTED_TASKS, N_EPISODES,
    seed_start=SEED_START, protocol=PROTOCOL, save_videos=SAVE_VIDEOS)
display(pd.DataFrame(summaries).T)
print("Результаты:", OUTPUT)
print("Видео:", OUTPUT / "videos")
'''),md('''## Проверки

### 10. Просмотр видео и проверка количества эпизодов
Один эпизод — проверка подключения, **не** надёжная оценка качества. Для сравнения используйте 50 и интервалы неопределённости во втором ноутбуке. Видео следует смотреть и при success, и при failure; удобный ролик не заменяет статистику всех эпизодов.
'''),code('''for key in SELECTED_TASKS:
    rows = json.loads((OUTPUT / key / "episodes.json").read_text())
    assert len(rows) == N_EPISODES
    assert all(row["steps"] <= row["horizon"] for row in rows)
    assert len({row["seed"] for row in rows}) == N_EPISODES
videos = sorted((OUTPUT / "videos").rglob("*.mp4"))
if videos:
    display(Video(str(videos[0]), embed=True, width=768))
print("Эпизодов проверено:", N_EPISODES * len(SELECTED_TASKS))
'''),md('''## Дальше

### 11. Как выбрать другую задачу LIBERO
Посмотрите список ниже. Для одной из существующих пяти задач достаточно изменить `SELECTED_TASKS`. Для другой стандартной задачи добавьте запись с **точным** именем в `TASKS`, задайте suite и horizon, выберите новый ключ. Новая задача не имеет замороженного BDDL-хэша вашего π₀-протокола; он проверяется автоматически для D0–H2.

Пример (не выполняется автоматически):
```python
TASKS["NEW"] = dict(suite="libero_object", name="pick_up_the_cream_cheese_and_place_it_in_the_basket", horizon=280)
SELECTED_TASKS = ["NEW"]
```
Сверьте имя со списком, затем выполните ячейки 9–10 с новым `OUTPUT`. Для suite `libero_90` опубликованная модель не заявляет того же обучающего покрытия; основной helper официального режима рассчитан на четыре стандартных suite.
'''),code('''display(pd.DataFrame(eval_utils.list_tasks("libero_object")))
# Освобождаем CUDA перед отдельным измерением VRAM во втором ноутбуке.
del policy, samples, states, predicted_normalized
import gc
gc.collect()
torch.cuda.empty_cache()
''')]
# Keep simulator setup and reset implementation visible and editable in the seminar.
index=next(i for i,c in enumerate(cells) if c.cell_type=='code' and 'env, task, task_id, suite, horizon = make_env' in c.source)
cells[index].source=cells[index].source.replace('from eval_utils import make_env, reset_episode, video_frame','from eval_utils import video_frame')
cells[index:index]=[
md("""**Создание среды — открытый код.** `benchmark` даёт список задач; `get_libero_path` находит локальные BDDL; `OffScreenRenderEnv` создаёт симуляцию с двумя offscreen-камерами. Проверка хэша защищает точное соответствие пяти зарегистрированных сцен. Для reference-mode отдельно фиксируем horizon и отключаем MSAA, как в исходном эксперименте. Официальный режим оставляет штатные параметры рендеринга.
"""),
code('from eval_utils import BDDL_HASHES, file_hash, find_task\n'+source_function('make_env')+'\n\neval_utils.make_env = make_env'),
md("""**Сброс эпизода — открытый код.** В режиме reference seed определяет размещение объектов; в official мы дополнительно задаём сохранённый benchmark initial state. Затем исполняем 10 нулевых arm-действий с открытым gripper. Хэш состояния после этих шагов нужен для проверки парности с π₀. `seed=None` сохраняет RNG-поток official-mode между эпизодами; там seed 7 устанавливается один раз на задачу.
"""),
code(source_function('reset_episode')+'\n\neval_utils.reset_episode = reset_episode')]
index=next(i for i,c in enumerate(cells) if c.cell_type=='markdown' and '### 4. Загружаем' in c.source)
cells[index:index]=[
md("""### Что находится внутри checkpoint
В этой ячейке загружаем файл **в RAM**, смотрим словарь и конфигурацию. `model_state_dict` содержит 672 тензора; префиксы разделяют текстовый encoder, визуальный encoder и обученные блоки политики. Это объясняет, почему не нужно отдельно скачивать исходные gated DINOv3-веса: финальные веса уже находятся в release. Локальные backbone-папки служат инициализации штатного loader, затем loader строго загружает полный state dict.

Для получения файлов на новой машине шаг загрузки выглядит так (его выполняет `download_assets.py`):
```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("H-EmbodVis/TurboVLA", "checkpoints/libero/turbovla_libero.pth",
                       revision="9cdb576ed9c56039b9c1149142ed654e876bad76")
```
Шаг `prepare_assets.py` извлекает веса по префиксам и проверяет оба backbone с `strict=True`; он выполняется один раз при setup, отдельно от измерения VRAM.
"""),
code("""checkpoint = torch.load(ASSETS / "turbovla_libero.pth", map_location="cpu", weights_only=False)
print("Checkpoint keys:", list(checkpoint))
weights = checkpoint["model_state_dict"]
print("State-dict tensors:", len(weights))
print("Vision config:", checkpoint["model_config"]["vision"])
print("Action config:", checkpoint["model_config"]["action"])
assert any(key.startswith("vision_encoder.backbone.") for key in weights)
assert any(key.startswith("text_encoder.bert.") for key in weights)
del checkpoint, weights  # не держим вторую CPU-копию без необходимости
""")]
notebook(cells,'01_turbovla_step_by_step.ipynb')

cells2=[md('''# TurboVLA: полный эвал и проверка VRAM на RTX 4090

## Цель и краткий результат
Здесь читаются сохранённые результаты полного прогона на D0/D1/D2/H1/H2 и независимого измерения памяти. Значения строятся из JSON, а не вручную. Все пять задач входили в обучение опубликованного checkpoint; H1/H2 являются held-out только для вариантов π₀ из исходного эксперимента.

Полные результаты: `results/reference_n50` (основное сопоставление с seed-протоколом π₀) и, если выполнен, `results/official_n50` (стандартные initial states). Пять задач не заменяют полный четырёх-suite benchmark статьи; среднее по нашим пяти нельзя выдавать за заявленные 97,6%.

Источники: [статья v3, приложение B.2](https://arxiv.org/pdf/2607.27205v3), [TurboVLA](https://github.com/H-EmbodVis/TurboVLA), [веса и сведения об обучении](https://huggingface.co/H-EmbodVis/TurboVLA), [π₀-протокол](https://github.com/qrvmil/maniskill_myws/blob/exp/pi0-multitask-sft-generalization/src/maniskill_myws/pld/multitask_protocol.py).
'''),md('''## Подготовка
Открыть в kernel TurboVLA LIBERO; этот ноутбук читает результаты на CPU. Новое измерение VRAM запускается отдельным процессом из `.venv`, чтобы память старого Jupyter kernel не попала в оценку. Сначала завершите GPU kernel первого ноутбука.
'''),code('''from pathlib import Path
import json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Video, Markdown
ROOT = Path.cwd()
sys.path.insert(0,str(ROOT))
from eval_utils import TASKS, summarize
plt.rcParams.update({"figure.figsize": (9,4), "font.size":11, "axes.spines.top":False, "axes.spines.right":False})
RESULTS = ROOT / "results"
PROTOCOLS = [name for name in ["reference_n50", "official_n50"] if (RESULTS / name / "complete.json").is_file()]
assert "reference_n50" in PROTOCOLS, "Сначала выполните полный run_eval.py"
'''),md('''## Метод

### 1. Что сравниваем
Один и тот же checkpoint и BF16, две камеры 256×256, BERT онлайн, chunk=12, open-loop=12. `reference`: seed 10000–10049, ten settling steps, MSAA off, горизонты 220/280; `official`: initial state index 0–49, seed 7 один раз на задачу (RNG-поток продолжается между эпизодами), Goal horizon=300. Это два разных распределения reset. У π₀ также был другой replan=5 и своя нормализация. Поэтому различие success rate нельзя приписывать только архитектуре или размеру SFT.

Реальный двухэпизодный smoke-тест D0 дал побайтовое совпадение всех действий нашего официального режима с неизменённым `vla_adapter.rollout._run_episode`; запись — `results/official_parity/upstream_action_parity.json`. Это проверка двух эпизодов одной задачи, не доказательство совпадения всех 2000 эпизодов авторов.

Успех проверяется симулятором; denominator включает все 50 завершённых эпизодов, timeout — failure. Wilson 95% CI оценивает биномиальную неопределённость внутри выбранного reset-распределения; это не разброс обучения модели. Между задачами веса одинаковые.
'''),code('''records=[]
for protocol in PROTOCOLS:
    for key in TASKS:
        rows=json.loads((RESULTS/protocol/key/"episodes.json").read_text())
        assert len(rows)==50 and len({r["seed"] for r in rows})==50
        summary=summarize(rows)
        saved=json.loads((RESULTS/protocol/key/"summary.json").read_text())
        assert summary==saved
        records.append(dict(protocol=protocol,task=key,**summary))
table=pd.DataFrame(records)
table["success_percent"]=table.success_rate*100
table["Wilson 95%, %"]=table.ci95.map(lambda ci: f"{ci[0]*100:.1f}–{ci[1]*100:.1f}")
display(table[["protocol","task","successes","episodes","success_percent","Wilson 95%, %","mean_steps"]].round(2))
'''),md('''## Результаты

### 2. Успех по задачам
Столбики начинаются с нуля; вертикальные интервалы — Wilson 95%. Даже 50/50 не означает истинную вероятность ровно 100%. Mean ниже относится только к выбранным пяти задачам.
'''),code('''fig,axes=plt.subplots(1,len(PROTOCOLS),figsize=(6*len(PROTOCOLS),4),squeeze=False)
for ax,protocol in zip(axes[0],PROTOCOLS):
    subset=table[table.protocol==protocol]
    values=subset.success_rate.to_numpy()*100
    ci=np.array(subset.ci95.tolist())*100
    ax.bar(subset.task,values,color="#23738b",width=.6)
    ax.errorbar(np.arange(len(values)),values,yerr=np.maximum(0,[values-ci[:,0],ci[:,1]-values]),fmt="none",color="black",capsize=4)
    for i,value in enumerate(values): ax.text(i,ci[i,1]+2,f"{int(subset.iloc[i].successes)}/50",ha="center")
    ax.set(ylim=(0,110),ylabel="Успешные эпизоды, %",title=protocol)
    ax.grid(axis="y",alpha=.2)
fig.tight_layout();display(fig);plt.close(fig)
for protocol in PROTOCOLS:
    subset=table[table.protocol==protocol]
    print(protocol, "Среднее по пяти задачам:",f"{subset.success_percent.mean():.1f}%")
'''),md('''### 3. Видео для всех задач
Все эпизоды обоих полных прогонов записаны в отдельную директорию. Ниже автоматически выбираем первый success и первый failure, если они есть. Отсутствие failure-видео при 50/50 обозначает отсутствие failures в выборке, а не ошибку сохранения. MP4 sidecar JSON содержит seed, длину и reset hash.
'''),code('''video_records=[]
for key in TASKS:
    rows=json.loads((RESULTS/"reference_n50"/key/"episodes.json").read_text())
    for outcome in [True,False]:
        candidates=[r for r in rows if r["success"]==outcome]
        if candidates:
            row=candidates[0]
            path=ROOT/row["video"]
            assert path.is_file()
            video_records.append(dict(task=key,outcome="success" if outcome else "failure",seed=row["seed"],path=str(path.relative_to(ROOT))))
display(pd.DataFrame(video_records))
if video_records:
    display(Video(str(ROOT/video_records[0]["path"]),embed=True,width=768))
'''),md('''### 4. Проверка совпадения reset с исходным π₀-экспериментом
Одинаковые числа seed ещё не доказывают одинаковые физические состояния. Для каждого seed сравниваем SHA256 состояния после settling с сохранённым π₀ `reset_hash`, где исходные записи доступны. Это проверка физического reset, не побайтового совпадения изображений между разными GPU/рендерерами.
'''),code('''audit_path=RESULTS/"reference_reset_audit.json"
if audit_path.exists():
    reset_audit=json.loads(audit_path.read_text())
    display(pd.DataFrame(reset_audit["tasks"]).T)
    display(Markdown(reset_audit["interpretation"]))
else:
    display(Markdown("Проверка reset ещё не выполнена; нельзя называть эпизоды парными только по seed."))
'''),md('''### 5. Что именно означает «<1 GB» в статье
В версии v3 заявлено **0,9 GiB**. GiB = 1024³ байт; GB = 10⁹ байт. Мы показываем обе единицы: расход чуть ниже 1 GiB может оказаться выше 1 GB.

В приложении B.2 память измеряется как `peak nvidia-smi − idle nvidia-smi` с момента запуска процесса, включая загрузку, 10 warmup и 50 timed inference. Batch=1. Симулятор и подготовка/перенос изображений исключены из измерения latency; output chunk остаётся на GPU. Поэтому `torch.memory_allocated()` само по себе не проверяет заявление статьи: оно не учитывает CUDA context и часть библиотечных буферов.

Наш parent-процесс не импортирует torch, запускает `nvidia-smi` каждые 50 ms **до старта** отдельного worker. Worker загружает всю модель и реальное сохранённое D0-наблюдение; BERT вычисляется онлайн. Без CPU offload, квантования, удаления модулей или кэша текста. На этой headless-машине проверяем также idle-память ≤32 MiB: внутри контейнера `nvidia-smi` иногда не показывает PID даже активного процесса. Нагрузка других GPU-процессов исключается перед запуском; sampling может пропустить очень короткий пик, поэтому дополнительно сохраняются PyTorch peak allocated/reserved. Полный rollout с EGL имеет дополнительную память рендеринга и измеряется отдельно.
'''),code('''vram_rows=[]
for precision in ["bf16","fp32"]:
    path=RESULTS/f"vram_{precision}"/"summary.json"
    if path.exists():
        measurement=json.loads(path.read_text())
        vram_rows.append({key:measurement[key] for key in ["precision","gpu","parameters","delta_gib","delta_gb","torch_peak_allocated_gib","torch_peak_reserved_gib","median_ms","p90_ms","sample_count","at_most_1_gib","at_most_1_gb"]})
assert vram_rows, "Запустите benchmark_vram.py"
vram=pd.DataFrame(vram_rows)
print("GPU:", vram.iloc[0].gpu, "| Parameters:", f"{vram.iloc[0].parameters:,}")
memory_columns=["precision", "delta_gib", "delta_gb",
                "torch_peak_allocated_gib", "torch_peak_reserved_gib"]
memory_labels={"delta_gib":"nvidia-smi, GiB", "delta_gb":"nvidia-smi, GB",
               "torch_peak_allocated_gib":"torch allocated, GiB",
               "torch_peak_reserved_gib":"torch reserved, GiB"}
display(vram[memory_columns].rename(columns=memory_labels).round(4))
display(vram[["precision","median_ms","p90_ms","sample_count",
              "at_most_1_gib","at_most_1_gb"]].round(3))
for row in vram_rows:
    display(Markdown(f"**{row['precision']}**: peak − idle = **{row['delta_gib']:.3f} GiB / {row['delta_gb']:.3f} GB**. Порог ≤1 GiB: {'выполнен' if row['at_most_1_gib'] else 'не выполнен'}; ≤1 GB: {'выполнен' if row['at_most_1_gb'] else 'не выполнен'}. Median policy forward: **{row['median_ms']:.2f} ms**."))
'''),md('''### 6. Память во времени
Latency измеряет вычисление одного чанка, а не скорость полного цикла симулятора. В LIBERO контроллер обычно работает на 20 Hz, и 12-step open-loop означает новую командную последовательность примерно через 0,6 s симуляционного времени. Поэтому обратную величину latency нельзя выдавать за фактическую частоту closed-loop управления этого rollout.

Кривая охватывает загрузку и forward, а не только уже прогретую модель. Пунктиры — 1 GiB и 1 GB сверх исходного idle. FP32 — проверка чувствительности к precision; основной режим релиза BF16.
'''),code('''fig,ax=plt.subplots(figsize=(9,4))
styles={"bf16":dict(color="#23738b",linestyle="-"),
        "fp32":dict(color="#a26722",linestyle=":")}
for precision in vram.precision:
    folder=RESULTS/f"vram_{precision}"
    samples=[json.loads(line) for line in (folder/"memory_samples.jsonl").read_text().splitlines()]
    idle=json.loads((folder/"summary.json").read_text())["idle_mib"]
    ax.plot([s["time_s"] for s in samples],[(s["used_mib"]-idle)/1024 for s in samples],label=precision,**styles[precision])
ax.axhline(1,color="black",linestyle="--",label="1 GiB")
ax.axhline(1e9/1024**3,color="#626262",linestyle="-.",label="1 GB")
ax.set(xlabel="Время с запуска измерения, s",ylabel="Память сверх idle, GiB",title="RTX 4090: полная политика, загрузка + inference")
ax.set_ylim(bottom=0);ax.legend();ax.grid(alpha=.2);fig.tight_layout();display(fig);plt.close(fig)
rollout_path=RESULTS/"rollout_memory.json"
if rollout_path.exists():
    rollout_memory=json.loads(rollout_path.read_text())
    display(pd.DataFrame([rollout_memory])[["delta_gib","delta_gb",
        "sample_interval_ms","duration_s"]].round(4))
    display(Markdown(
        f"Полный официальный эвал с LIBERO/EGL и видео: **{rollout_memory['delta_gib']:.3f} GiB / "
        f"{rollout_memory['delta_gb']:.3f} GB**. Память симулятора входит в эту величину; "
        "это отдельная граница измерения от policy-only VRAM статьи."))
recheck_path=RESULTS/"vram_bf16_recheck/summary.json"
if recheck_path.exists():
    recheck=json.loads(recheck_path.read_text())
    display(Markdown(
        f"Повторная проверка BF16 после обоих эвалов: **{recheck['delta_gib']:.3f} GiB / "
        f"{recheck['delta_gb']:.3f} GB**, median **{recheck['median_ms']:.2f} ms**. "
        "Raw samples и описание входного наблюдения сохранены отдельно."))
'''),md('''## Проверки и дальнейшая работа

### 7. Повторить самостоятельно
Эти команды запускаются из `turbovla_eval` в терминале; результаты пишутся в новую папку. Для VRAM предварительно завершите все GPU kernels и убедитесь `nvidia-smi`, что карта свободна.
```bash
.venv/bin/python run_eval.py --protocol reference --episodes 50 --tasks D0,D1,D2,H1,H2 --output results/my_reference_run
.venv/bin/python run_eval.py --protocol official --episodes 50 --tasks D0,D1,D2,H1,H2 --output results/my_official_run
.venv/bin/python benchmark_vram.py --precision bf16 --output results/my_vram_run
```
Или в первом ноутбуке выберите другую задачу, измените количество эпизодов и запустите Run All.

Вывод о VRAM относится к нашей измеренной реализации, версии библиотек, precision и sampling. Один прогон не доказывает универсальный предел для любого процесса/драйвера. Значения статьи и нашего прогона сравнивайте в **одинаковых единицах и границах измерения**. Результаты H1/H2 не подтверждают OOD-обобщение TurboVLA, поскольку модель видела их демонстрации при обучении.
''')]
# Add context from the user's committed final pi0 experiment; preserve source evidence.
index=next(i for i,c in enumerate(cells2) if c.cell_type=='markdown' and '### 3. Видео' in c.source)
cells2[index:index]=[
md("""### Исходные варианты π₀ рядом с TurboVLA
Ниже — **сохранённые** результаты A/B/C после 3001 обновления из вашего репозитория, не новые прогоны π₀. A обучался на D0, B на D0+D1, C на D0+D1+D2; TurboVLA обучался на всех 40 задачах. Для TurboVLA берём `reference_n50`, чтобы проверить физические reset по одинаковым seed. Из-за разных обучающих данных, preprocessing и replan это сравнение опубликованных систем, а не контролируемая оценка влияния одного метода SFT.

Исходные минимальные записи сохранены в `provenance/pi0/<variant>/<task>/episodes.json`; ревизия reference — в `provenance/source_revisions.json`. H1/H2 для A/B/C были held-out; для TurboVLA — seen.
"""),
code("""comparison=[]
for variant,label in [("A","π₀ A: D0"),("B","π₀ B: D0+D1"),("C","π₀ C: D0+D1+D2")]:
    for key in TASKS:
        path=ROOT/"provenance/pi0"/variant/key/"episodes.json"
        rows=json.loads(path.read_text())
        assert len(rows)==50
        comparison.append(dict(model=label,task=key,success_percent=100*sum(r["success"] for r in rows)/50))
for _,row in table[table.protocol=="reference_n50"].iterrows():
    comparison.append(dict(model="TurboVLA: 40 задач",task=row.task,success_percent=row.success_percent))
comparison_table=pd.DataFrame(comparison).pivot(index="model",columns="task",values="success_percent")
display(comparison_table[["D0","D1","D2","H1","H2"]])
""")]
notebook(cells2,'02_results_and_vram.ipynb')
print('Created two seminar notebooks')

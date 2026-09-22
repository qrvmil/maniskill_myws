#!/usr/bin/env python3
"""Generate the seminar/re-evaluation notebook with nbformat, not raw JSON."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
cells=[]
def md(text):cells.append(nbf.v4.new_markdown_cell(text))
def code(text,**metadata):cells.append(nbf.v4.new_code_cell(text,metadata=metadata))
md('''# Multi-task π₀ SFT: acquisition, retention, and held-out transfer

This notebook accompanies the controlled 1/2/3-task experiment. All runs start independently from official π₀ base weights, use LoRA rank/alpha 32, batch 8, seed 0, and 3,001 optimizer updates. **H1 and H2 are excluded from every SFT corpus.** D1 and D2 scores count as transfer only for variants that did not train on them.

The default mode reads committed experiment evidence and shows its recorded rollouts. Switch `LIVE_EVALUATION` in the single configuration cell to run a compatible checkpoint again. A GPU and the pinned OpenPI/LIBERO environment are required for live inference. Reports/figures can be read on CPU.

Primary data sizes are 50/100/150 demonstrations. Task diversity is therefore confounded with total unique data and corpus-specific normalization. One training seed does not measure training-seed variance.''')
md('## 1. Configure once\nChange checkpoint, task, episode count, seeds, and video count here. Paths are explicit; new rollouts receive a new output directory.')
code('''from pathlib import Path
import os, sys
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src/maniskill_myws").exists())
EXPERIMENT_ROOT = Path("/workspace/multitask-sft")
VARIANT = "C"                 # A: D0; B: D0+D1; C: D0+D1+D2
UPDATES = 3001
CHECKPOINT = EXPERIMENT_ROOT / VARIANT / "checkpoints/pi0_libero_seen_lora32/EXP-001" / str(UPDATES)
TASK = "H1"                   # D0, D1, D2, H1, H2
N_EPISODES = 50
SEED_START = 10000
SEED_LIST = None               # e.g. [10002, 10005]; overrides N_EPISODES
VIDEOS_PER_OUTCOME = 2
LIVE_EVALUATION = False
EVIDENCE = REPO / "docs/multitask_sft/evidence/eval"
LIBERO_ROOT = Path("/workspace/LIBERO")
NORMALIZATION = None           # checkpoint's own statistics by default
''',tags=['parameters'])
md('## 2. Load reusable helpers\nInstall the pinned environment with `bash scripts/pld/setup_libero.sh`. Select its Python kernel. Core evaluation lives in repository modules; the notebook supplies configuration and presentation.')
code('''import json
import numpy as np
import pandas as pd
from IPython.display import display, Video, Image, Markdown
sys.path[:0] = [str(REPO / "src"), str(LIBERO_ROOT)]
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from maniskill_myws.pld.multitask_protocol import TASKS, TRAIN_SETS, prompt, summarize, paired_change
from maniskill_myws.pld.multitask_eval import EvalConfig, EvaluationSession, save_video
from maniskill_myws.pld.multitask_report import read_results, analysis, LABELS
seeds = tuple(SEED_LIST) if SEED_LIST is not None else tuple(range(SEED_START, SEED_START + N_EPISODES))
config = EvalConfig(str(CHECKPOINT), VARIANT, TASK, seeds, VIDEOS_PER_OUTCOME, NORMALIZATION)
''')
md('## 3. Available trained checkpoints\nDirectory names count completed optimizer updates. Update 0 is the saved LoRA initialization, including the pinned implementation’s small random functional LoRA perturbation.')
code('''checkpoint_rows = []
for variant in TRAIN_SETS:
    for update in (0, 500, 1000, 2000, 3001):
        path = EXPERIMENT_ROOT / variant / "checkpoints/pi0_libero_seen_lora32/EXP-001" / str(update)
        checkpoint_rows.append({"Variant": variant, "Train set": LABELS[variant], "Updates": update, "Available locally": path.exists(), "Checkpoint": str(path)})
display(pd.DataFrame(checkpoint_rows))
''')
md('## 4. Evaluation tasks and seen/held-out status')
code('''display(pd.DataFrame([{"Task": key, "Suite": value["suite"], "Instruction": prompt(key), "Seen in selected variant": key in TRAIN_SETS[VARIANT], "Horizon": value["horizon"]} for key, value in TASKS.items()]))
''')
md('## 5. Load one policy and inspect its contract\nNormalization is loaded from the selected checkpoint. Inference checks the exact prompt immediately before tokenization. Native cameras are rotated 180°, resized/padded to 224, and accompanied by 8D proprioception. π₀ predicts 50 actions; the adapter inverse-normalizes once, returns 7D OSC commands, clips to [−1,1], and executes 5 before replanning.')
code('''session = None
if LIVE_EVALUATION:
    session = EvaluationSession(config)
    session.set_task(TASK)
    binding = session.binding
else:
    binding = json.loads((EVIDENCE / VARIANT / str(UPDATES) / TASK / "binding.json").read_text())
display({"prompt": prompt(TASK), "normalization_sha256": binding["normalization_sha256"], "normalization_path": binding["normalization_path"], "action_contract": binding["action_contract"], "action_horizon": binding["action_horizon"], "replan_steps": binding["replan_steps"]})
''')
md('## 6. Run one rollout\nLive mode uses the same `run_one` method as the command-line evaluator. Cached mode inspects the corresponding recorded episode without adding an observation to the original experiment.')
code('''if LIVE_EVALUATION:
    one_row, one_transitions = session.run_one(seeds[0])
else:
    cached_rows = json.loads((EVIDENCE / VARIANT / str(UPDATES) / TASK / "episodes.json").read_text())
    one_row = next(r for r in cached_rows if r["seed"] == seeds[0])
display({k: one_row[k] for k in ("variant", "task", "seed", "success", "length", "reset_hash", "prompt")})
''')
md('## 7. Show a rollout\nVideos show agentview and wrist images side by side, including the terminal observation. Cached representative videos are the first two successes/failures in seed order; they may differ from the single episode above.')
code('''from datetime import datetime, timezone
run_output = EXPERIMENT_ROOT / "notebook_runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
if LIVE_EVALUATION:
    meta = save_video(run_output / "one_rollout", VARIANT, TASK, one_row, one_transitions)
    video_path = run_output / "one_rollout" / meta["filename"]
else:
    choices = sorted((REPO / "videos/multitask_sft").glob(f"train_{'_'.join(TRAIN_SETS[VARIANT])}_{TASK}_seed*.mp4")) if UPDATES == 3001 else []
    video_path = choices[0] if choices else None
if video_path is not None:
    display(Video(str(video_path), embed=True, width=640))
else:
    display(Markdown("No representative video is available for this selection. Committed videos are final (3001-update) checkpoints only; enable live evaluation to create an intermediate-checkpoint video."))
''')
md('## 8. Run N paired episodes\nThe helper verifies each task/seed reset hash against the shared registry. Changing seeds is valid for a new evaluation, but comparisons require the same seeds and reset states on both sides.')
code('''if LIVE_EVALUATION:
    rows = session.evaluate(TASK, run_output / TASK, seeds=seeds, videos=VIDEOS_PER_OUTCOME)
else:
    rows = [r for r in cached_rows if r["seed"] in set(seeds)]
    if {r["seed"] for r in rows} != set(seeds):
        raise ValueError("Requested seeds absent from cached evidence; enable LIVE_EVALUATION")
display(pd.DataFrame([{k:r[k] for k in ("seed","success","length","reset_hash")} for r in rows]).head(10))
''')
md('## 9. Success rate and Wilson 95% interval')
code('''summary = summarize(rows)
display(pd.DataFrame([{"Successes/N": f"{summary['successes']}/{summary['n']}", "SR (%)": 100*summary["success_rate"], "Wilson low (%)": 100*summary["wilson_95ci"][0], "Wilson high (%)": 100*summary["wilson_95ci"][1], "Mean episode length": summary["mean_episode_length"]}]))
''')
md('## 10. Reach, contact, lift, and success\nAvailable for D0/D1/H1. These are descriptive, not necessarily nested stages. A sampled false contact flag does not establish absence of a physical grasp.')
code('''if "task_progress" in rows[0]:
    stages = {"Reach <10 cm": np.mean([r["ever_reached_10cm"] for r in rows]), "Grasp contact": np.mean([r["task_progress"]["ever_grasped"] for r in rows]), "Lift >3 cm": np.mean([r["task_progress"]["ever_lifted_3cm"] for r in rows]), "Success": summary["success_rate"]}
    display(pd.Series(stages, name="Episode fraction").to_frame())
else:
    display(Markdown("The original bowl/plate stage definitions apply to D0/D1/H1; D2/H2 use task success only."))
''')
md('## 11. Representative video selection\n`evaluate` saves and fully decodes the first requested number of success/failure clips in seed order. Missing categories are recorded explicitly.')
code('''from maniskill_myws.pld.multitask_protocol import select_videos
selected = select_videos(rows, VIDEOS_PER_OUTCOME)
display(pd.DataFrame([{k:r[k] for k in ("seed","success","length")} for r in selected]))
''')
md('## 12. Compare checkpoints on paired seeds\nA positive gain means rescue episodes exceed harm episodes. Bootstrap intervals resample pairs within each task; boundary-degenerate bootstrap intervals are accompanied by an exact bound in the detailed statistics.')
code('''episodes, summaries, sensitivity = read_results(EVIDENCE)
comparisons = analysis(episodes)
before = episodes["A"][3001][TASK]
after = episodes["C"][3001][TASK]
display(paired_change(before, after))
''')
md('## 13. Main task matrix and common held-out mean\nH1/H2 use equal task weights. D1/D2 acquisition by a trained variant is not held-out generalization.')
code('''final_table = pd.DataFrame([r for r in summaries if r["updates"] == 3001])
matrix = final_table.pivot(index="variant", columns="task", values="success_rate").reindex(index=list(TRAIN_SETS), columns=list(TASKS))*100
matrix["Held-out mean"] = [comparisons["heldout"][v]["heldout_mean"]*100 for v in matrix.index]
display(matrix.round(1))
''')
md('## 14. Plot the task matrix\nThe plotting code remains editable. Gold outlines identify tasks seen during SFT.')
code('''import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
fig, ax = plt.subplots(figsize=(9,4))
values = matrix[list(TASKS)].to_numpy()
im = ax.imshow(values, vmin=0, vmax=100, cmap="Blues", aspect="auto")
for i, variant in enumerate(TRAIN_SETS):
    for j, task in enumerate(TASKS):
        ax.text(j, i, f"{values[i,j]:.0f}%", ha="center", va="center", color="white" if values[i,j]>55 else "black")
        if task in TRAIN_SETS[variant]: ax.add_patch(Rectangle((j-.46,i-.43),.92,.86,fill=False,ec="#d5982d",lw=2.5))
ax.set_xticks(range(5), list(TASKS)); ax.set_yticks(range(3), [LABELS[v] for v in TRAIN_SETS])
ax.set_title("Final success rate · 50 paired resets per cell")
fig.colorbar(im, ax=ax, label="SR (%)"); fig.tight_layout(); plt.show()
''')
md('## 15. Image sensitivity and training trajectory\nThese diagnostics explain behavior without choosing checkpoints or tuning on held-out outcomes.')
code('''display(Image(filename=str(REPO / "docs/multitask_sft/heldout_generalization.png")))
display(Image(filename=str(REPO / "docs/multitask_sft/sft_trajectory.png")))
display(Image(filename=str(REPO / "docs/multitask_sft/image_sensitivity.png")))
''')
md('## 16. Change task or checkpoint and rerun\nEdit only the configuration cell: for example, set `VARIANT="B"`, point `CHECKPOINT` at its 1,000-update directory, set `TASK="D1"`, `N_EPISODES=5`, `VIDEOS_PER_OUTCOME=1`, and enable `LIVE_EVALUATION`. Then restart the kernel and run all cells. Arbitrary compatible π₀ LoRA32 checkpoints can be supplied with their own normalization file. The command-line equivalent is:\n\n```bash\nPYTHONPATH=/workspace/LIBERO:src third_party/openpi/.venv/bin/python scripts/eval_multitask_sft.py \\\n  --checkpoint /path/to/checkpoint --variant B --task H1 \\\n  --episodes 50 --seed-start 10000 --videos 2\n```\n\nUse the human report for interpretation and the methods appendix for provenance. Never replace the preregistered final table with an exploratory rerun.')
code('''if session is not None:
    session.close()
''')
nb.cells=cells
nb.metadata={'kernelspec':{'display_name':'Python (pi0 multitask)','language':'python','name':'pi0-multitask'},'language_info':{'name':'python','version':'3.11'}}
nbf.validate(nb)
folder=ROOT/'notebooks';folder.mkdir(exist_ok=True)
nbf.write(nb,folder/'01_multitask_sft_eval.ipynb')

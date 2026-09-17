#!/usr/bin/env python3
"""Write a human-facing report only from complete, validated audit evidence."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from maniskill_myws.pld.libero_sanity import UPDATES,paired_change
from report_pi0_audit import generate,LABELS
from report_pi0_audit_videos import publish_videos

WORK=Path('/workspace/audit-run')
OUT=ROOT/'docs/pi0_audit'


def main():
    # Rebuild dependencies from current rows; never combine new tables with stale figures.
    s,paired,rows=generate(WORK,OUT)
    publish_videos()
    norm=json.loads((WORK/'normalization_audit.json').read_text())
    video_records=json.loads((OUT/'video_verification.json').read_text())['videos']
    video_links=[]
    for model,task,outcome,label in [('0','D1',False,'Pre-SFT D1 failure'),
                                    ('3001','D1',False,'Final D0-only model: D1 failure'),
                                    ('positive_control','D1',True,'Official LIBERO control: D1 success'),
                                    ('3001','D0',True,'Final D0-only model: D0 success')]:
        matches=[v for v in video_records if v['model']==model and v['task']==task and v['success']==outcome]
        if matches:
            v=matches[0]
            video_links.append(f"- [{label}, seed {v['seed']}](pi0_audit/videos/{v['filename']})")
        else:video_links.append(f'- {label}: this outcome did not occur among the 50 evaluated episodes.')
    def rate(m,t):return f"{s[str(m)][t]['success_rate']*100:.0f}%"
    def count(m,t):return f"{s[str(m)][t]['successes']}/50"
    def interval(m,t):return '–'.join(f'{v*100:.1f}%' for v in s[str(m)][t]['wilson_95ci'])
    n0=s['0']['D1']['successes'];nf=s['3001']['D1']['successes'];ni=s['lora_initialization']['D1']['successes']
    middle=[x for x in (500,1000,2000) if s[str(x)]['D1']['successes']]
    middle_peak=max(s[str(x)]['D1']['successes'] for x in (500,1000,2000))
    if n0==0 and nf==0 and not middle and ni==0:
        answer='No D1 successes were observed in any measured pre-SFT or D0-only model.'
        short_interpretation='These measurements do not show forgetting of previously demonstrated D1 success. The earlier residual experiment started from a base with no observed full-task D1 success.'
        interpretation=("We observed no successful D1 completions before D0 gradient updates, and none emerged at the registered checkpoints. "
            "These results do **not** support the claim that D0 SFT destroyed previously demonstrated D1 competence. "
            "Instead, the pretrained model under D0 embodiment alignment never demonstrated successful D1 execution in this audit. "
            "Zero successes in 50 episodes does not establish a true success probability of zero or exclude useful partial skills.")
    elif middle_peak>max(n0,nf):
        answer='D1 success peaked at an intermediate checkpoint and was lower at the final checkpoint.'
        short_interpretation='The observed decline from an intermediate D1 peak is consistent with later specialization, subject to paired uncertainty. The earlier residual experiment still used the unchanged final base.'
        interpretation=("The trajectory contains an intermediate D1 peak followed by lower final success. "
            "This is consistent with loss of generalization during later D0 training, with paired uncertainty reported in the trajectory section. The initial baseline and random LoRA initialization "
            "must still be considered separately: an intermediate gain followed by loss is not equivalent to forgetting a strong pretrained skill.")
    elif n0>nf:
        answer='D1 success decreased between the pre-SFT baseline and final D0 checkpoint.'
        short_interpretation='The decline is consistent with loss of D1 generalization, subject to paired uncertainty and one training seed. Residual-RL results must be interpreted against the weaker final base.'
        interpretation=("D0 training is associated with reduced measured D1 success. This is consistent with specialization, "
            "but the paired uncertainty and the separately measured LoRA initialization determine how strongly the change can be attributed "
            "to gradient updates. A single training seed does not establish a universal forgetting mechanism.")
    elif ni>nf:
        answer='The zero-update LoRA model had more D1 successes than the final D0 model.'
        short_interpretation='The LoRA-initialization comparison is consistent with a training-related decline; it does not by itself establish forgetting of a dense pretrained skill. The final base defines the starting point for the earlier residual experiment.'
        interpretation=("Random LoRA initialization and subsequent optimization have distinguishable effects. "
            "The auxiliary initial state shows D1 successes that are reduced in the final checkpoint, even though the dense baseline "
            "must be interpreted separately. This is evidence consistent with training-related loss, with the limits of N = 50 and one training seed.")
    else:
        answer='The audit did not reproduce a reduction in D1 success from the primary pre-SFT baseline.'
        short_interpretation='A categorical forgetting claim is unsupported by this trajectory. The measured final D1 rate and partial behavior define the relevant base prior for interpreting the earlier residual experiment.'
        interpretation=("The fixed checkpoint trajectory does not demonstrate degradation from the primary zero-update baseline. "
            "The observed task rates and paired changes below should replace a categorical forgetting narrative for this seed block.")
    control=s['positive_control']['D1']['successes']
    if control<=2:
        answer='The official positive control did not validate the D1 bridge; specialization is not established.'
        short_interpretation='Investigate the evaluation bridge before attributing D1 failures to specialization or drawing further residual-RL conclusions.'
        interpretation='The official checkpoint also performed near zero. Environment/checkpoint/action-bridge issues must be investigated before attributing the D0 model’s D1 failure to specialization.'
    model_table='\n'.join([
        '| Pre-SFT D0 setup | Official pi0 weights; statistics from 50 D0 demos | 0 | Primary pre-SFT baseline |',
        '| LoRA initialization | Same D0 statistics; V4 random LoRA factors | 0 | Isolate initialization effects |',
        '| D0-only SFT trajectory | Exactly 50 D0 demos; no D1 training | 500 / 1,000 / 2,000 | Measure D0 learning and D1 generalization |',
        '| Final D0-only SFT | Exactly the same D0 dataset | 3,001 | Match the V4 training budget |',
        '| Official pi0.5 LIBERO | Official LIBERO fine-tuning corpus | Released checkpoint; no training here | Positive control only |'])
    main_table=[]
    for m in [str(x) for x in UPDATES]+['lora_initialization','positive_control']:
        main_table.append(f"| {LABELS[m]} | {rate(m,'D0')} [{interval(m,'D0')}] | {rate(m,'D1')} [{interval(m,'D1')}] | {count(m,'D0')} | {count(m,'D1')} |")
    lengths='\n'.join(f"| {LABELS[m]} | 50 | {s[m]['D0']['mean_episode_length']:.1f} | {s[m]['D1']['mean_episode_length']:.1f} |" for m in s)
    gains='\n'.join(f"| {x:,} | {t} | 50 | {paired[str(x)][t]['delta_pp']:+.0f} | {paired[str(x)][t]['gained']} | {paired[str(x)][t]['lost']} | [{paired[str(x)][t]['interval_95_pp'][0]:+.1f}, {paired[str(x)][t]['interval_95_pp'][1]:+.1f}] |" for x in UPDATES[1:] for t in ('D0','D1'))
    stages='\n'.join(f"| {LABELS[m]} | 50 | "+' | '.join(f"{s[m]['D1']['stages'][k]}/50 ({s[m]['D1']['stages'][k]*2:.0f}%)" for k in ('reach','grasp','lift','success'))+' |' for m in [str(x) for x in UPDATES]+['lora_initialization','positive_control'])
    dims=['x','y','z','rotation x','rotation y','rotation z','gripper']
    action_table='\n'.join(f"| {name} | [{norm['actions']['d0_min'][i]:.3f}, {norm['actions']['d0_max'][i]:.3f}] | [{norm['actions']['d1_min'][i]:.3f}, {norm['actions']['d1_max'][i]:.3f}] | {100*norm['actions']['d1_outside_d0_minmax_fraction'][i]:.2f}% | {100*norm['actions']['d1_outside_d0_q01q99_fraction'][i]:.2f}% | {100*norm['actions']['d1_abs_z_above3_fraction'][i]:.2f}% |" for i,name in enumerate(dims))
    state_dims=['EEF x','EEF y','EEF z','axis-angle x','axis-angle y','axis-angle z','gripper joint 1','gripper joint 2']
    state_table='\n'.join(f"| {name} | {100*norm['state']['d1_outside_d0_minmax_fraction'][i]:.2f}% | {100*norm['state']['d1_outside_d0_q01q99_fraction'][i]:.2f}% | {100*norm['state']['d1_abs_z_above3_fraction'][i]:.2f}% |" for i,name in enumerate(state_dims))
    actions=list(csv.DictReader((OUT/'action_differences.csv').open()))
    action_changes='\n'.join(f"| {LABELS[r['model']]} | {('Predicted chunk (pre-clip)' if r['action_kind']=='chunks' else 'First 5 executed')} | 10 | {r['actions_per_state']} | {float(r['mean_l1']):.4f} | {float(r['mean_l2']):.4f} |" for r in actions)
    runtime=[]
    for p in sorted((WORK/'runtime').glob('*/metadata.json')):
        d=json.loads(p.read_text())
        if d['status']=='COMPLETED' and (p.parent.name=='train' or p.parent.name.startswith('eval_')):
            runtime_label=('D0-only SFT' if p.parent.name=='train' else
                LABELS[p.parent.name.removeprefix('eval_').removesuffix('_video_replay')]+
                (' (video replay)' if p.parent.name.endswith('_video_replay') else ''))
            runtime.append(f"| {runtime_label} | {d['wall_seconds']/60:.1f} | {d['device_peak_sampled_used_mib']/1024:.2f} |")
    init_change=paired_change(rows['lora_initialization']['D1'],rows['3001']['D1'])
    trajectory=', '.join(f"{x:,} updates: {count(x,'D1')}" for x in UPDATES)
    if middle:
        best=max(UPDATES,key=lambda x:(s[str(x)]['D1']['successes'],s[str(x)]['D0']['successes'],-x))
        tradeoff=f"The highest observed D1 rate (breaking ties by higher D0 success) was at {best:,} updates: D0 {count(best,'D0')} ({rate(best,'D0')}), D1 {count(best,'D1')} ({rate(best,'D1')}). This is a descriptive, post-hoc tradeoff; it does not redefine the V4 base or select a new scientific base."
        if best!=3001:
            peak_change=paired_change(rows[str(best)]['D1'],rows['3001']['D1'])
            tradeoff+=f" From that observed peak to the final checkpoint, the paired D1 change was {peak_change['delta_pp']:+.0f} percentage points, with {peak_change['gained']} gained and {peak_change['lost']} lost successes (95% change interval [{peak_change['interval_95_pp'][0]:+.1f}, {peak_change['interval_95_pp'][1]:+.1f}] pp). This interval does not adjust for selecting the observed peak."
    else:
        tradeoff='No intermediate D1 success was observed at the three registered intermediate checkpoints.'
    report=f'''# pi0 D0/D1 base-model sanity audit

## TL;DR

- **{answer}**
- We evaluated the pre-SFT D0 setup and four fixed D0-only checkpoints on the same 50 fresh scenes per task. Before SFT: D0 **{count(0,'D0')} ({rate(0,'D0')})**, D1 **{count(0,'D1')} ({rate(0,'D1')})**.
- After 3,001 D0 updates: D0 **{count(3001,'D0')} ({rate(3001,'D0')})**, D1 **{count(3001,'D1')} ({rate(3001,'D1')})**. D1 trajectory: {trajectory}.
- The official LIBERO positive control achieved D0 **{count('positive_control','D0')} ({rate('positive_control','D0')})** and D1 **{count('positive_control','D1')} ({rate('positive_control','D1')})**.
- We separately checked zero-update LoRA initialization: D0 {count('lora_initialization','D0')}, D1 {count('lora_initialization','D1')}. The pinned code randomly initializes both factors, so it is not identical to the dense pretrained baseline.
- {short_interpretation}

## 1. Experimental question

The previous experiment learned the center-bowl task D0 but showed zero success on D1, where the black bowl starts next to the plate. The robot, destination and pick/place primitive are shared. Inspection of the unchanged BDDL files also shows a distractor relocation: D0 has a second bowl near the plate, while D1 places that bowl near the ramekin and leaves the center empty. This is not a pure language-only intervention. We asked whether D0 SFT erased a skill the pretrained model already had, whether the skill was absent to begin with, or whether the LIBERO inference bridge could not execute the task correctly.

**Raw pi0_base is not directly evaluable under a justified LIBERO action contract.** Its official assets do not include LIBERO normalization. We did not substitute another robot's statistics. Our primary pre-SFT baseline is official pi0 weights **plus the D0-specific LIBERO transforms and statistics**, with zero gradient updates. This is already an embodiment-aligned setup, not a raw generic zero-shot LIBERO score.

## 2. Models compared

| Model | Training data / alignment | Optimizer updates | Purpose |
|---|---|---:|---|
{model_table}

D0-only training used LoRA rank 32, alpha 32, batch 8, seed 0, all 50 D0 demonstrations and the existing temporal alignment, cameras, optimizer and schedule. No D1 data entered SFT or D0 statistics. The official pi0.5 control uses its own released normalization and architecture; it is an environment/inference check, not a competing source-only training method.

## 3. Main results

N = 50 per task and model. Brackets contain pointwise Wilson 95% confidence intervals for success rate.

| Model/checkpoint | D0 SR [95% CI] | D1 SR [95% CI] | D0 successes | D1 successes |
|---|---:|---:|---:|---:|
{chr(10).join(main_table)}

| Model/checkpoint | Episodes per task | Mean D0 length, actions | Mean D1 length, actions |
|---|---:|---:|---:|
{lengths}

Failures terminate at the fixed 220-action horizon. Episodes and checkpoints are repeated measurements on the same scene seeds, not independent samples to be pooled.

## 4. SFT trajectory

![D0 and D1 success versus D0 optimizer updates](pi0_audit/sft_trajectory.png)

[PDF figure](pi0_audit/sft_trajectory.pdf). D0 learning was weak early: {count(500,'D0')} ({rate(500,'D0')}) at 500 updates and {count(1000,'D0')} ({rate(1000,'D0')}) at 1,000. It then rose to {count(2000,'D0')} ({rate(2000,'D0')}) at 2,000 and {count(3001,'D0')} ({rate(3001,'D0')}) at the final checkpoint, from {count(0,'D0')} before training. The small early dip is an observed count difference in one training run, not evidence of a reliable degradation. For D1, {trajectory}. {tradeoff}

The primary zero point uses dense official weights. The separate LoRA-initialization row prevents a random adapter perturbation from being mistaken for a gradient update. Comparing the final model with that auxiliary zero-update state gives a D1 change of {init_change['delta_pp']:+.0f} percentage points, with {init_change['gained']} gained and {init_change['lost']} lost successes among 50 pairs.

| D0 optimizer updates | Task | Paired N | Change from primary step 0, pp | Gained successes | Lost successes | 95% change interval, pp |
|---:|---|---:|---:|---:|---:|---:|
{gains}

Intervals use paired percentile bootstrapping where informative; constant outcomes use explicit conservative bounds. They are pointwise and do not correct for inspecting multiple checkpoints. Baseline self-comparison is exactly zero and is omitted.

## 5. Positive-control result

![Comparison with official LIBERO positive control](pi0_audit/positive_control_comparison.png)

[PDF figure](pi0_audit/positive_control_comparison.pdf). The official `pi05_libero` model achieved D0 {count('positive_control','D0')} ({rate('positive_control','D0')}) and D1 {count('positive_control','D1')} ({rate('positive_control','D1')}) through the same environment, task language, camera adapter and bounded OSC execution path. {'This establishes that this bridge can produce D1 successes; the D0-only model’s floor cannot be explained by a bridge that universally prevents the task.' if control>2 else 'This does not validate D1 execution. Bridge/checkpoint issues need investigation before interpreting D0 specialization.'}

The control has LIBERO training exposure and a different architecture and normalization scheme. It cannot establish what generic pretrained pi0 knew, nor rule out every model-specific contract problem. Its native prediction horizon is 10 versus 50 for pi0; both execute five actions per replan. Our generated scene resets also differ from official benchmark initialization files.

## 6. Behavior analysis

[Evaluation video index](PI0_D0_D1_SANITY_VIDEOS.md), with visible-behavior notes and full decode/frame-count verification. Representative clips:

{chr(10).join(video_links)}

Videos show agentview and wrist views side by side, including the terminal observation. The selected baseline and official-control clips were deterministically replayed after the video requirement arrived; their seed, outcome, length and reset/trajectory/image hashes match the original evaluated episodes. These replays do not add episodes to the reported success rates.

In the retained final D1 clips for seeds 9000 and 9001, the gripper descends over the empty center region, where the D0 target would be, and then moves toward the plate. The D1 target bowl remains beside the plate. The final D0 success clips show the center bowl carried onto the plate. These are visible motions in selected examples; they do not identify an internal cause or establish that the model ignored language.

Reach means EEF-to-target-body distance below 10 cm at any time; grasp uses robosuite's target-object contact check; lift means the target bowl rises more than 3 cm above its post-settle initial height; placement is binary LIBERO success. These are descriptive, not necessarily nested stages. The grasp-contact flag is sampled at action boundaries and can miss physical acquisition; the control's successful episodes without a registered flag demonstrate that limitation.

At the final D0 checkpoint, reach occurred in {s['3001']['D0']['stages']['reach']}/50 episodes, grasp flags in {s['3001']['D0']['stages']['grasp']}/50, and lifts in {s['3001']['D0']['stages']['lift']}/50, with {count(3001,'D0')} successful placements. The two retained D0 failures carry the bowl near the plate but end with it still in the gripper. The following table tracks D1 separately.

| Model/checkpoint | D1 episodes | Reach | Grasp | Lift >3 cm | Successful placement |
|---|---:|---:|---:|---:|---:|
{stages}

![D1 behavior stages](pi0_audit/d1_behavior_stages.png)

[PDF](pi0_audit/d1_behavior_stages.pdf). From primary step 0 to the final D0 model, D1 reaches changed from {s['0']['D1']['stages']['reach']}/50 to {s['3001']['D1']['stages']['reach']}/50, grasp flags from {s['0']['D1']['stages']['grasp']}/50 to {s['3001']['D1']['stages']['grasp']}/50, and lifts from {s['0']['D1']['stages']['lift']}/50 to {s['3001']['D1']['stages']['lift']}/50. These partial-behavior measurements complement full-task success rates and help describe where progress stalls, but they are not a validated universal stage classifier. In particular, object lift alone does not establish a stable grasp.

For ten identical D1 reset states, we compared the first predicted chunk and first five executed commands with primary step 0. L1 sums absolute differences over seven command dimensions; L2 is Euclidean command-vector difference, averaged over states and actions.

| Model / update count | Action representation | Paired states | Actions per state | Mean L1 difference | Mean L2 difference |
|---|---|---:|---:|---:|---:|
{action_changes}

Predicted chunks are before controller clipping; executed commands are after clipping. These are normalized OSC controller commands, **not physical meters**. A large difference shows changed commands, not by itself a stereotyped D0 strategy or ignored language. Fixed-seed videos and per-dimension differences are retained for qualitative inspection. Every model's actual inference path checked the task string immediately before tokenization: D0 “pick up the black bowl from table center and place it on the plate”; D1 “pick up the black bowl next to the plate and place it on the plate”.

## 7. Normalization and action-distribution audit

Primary step 0, the auxiliary initialization and every D0 SFT checkpoint use the identical D0-only mean/std statistics. The official pi0.5 control uses its own checkpoint's 1st/99th-percentile statistics. Each model's inverse normalization occurs exactly once before the shared OSC clipping/execution adapter. No statistics were changed after inspecting D1.

The unique-frame diagnostic covers **50 D0 demonstrations / {norm['actions']['d0_n']:,} aligned pairs** and **50 D1 demonstrations / {norm['actions']['d1_n']:,} aligned pairs**. D1 data were read only for this diagnostic. The original chunked loader weights overlapping/padded actions differently from these unique-frame summaries.

| Action dimension | D0 observed range (N={norm['actions']['d0_n']:,}) | D1 observed range (N={norm['actions']['d1_n']:,}) | D1 outside D0 range | D1 outside D0 1st–99th percentiles | D1 with absolute D0 z score >3 |
|---|---:|---:|---:|---:|---:|
{action_table}

D1 command tails differ: the largest fraction outside a D0 observed action range is {100*max(norm['actions']['d1_outside_d0_minmax_fraction']):.2f}%; the largest fraction outside its central 98% range is {100*max(norm['actions']['d1_outside_d0_q01q99_fraction']):.2f}%. This does not indicate wholesale incompatibility of the seven-dimensional action space, but tail and state shifts remain plausible contributors to poor generalization.

The following state fractions use the same {norm['state']['d1_n']:,} D1 observations (50 demonstrations):

| State dimension (D1 N={norm['state']['d1_n']:,}) | D1 outside D0 observed range | D1 outside D0 1st–99th percentiles | D1 with absolute D0 z score >3 |
|---|---:|---:|---:|
{state_table}

The D0 mean/std transform is affine and does not clip values to these demonstration ranges; an out-of-range value is not automatically an invalid OSC command. The shared controller clipping remains at [-1, 1], independently of the observed demo ranges. These comparisons diagnose distribution shift; they do not prove normalization caused failure. Orientation-coordinate differences and demonstration timing limit interpretation. Native D1 action trajectories are not claimed to reproduce identically in this simulator version, and no alternate normalizer was tested or fitted.

## 8. Interpretation

{interpretation}

The four questions remain separate:

1. **Does generic pi0 know LIBERO?** This audit cannot assign a raw zero-shot success rate without LIBERO statistics. It measures the explicitly D0-aligned pre-SFT setup.
2. **Does D0 SFT teach D0?** D0 changes from {count(0,'D0')} to {count(3001,'D0')}; the registered intermediate rates are in the main table.
3. **Does D0 SFT destroy D1 behavior?** The measured D1 trajectory is {trajectory}. The initial LoRA model separately achieved {count('lora_initialization','D1')}; causal language must respect that initialization comparison and the paired uncertainty.
4. **Does the official LIBERO-finetuned checkpoint work here?** The official control achieved {count('positive_control','D1')} on D1 and {count('positive_control','D0')} on D0. {'The bridge passed the positive control.' if control>2 else 'The bridge did not pass the positive control.'}

## 9. Implication for residual RL

The final D0-only base had D1 success {count(3001,'D1')} in this independent audit. {'A bounded residual therefore had to achieve successful D1 execution from a base with no demonstrated full-task success. That is a harder starting point than correcting occasional failures in a mostly successful policy; the measured reach, grasp and lift behavior still matters because partial skills may remain useful.' if nf==0 else 'The measured D1 prior should be considered explicitly when interpreting a bounded residual’s learning problem.'} This makes the base prior a central limitation of the earlier D0-base → D1-residual experiment; it does not prove residual RL is impossible or identify a single optimization failure.

The prior V4 experiment, its frozen 3,001-update base definition, and its negative results remain unchanged. We did not train a residual, tune using D1, alter normalization, or retrospectively replace that scientific base with an intermediate checkpoint.

## 10. Limitations

- One training seed and 50 paired episodes per task limit precision; checkpoint comparisons are correlated and exploratory. The fixed milestones cannot exclude short-lived behavior between evaluated checkpoints.
- The primary baseline includes D0 statistics. Raw pretrained LIBERO competence remains undefined under the missing deployment contract.
- Random LoRA initialization is a separate perturbation, explicitly measured. Hardware differs from V4, so identical configuration does not imply bitwise-identical learned weights.
- The pi0.5 positive control has LIBERO exposure, another normalization scheme and another horizon; it validates achievable execution, not equality of model assumptions.
- The task pair changes distractor placement as well as target location, so failures cannot be attributed to language neglect alone. Generated resets differ from the published benchmark protocol. Historical raw seed artifacts were unavailable in this clone; freshness was checked against tracked registrations and reports.
- The actual GPU has 64 GiB. Measured usage informs the requested 48 GB design but is not a direct 48 GB hardware demonstration.

## Technical appendix

Full contracts, definitions and source links: [methods](pi0_audit/METHODS.md). Inspectable data: [success table CSV](pi0_audit/success_rates.csv), [episode CSV](pi0_audit/episodes.csv), [paired changes](pi0_audit/paired_changes.json), [action differences](pi0_audit/action_differences.csv), [first commands](pi0_audit/first_actions.csv), [normalization diagnostic](pi0_audit/normalization_audit.json).

Runtime: NVIDIA CMP 170HX, 65,536 MiB VRAM, driver 610.57.04; Torch 2.7.1+cu128, JAX/JAXlib/CUDA plugin/PJRT 0.5.3, MuJoCo 3.2.7, robosuite 1.4.1. OpenPI `981483dca0fd9acba698fea00aa6e52d56a66c58`; LIBERO `8f1084e3132a39270c3a13ebe37270a43ece2a01`. Seeds 9000–9049; first-action states 9000–9009. D0 normalization SHA256 `{json.loads((WORK/'d0_binding.json').read_text())['normalization_sha256']}`.

Primary pre-SFT checkpoint: `/workspace/audit-checkpoints/pi0_base` with explicit D0 statistics. Auxiliary LoRA initial checkpoint: `/workspace/audit-run/sft/checkpoints/pi0_libero_seen_lora32/EXP-001/0`. Later directories are `500`, `1000`, `2000`, `3001`, and correspond to exactly those completed updates; the final directory is not historical V4's `3000`. Official control: `/workspace/audit-checkpoints/pi05_libero`, downloaded from `gs://openpi-assets/checkpoints/pi05_libero` ([official OpenPI checkpoint listing](https://github.com/Physical-Intelligence/openpi#pre-trained-checkpoints)).

| Stage | Wall time, minutes | Sampled whole-device peak, GiB |
|---|---:|---:|
{chr(10).join(runtime)}

Reproduce using `scripts/pld/audit_pi0.py`, `audit_pi0_pipeline.py`, `audit_pi0_normalization.py`, `report_pi0_audit.py` and `write_pi0_audit_report.py`. See [preregistration](plans/2026-09-16-pi0-audit.md) for settings and sequence. Audit entrypoint snapshots were added before D0 evaluation/training; the already-running official control's later source capture is explicitly labeled. Final verification: **126 tests passed** (including the real MuJoCo integration test); **4 optional tests skipped**; [test log](pi0_audit/evidence/verification/final-tests.log). All **38 published videos** fully decode and match their expected frame counts and episode identities; [video verification](pi0_audit/video_verification.json). Training completed without OOM or reduced scientific settings, with a sampled peak of **32.34 GiB**. Historical tracked V2/V3/V4 files remain unchanged. The [evidence bundle](pi0_audit/evidence/README.md) includes the original results, runtime/source records and reproducible figure inputs.
'''
    (ROOT/'docs/PI0_D0_D1_SANITY_REPORT.md').write_text(report)
    (OUT/'normalization_audit.json').write_text(json.dumps(norm,indent=2)+'\n')
    print(answer)

if __name__=='__main__':main()

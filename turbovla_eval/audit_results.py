"""Audit real episode coverage, paired reset hashes, action/video consistency."""
import json
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from eval_utils import ROOT,TASKS,summarize,write_json


def audit_reference():
    output=ROOT/'results/reference_n50'
    reference=ROOT.parent/'docs/multitask_sft/evidence/eval/C/3001'
    if not reference.is_dir():
        reference=ROOT.parent/'reference_maniskill/docs/multitask_sft/evidence/eval/C/3001'
    audit={'tasks':{}}
    for key in TASKS:
        rows=json.loads((output/key/'episodes.json').read_text())
        assert len(rows)==50
        assert {r['seed'] for r in rows}==set(range(10000,10050))
        assert summarize(rows)==json.loads((output/key/'summary.json').read_text())
        old={r['seed']:r for r in json.loads((reference/key/'episodes.json').read_text())}
        matches=sum(r['reset_hash']==old[r['seed']]['reset_hash'] for r in rows if r['seed'] in old)
        covered=sum(r['seed'] in old for r in rows)
        videos=0
        for row in rows:
            assert len(row['actions'])==row['steps']<=row['horizon']
            assert np.isfinite(row['actions']).all()
            assert (np.abs(row['actions'])<=1).all()
            path=ROOT/row['video']
            assert path.is_file()
            sidecar=json.loads(path.with_suffix('.json').read_text())
            assert all(sidecar[k]==row[k] for k in ['seed','success','reset_hash','steps','video_frames'])
            reader=imageio.get_reader(path)
            try: assert reader.count_frames()==row['steps']+1
            finally: reader.close()
            videos+=1
        audit['tasks'][key]=dict(episodes=len(rows),reference_covered=covered,reset_hash_matches=matches,
            reset_hash_mismatches=covered-matches,videos_verified=videos)
    all_match=all(x['reference_covered']==50 and x['reset_hash_matches']==50 for x in audit['tasks'].values())
    audit['interpretation']=('Все 250 физических reset-состояний побайтово совпадают с исходным π₀-экспериментом по SHA256. Это позволяет сопоставлять исходы по seed; модели имеют разное обучающее покрытие и разные replan/preprocessing.' if all_match else 'Есть несовпадения или недостающие исходные состояния: нельзя заявлять полностью парный эвал. См. числа по задачам.')
    write_json(ROOT/'results/reference_reset_audit.json',audit)
    print(json.dumps(audit,indent=2,ensure_ascii=False))

if __name__=='__main__': audit_reference()


def audit_official():
    output=ROOT/'results/official_n50'
    audit={}
    for key in TASKS:
        rows=json.loads((output/key/'episodes.json').read_text())
        assert len(rows)==50
        assert {r['init_state_index'] for r in rows}==set(range(50))
        assert all(r['rng_seed']==7 for r in rows)
        assert summarize(rows)==json.loads((output/key/'summary.json').read_text())
        for row in rows:
            assert len(row['actions'])==row['steps']<=row['horizon']
            assert np.asarray(row['actions']).shape==(row['steps'],7) and np.isfinite(row['actions']).all()
            path=ROOT/row['video']
            sidecar=json.loads(path.with_suffix('.json').read_text())
            assert sidecar['init_state_index']==row['init_state_index'] and sidecar['success']==row['success']
            reader=imageio.get_reader(path)
            try: assert reader.count_frames()==row['steps']+1
            finally: reader.close()
        audit[key]=dict(episodes=50,videos_verified=50,init_state_indices='0–49',rng_seed_once_per_task=7)
    write_json(ROOT/'results/official_audit.json',audit)
    print(json.dumps(audit,indent=2))

"""Use real pinned demonstrations to verify NativeDataset↔source teacher replay."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'scripts/adaptation'),str(Path(__file__).parent)]
from common import NativeDataset
from alignment_runtime import corpus_root,augmented
from alignment.corpus import Corpus
from alignment.data import image_hash

def test_actual_native_data_and_source_teacher_replay_match():
 for task in ['D0','D1']:
  dataset=NativeDataset(task);corpus=Corpus(corpus_root(),tasks=(task,))
  assert len(dataset)==len(corpus.records)
  for index in [0,37,len(dataset)-1]:
   for aug in range(4):
    transformed,identity=augmented(dataset,index,aug)
    row=corpus.records[index];reference=corpus.augmented(row,aug)
    assert identity['sample_id']==row['sample_id']
    assert identity['params']==reference['params']
    assert identity['image_sha256']==reference['image_sha256']
    for key,image in zip(['base_0_rgb','left_wrist_0_rgb'],reference['images']):np.testing.assert_array_equal(transformed['image'][key],image)
    raw=dataset.raw(index)
    np.testing.assert_array_equal(raw['actions'],reference['actions'])
    np.testing.assert_array_equal(raw['observation/state'],reference['state'])
  del dataset,corpus

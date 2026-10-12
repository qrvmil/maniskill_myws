import unittest
import importlib
from pathlib import Path

class EvaluationContracts(unittest.TestCase):
    def module(self):
        self.assertTrue(Path(__file__).with_name('eval_utils.py').exists(), 'Evaluation API must exist')
        return importlib.import_module('eval_utils')
    def test_reset_without_seed_preserves_task_rng_stream(self):
        m=self.module()
        class Env:
            seed_calls=[]
            def seed(self, value): self.seed_calls.append(value)
            def reset(self): return {}
            def get_sim_state(self): return [1.,2.]
            def step(self, action): return {},0,False,{}
        env=Env()
        m.reset_episode(env,None)
        self.assertEqual(env.seed_calls,[])

    def test_wilson_zero_success_is_not_zero_uncertainty(self):
        m=self.module(); result=m.summarize([{'success':False,'seed':s,'steps':220} for s in range(50)])
        self.assertEqual(result['success_rate'],0)
        self.assertEqual(result['ci95'][0],0)
        self.assertAlmostEqual(result['ci95'][1],0.0713476,places=6)
    def test_duplicate_episodes_rejected(self):
        m=self.module()
        with self.assertRaises(ValueError): m.summarize([{'success':True,'seed':1,'steps':2}]*2)
    def test_task_name_resolution_independent_of_index(self):
        m=self.module(); from types import SimpleNamespace
        tasks=[SimpleNamespace(name='another'),SimpleNamespace(name=m.TASKS['H2']['name'])]
        idx, task=m.find_task(SimpleNamespace(tasks=tasks),m.TASKS['H2']['name'])
        self.assertEqual(idx,1)
        with self.assertRaises(ValueError): m.find_task(SimpleNamespace(tasks=tasks),'unknown')
    def test_nonidle_gpu_memory_rejected_even_without_visible_pids(self):
        m=self.module()
        self.assertTrue(hasattr(m,'validate_idle_memory'))
        with self.assertRaises(RuntimeError): m.validate_idle_memory(1429)
        m.validate_idle_memory(1)

    def test_memory_units_distinguish_gb_and_gib(self):
        m=self.module(); r=m.memory_summary([{'used_mib':1},{'used_mib':901}],1)
        self.assertAlmostEqual(r['delta_gib'],900/1024)
        self.assertAlmostEqual(r['delta_gb'],900*1024**2/1e9)
    def test_invalid_episode_counts_rejected(self):
        m=self.module()
        with self.assertRaises(ValueError): m.validate_run(['D0'],0,'reference')
        with self.assertRaises(ValueError): m.validate_run(['D0'],51,'official')
        with self.assertRaises(ValueError): m.validate_run(['bogus'],1,'reference')

if __name__=='__main__': unittest.main()

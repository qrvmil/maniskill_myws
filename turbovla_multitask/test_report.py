import copy
import unittest
from data import TASKS, TRAIN_SETS
from report import summarize_cell, validate_evaluation_contract


def episode_rows(variant='A',task='D1',successes=6):
    return [dict(variant=variant,task=task,seed=10000+i,episode=i,success=i<successes,
                 steps=12,horizon=TASKS[task]['horizon'],
                 seen_during_training=task in TRAIN_SETS[variant],
                 actions=[[0.]*6+[-1.]]*12) for i in range(15)]


class ReportContracts(unittest.TestCase):
    def test_publication_requires_exact_checkpoint_and_full_reference_contract(self):
        metadata={'checkpoint_sha256':'checkpoint','stats_sha256':'stats'}
        config=dict(variant='A',train_tasks=['D0'],tasks=list(TASKS),episodes=15,
            protocol='reference',seed_start=10000,settling_steps=10,open_loop_steps=12,
            image_size=256,precision='fp32',checkpoint_sha256='checkpoint',stats_sha256='stats',
            primary_weights='ema_model_state_dict',all_selected_tasks_seen_by_release=False)
        validate_evaluation_contract(config,{'total_episodes':75,'config':config},'A',metadata)
        for field,value in [('protocol','official'),('variant','B'),('train_tasks',['D0','D1']),
                            ('primary_weights','model_state_dict'),('precision','bf16'),('open_loop_steps',8)]:
            changed=dict(config);changed[field]=value
            with self.assertRaises(ValueError):
                validate_evaluation_contract(changed,{'total_episodes':75,'config':changed},'A',metadata)

    def test_successes_are_recomputed_from_exact_episode_records(self):
        row=summarize_cell(episode_rows(),'A','D1')
        self.assertEqual(row['successes'],6)
        self.assertEqual(row['episodes'],15)
        self.assertEqual(row['success_rate'],.4)
        self.assertFalse(row['seen'])
        self.assertLess(row['ci95'][0],.4)
        self.assertGreater(row['ci95'][1],.4)

    def test_incomplete_duplicate_or_wrong_checkpoint_variant_cannot_enter_matrix(self):
        rows=episode_rows()
        with self.assertRaises(ValueError): summarize_cell(rows[:14],'A','D1')
        bad=copy.deepcopy(rows);bad[-1]['seed']=10000
        with self.assertRaises(ValueError): summarize_cell(bad,'A','D1')
        with self.assertRaises(ValueError): summarize_cell(rows,'B','D1')

    def test_string_failure_is_rejected_instead_of_counted_as_success(self):
        rows=episode_rows();rows[0]['success']='false'
        with self.assertRaises(ValueError): summarize_cell(rows,'A','D1')

    def test_impossible_or_wrong_task_horizon_is_rejected(self):
        rows=episode_rows();rows[0]['steps']=221
        with self.assertRaises(ValueError): summarize_cell(rows,'A','D1')
        rows=episode_rows();rows[0]['horizon']=300
        with self.assertRaises(ValueError): summarize_cell(rows,'A','D1')


if __name__ == '__main__':
    unittest.main()

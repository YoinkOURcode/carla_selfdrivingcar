"""Natural-frequency event fits, safe deadlines and learned neural probabilities."""

import numpy as np
import copy
import json
import pytest
import torch
from torch.nn import functional as F

from world_model_rl.factorized_termination import fit_natural_logistic, logistic_probabilities
from world_model_rl.data import Dataset, write_json
from world_model_rl.deadline import TrainingClock
from world_model_rl.model import RSSM
from world_model_rl.schema import Normalizer, EVENTS
from world_model_rl.training import save_checkpoint, load_checkpoint
from world_model_rl.tests.test_pipeline import cfg, make_episode


def factorized_config(cfg):
    cfg.model.deterministic, cfg.model.stochastic = 128, 32
    cfg.training.policy_observation_skip = cfg.training.termination_observation_skip = True
    cfg.training.termination_factorized = True
    cfg.training.refresh_updates = 0
    cfg.training.device = 'cpu'
    cfg.validate()
    return cfg


def test_natural_logistic_recovers_generated_probabilities_deterministically():
    rng = np.random.default_rng(18)
    x = torch.tensor(rng.normal(size=(5000, 2)), dtype=torch.float64)
    truth = torch.tensor([2., -1., -2.], dtype=torch.float64)
    probability = logistic_probabilities(x, truth)
    labels = rng.random(len(x)) < probability.numpy()
    weights, stats = fit_natural_logistic(x, labels, .1)
    repeated, _ = fit_natural_logistic(x, labels, .1)
    torch.testing.assert_close(weights, truth, atol=.15, rtol=0)
    torch.testing.assert_close(weights, repeated, atol=0, rtol=0)
    fitted = logistic_probabilities(x, weights)
    target = torch.tensor(labels, dtype=torch.float64)
    constant = torch.full_like(target, float(target.mean()))
    assert F.binary_cross_entropy(fitted, target) < F.binary_cross_entropy(constant, target)
    assert 0 < stats['updates'] <= 100


def test_newton_fit_does_not_commit_an_update_after_deadline():
    class Clock:
        checks = 0
        @property
        def expired(self):
            self.checks += 1
            return self.checks >= 3
    events = []
    _, stats = fit_natural_logistic(torch.arange(20, dtype=torch.float64)[:, None],
        np.arange(20) > 12, .01, Clock(), progress=lambda weights, count: events.append(count))
    assert stats['updates'] == 1 and events == [1]


def test_learned_arrival_failure_probabilities_have_exact_noisy_or_semantics(cfg):
    cfg = factorized_config(cfg)
    model = RSSM(cfg.model, termination_observation_skip=True, termination_factorized=True)
    episode = make_episode(cfg)
    obs = {key: torch.as_tensor(episode.arrays[key][:2]) for key in ('continuous','present','controls','light')}
    with torch.no_grad():
        for p in model.termination_head.parameters(): p.zero_()
        model.arrival_readout.weight.zero_()
        model.arrival_readout.bias.fill_(np.log(.2/.8))
        model.termination_logit_offset.fill_(np.log(.3/.7))
    logits = model.termination_logits(model.initial(2, 'cpu'), applied=torch.zeros(2,3), observation=obs)
    torch.testing.assert_close(logits.sigmoid(), torch.full((2,), 1 - .8*.7))


def test_factorized_checkpoint_loads_new_optimizer_group_and_rejects_silent_contract_change(cfg, tmp_path):
    cfg = factorized_config(cfg)
    model = RSSM(cfg.model, termination_observation_skip=True, termination_factorized=True)
    norm = Normalizer.fit([make_episode(cfg)])
    path = tmp_path / 'model.pt'
    save_checkpoint(path, cfg, model, norm, torch.optim.Adam(model.parameters()), 2, 20000)
    _, restored, _, optimizer = load_checkpoint(path, cfg, 'cpu')
    assert restored.termination_factorized and len(optimizer.param_groups[0]['params']) == len(list(model.parameters()))
    wrong = copy.deepcopy(cfg);wrong.training.termination_factorized = False
    with pytest.raises(ValueError, match='termination input contract'):
        load_checkpoint(path, wrong, 'cpu')


def test_full_factorized_recovery_preserves_source_and_checks_external_only_after_calibration(cfg, tmp_path, monkeypatch):
    from world_model_rl import factorized_termination as recovery, runner
    cfg = factorized_config(cfg)
    cfg.training.termination_factorized = cfg.training.termination_observation_skip = False
    write_json(tmp_path / 'config.json', cfg.to_dict())
    dataset = Dataset(tmp_path / 'data', cfg, create=True)
    for i in range(30):
        ep = make_episode(cfg, i, terminal=i%3 != 2)
        if i%3 != 2:
            ep.arrays['events'][-1, EVENTS.index('success' if i%3 == 0 else 'collision')] = 1
        ep.metadata['terminal_reason'] = ('success','collision','time_limit')[i%3]
        dataset.save(ep)
    model, norm = RSSM(cfg.model), Normalizer.fit(dataset.load('train'))
    save_checkpoint(tmp_path/'world_model.pt',cfg,model,norm,torch.optim.Adam(model.parameters()),1,20000)
    write_json(tmp_path/'world_model.validation.json',{'available':False})
    original = (tmp_path/'world_model.pt').read_bytes()
    clock=TrainingClock(tmp_path/'training_clock.json');clock.start();clock.close()
    monkeypatch.setattr(recovery,'FAILURE_COUNTS',(2,4))
    monkeypatch.setattr(recovery,'FAILURE_HIDDEN',8)
    monkeypatch.setattr(recovery,'CALIBRATION_UPDATES',2)
    def forbidden(*args,**kwargs): raise AssertionError('Offline readout fitting cannot drive CARLA')
    monkeypatch.setattr(runner,'CarlaDrivingEnv',forbidden)
    actual_evaluate=recovery.evaluate_model
    def evaluate_after_calibration(model,episodes,*args,**kwargs):
        assert (tmp_path/'factorized_termination/calibration_before_external_validation.json').exists()
        assert all(ep.metadata['split']=='validation' for ep in episodes)
        return actual_evaluate(model,episodes,*args,**kwargs)
    monkeypatch.setattr(recovery,'evaluate_model',evaluate_after_calibration)
    recovery.diagnose_arrivals(tmp_path)
    recovery.run(tmp_path)
    fitted_cfg=copy.deepcopy(cfg)
    fitted_cfg.training.termination_observation_skip=fitted_cfg.training.termination_factorized=True
    payload,restored,restored_norm,_=load_checkpoint(tmp_path/'factorized_termination/world_model_factorized.pt',fitted_cfg,'cpu')
    assert payload['model_updates']==20000 and restored_norm.to_dict()==norm.to_dict()
    for name,value in model.state_dict().items():
        if not name.startswith('termination_head.') and name!='termination_logit_offset':
            torch.testing.assert_close(restored.state_dict()[name],value,rtol=0,atol=0)
    assert (tmp_path/'factorized_termination/source_world_model.pt').read_bytes()==original
    assert 'components' in payload['validation']['termination']
    with pytest.raises(FileExistsError): recovery.run(tmp_path)

"""Training-only learned arrival and failure readouts for a frozen fresh RSSM."""

import argparse
import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import Config, experiment_config_path
from .data import Dataset, write_json
from .deadline import TrainingClock
from .model import arrival_features
from .model import RSSM, mlp
from .refit_termination import internal_partition, install_readout
from .runner import validation_suite
from .schema import EVENTS
from .training import (evaluate_model, imagination_gate, load_checkpoint, rng_state,
                       save_checkpoint, seed_everything, termination_examples, termination_metrics)


ARRIVAL_L2 = (1e-7, 1e-5, 0.001, 0.1)
FAILURE_COUNTS = (500, 1000, 2000, 4000)
FAILURE_HIDDEN = 32
CALIBRATION_UPDATES = 1000


def fit_natural_logistic(features, labels, l2, clock=None, progress=None, iterations=100):
    """Deterministic damped Newton fit at natural class frequency, in float64.

    A line search is read-only until an improving step is accepted. Deadline
    checks precede every committed update; no class balancing/threshold tuning.
    Returned coefficients include the bias as the final element.
    """
    x = torch.cat([features.double(), torch.ones(len(features), 1, dtype=torch.float64)], -1)
    y = torch.as_tensor(labels, dtype=torch.float64)
    weights = torch.zeros(x.shape[-1], dtype=torch.float64)
    penalty = torch.eye(x.shape[-1], dtype=torch.float64) * l2
    def objective(value):
        return F.binary_cross_entropy_with_logits(x @ value, y, reduction='sum') + .5 * l2 * value.square().sum()
    completed = 0
    for step in range(iterations):
        if clock is not None and clock.expired:
            break
        logits = x @ weights
        probabilities = logits.sigmoid()
        gradient = x.T @ (probabilities - y) + l2 * weights
        hessian = x.T @ ((probabilities * (1 - probabilities))[:, None] * x) + penalty
        direction = torch.linalg.solve(hessian, gradient)
        old = objective(weights)
        rate = 1.0
        for _ in range(50):
            proposal = weights - rate * direction
            if objective(proposal) <= old:
                break
            rate *= .5
        else:
            break
        if clock is not None and clock.expired:
            break
        weights = proposal
        completed += 1
        if progress is not None:
            progress(weights, completed)
        if float((rate * direction).abs().max()) < 1e-8:
            break
    return weights, {'updates': completed, 'natural_nll_sum': float(objective(weights)), 'l2': l2}


def logistic_probabilities(features, weights):
    return (features.double() @ weights[:-1] + weights[-1]).sigmoid()


def diagnose_arrivals(experiment):
    """Commit the design and select regularization using training episodes only."""
    root = Path(experiment)
    work = root / 'factorized_termination';work.mkdir(exist_ok=True)
    if (work / 'arrival_internal.json').exists():
        raise FileExistsError('Arrival diagnostic already complete; retain its committed result')
    cfg = Config.load(experiment_config_path(root))
    clock = TrainingClock(root / 'training_clock.json');clock.start()
    try:
        if clock.expired:
            raise RuntimeError('The shared training deadline has expired')
        torch.set_num_threads(4)
        train = Dataset(root / 'data', cfg).load('train')
        held, held_ids = internal_partition(train, cfg.training.seed + 5101)
        plan = {'method': 'learned-arrival-failure-factorization-v1', 'arrival_l2': ARRIVAL_L2,
            'created_utc': datetime.now(timezone.utc).isoformat(), 'device': 'cpu', 'threads': 4,
            'source_sha256': hashlib.sha256((root / 'world_model.pt').read_bytes()).hexdigest(),
            'source_config_sha256': hashlib.sha256(experiment_config_path(root).read_bytes()).hexdigest(),
            'diagnostic_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'arrival_fit': 'natural-frequency float64 damped Newton; 100 accepted steps maximum',
            'arrival_selection': 'internal training-episode natural log loss; stronger L2 breaks exact ties',
            'failure_head': {'hidden': FAILURE_HIDDEN, 'weight_decay': .001, 'latent_noise': 1.0,
                'updates': FAILURE_COUNTS, 'batch_size': 1024,
                'selection': 'internal natural log loss; retain selected fit rather than refitting',
                'calibration': 'two scalars on natural-frequency internal training holdout'},
            'held_out_training_episode_ids': [train[i].metadata['episode_id'] for i in held_ids],
            'training_episode_ids': [ep.metadata['episode_id'] for ep in train],
            'external_validation_used_for_fit_selection_or_calibration': False,
            'deadline_unix': clock.value['deadline_unix']}
        write_json(work / 'plan.json', plan)
        continuous = torch.as_tensor(np.concatenate([ep.arrays['continuous'][:-1] for ep in train]))
        applied = torch.as_tensor(np.concatenate([ep.arrays['applied_actions'] for ep in train]))
        labels = np.concatenate([ep.arrays['events'][:, EVENTS.index('success')] > 0 for ep in train])
        features = arrival_features(continuous, applied)
        records, best = [], None
        for l2 in ARRIVAL_L2:
            weights, stats = fit_natural_logistic(features[~held], labels[~held], l2, clock)
            if clock.expired:
                raise RuntimeError('Shared deadline reached during arrival fitting')
            probabilities = logistic_probabilities(features[held], weights)
            nll = float(F.binary_cross_entropy(probabilities, torch.as_tensor(labels[held], dtype=torch.float64)))
            record = {'l2': l2, 'natural_holdout_nll': nll,
                'metrics': termination_metrics(probabilities.numpy(), labels[held]), 'fit': stats}
            records.append(record)
            if best is None or (nll, -l2) < (best['natural_holdout_nll'], -best['l2']):
                best = record
            print(json.dumps(record), flush=True)
            write_json(work / 'arrival_internal.json', {'records': records, 'selected': best})
        weights, stats = fit_natural_logistic(features, labels, best['l2'], clock)
        if clock.expired:
            raise RuntimeError('Shared deadline reached during final arrival refit')
        torch.save({'weights': weights, 'fit': stats, 'selected': best}, work / 'arrival_readout.pt')
        write_json(work / 'arrival_training.json', {'fit': stats,
            'metrics': termination_metrics(logistic_probabilities(features, weights).numpy(), labels)})
        return best
    finally:
        clock.close()


def run(experiment):
    """Complete the committed training-only failure fit, then check external gates."""
    root = Path(experiment)
    work = root / 'factorized_termination'
    if not (work / 'arrival_readout.pt').exists():
        diagnose_arrivals(root)
    if (work / 'full_stage_plan.json').exists():
        raise FileExistsError('Factorized recovery already started; inspect its process and artifacts')
    plan = json.loads((work / 'plan.json').read_text())
    if hashlib.sha256((root / 'world_model.pt').read_bytes()).hexdigest() != plan['source_sha256']:
        raise ValueError('The committed source model changed')
    source_cfg = Config.load(experiment_config_path(root))
    cfg = copy.deepcopy(source_cfg)
    cfg.training.device = 'cpu'
    cfg.training.termination_context_skip = False
    cfg.training.termination_observation_skip = cfg.training.termination_factorized = True
    cfg.validate()
    clock = TrainingClock(root / 'training_clock.json');clock.start()
    try:
        if clock.expired:
            raise RuntimeError('Shared training deadline has expired')
        seed_everything(cfg.training.seed, cfg.training.deterministic_algorithms)
        torch.set_num_threads(4)
        source, _, normalizer, _ = load_checkpoint(root / 'world_model.pt', source_cfg, 'cpu')
        model = RSSM(cfg.model, termination_observation_skip=True, termination_factorized=True)
        missing = model.load_state_dict(source['model'], strict=False)
        if set(missing.missing_keys) != {'arrival_readout.weight', 'arrival_readout.bias'} or missing.unexpected_keys:
            raise ValueError('Unexpected architecture transition')
        arrival = torch.load(work / 'arrival_readout.pt', map_location='cpu', weights_only=False)
        with torch.no_grad():
            model.arrival_readout.weight.copy_(arrival['weights'][:-1].float()[None])
            model.arrival_readout.bias.copy_(arrival['weights'][-1:].float())
            for parameter in model.termination_head.parameters(): parameter.zero_()
        # Added parameters change optimizer group size. Retain the entire source
        # checkpoint/optimizer; the new, unused joint optimizer starts explicitly fresh.
        joint_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.learning_rate)
        shutil.copyfile(root / 'world_model.pt', work / 'source_world_model.pt')
        shutil.copyfile(root / 'world_model.validation.json', work / 'source_validation.json')
        write_json(work / 'source_config.json', source_cfg.to_dict())
        write_json(work / 'config.json', cfg.to_dict())
        full_plan = {'committed_plan': plan, 'calibration_updates': CALIBRATION_UPDATES,
            'calibration_learning_rate': .03, 'source_checkpoint': str(work / 'source_world_model.pt'),
            'arrival_weights_sha256': hashlib.sha256((work / 'arrival_readout.pt').read_bytes()).hexdigest(),
            'optimizer_transition': 'New joint optimizer for added arrival parameters; body frozen; original optimizer preserved in source checkpoint',
            'config': cfg.to_dict(), 'clock': clock.summary()}
        write_json(work / 'full_stage_plan.json', full_plan)
        dataset = Dataset(root / 'data', source_cfg)
        train = dataset.load('train')
        held, held_indices = internal_partition(train, cfg.training.seed + 5101)
        if [train[i].metadata['episode_id'] for i in held_indices] != plan['held_out_training_episode_ids']:
            raise ValueError('Training-internal episode partition changed')
        inputs, terminal_labels = termination_examples(model, train, normalizer, cfg, 'cpu')
        success_labels = np.concatenate([ep.arrays['events'][:, EVENTS.index('success')] > 0 for ep in train])
        labels = terminal_labels & ~success_labels
        fit = np.flatnonzero(~held)
        positive, negative = fit[labels[fit]], fit[~labels[fit]]
        if not len(positive) or not len(negative):
            raise ValueError('Failure readout lacks positive/negative training support')
        center = torch.as_tensor(inputs[fit].mean(0))
        scale = torch.as_tensor(np.maximum(inputs[fit].std(0), 1e-3))
        x = (torch.as_tensor(inputs) - center) / scale
        spec = plan['failure_head']
        if spec['hidden'] > cfg.model.hidden:
            raise ValueError('Failure readout exceeds original MLP capacity')
        torch.manual_seed(cfg.training.seed + 7300)
        generator = np.random.default_rng(cfg.training.seed + 7400)
        head = mlp(inputs.shape[1], 1, spec['hidden'])
        optimizer = torch.optim.Adam(head.parameters(), lr=cfg.model.learning_rate, weight_decay=spec['weight_decay'])
        offset = float(np.log(len(positive) / len(negative)))
        records, best = [], None
        diagnostic = json.loads((work / 'arrival_internal.json').read_text())
        total = sum(row['fit']['updates'] for row in diagnostic['records']) + arrival['fit']['updates']
        report = {'available': False, 'reason': 'Factorized recovery is not validated'}
        output = work / 'world_model_factorized.pt'
        def install(a=1., b=0.):
            install_readout(model.termination_head, head, center, scale)
            with torch.no_grad():
                model.termination_head[4].weight.mul_(a);model.termination_head[4].bias.mul_(a)
                model.termination_logit_offset.fill_(a * offset + b)
        def save(stage, force=False):
            if not (force or clock.checkpoint_due): return
            save_checkpoint(output, cfg, model, normalizer, joint_optimizer,
                source['model_version'] + 1, source['model_updates'], kind='world_model',
                source_checkpoint=str(work / 'source_world_model.pt'), normalizer_frozen=True,
                training_episode_ids=[ep.metadata['episode_id'] for ep in train],
                termination_updates=source.get('termination_updates', 0) + total,
                validation=report, training_clock=clock.summary(), fit_stage=stage,
                rng=rng_state(generator), factorized_plan=full_plan,
                readout_optimizer={'state': optimizer.state_dict(), 'center': center, 'scale': scale,
                    'parameterization': 'standardized inputs before folding'},
                optimizer_transition=full_plan['optimizer_transition'])
            clock.record_checkpoint(output, stage, {'model_updates': source['model_updates'],
                'termination_updates': source.get('termination_updates', 0) + total}, force=True)
        install();save('factorized_initial', True)
        batch_size = spec['batch_size']
        for update in range(1, max(spec['updates']) + 1):
            if clock.expired:
                install();save('deadline', True)
                raise RuntimeError('Shared deadline reached during failure readout fitting')
            indices = np.concatenate([generator.choice(positive, batch_size // 2),
                generator.choice(negative, batch_size - batch_size // 2)])
            batch = x[indices]
            noise = torch.randn_like(batch) * spec['latent_noise']
            noise[:, -model.termination_physical_size:] = 0
            target = torch.as_tensor(labels[indices], dtype=torch.float32)
            loss = F.binary_cross_entropy_with_logits(head(batch + noise).squeeze(-1), target)
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.model.gradient_clip, error_if_nonfinite=True)
            optimizer.step();total += 1
            if clock.checkpoint_due:
                install();save('failure_readout')
            if update in spec['updates']:
                with torch.no_grad():
                    logits = head(x[held]).squeeze(-1) + offset
                    nll = float(F.binary_cross_entropy_with_logits(logits, torch.tensor(labels[held], dtype=torch.float32)))
                record = {'updates': update, 'natural_holdout_nll': nll,
                    'metrics': termination_metrics(logits.sigmoid().numpy(), labels[held])}
                records.append(record)
                if best is None or nll < best['record']['natural_holdout_nll']:
                    best = {'record': record, 'state': copy.deepcopy(head.state_dict()),
                            'optimizer': copy.deepcopy(optimizer.state_dict()), 'rng': rng_state(generator)}
                write_json(work / 'failure_internal.json', {'records': records, 'selected': best['record']})
                print(json.dumps(record), flush=True)
        head.load_state_dict(best['state'])
        optimizer.load_state_dict(best['optimizer'])
        generator.bit_generator.state = best['rng']['generator']
        torch.set_rng_state(best['rng']['torch'])
        torch.save({**best, 'center': center, 'scale': scale, 'logit_offset': offset}, work / 'failure_readout.pt')
        with torch.no_grad():
            calibration_logits = head(x[held]).squeeze(-1) + offset
        target = torch.as_tensor(labels[held], dtype=torch.float32)
        raw_scale = torch.nn.Parameter(torch.tensor(float(np.log(np.expm1(1.)))))
        intercept = torch.nn.Parameter(torch.zeros(()))
        calibration_optimizer = torch.optim.Adam([raw_scale, intercept], lr=.03)
        for update in range(CALIBRATION_UPDATES):
            if clock.expired:
                install(float(F.softplus(raw_scale).detach()), float(intercept.detach()));save('deadline', True)
                raise RuntimeError('Shared deadline reached during failure probability calibration')
            loss = F.binary_cross_entropy_with_logits(F.softplus(raw_scale) * calibration_logits + intercept, target)
            calibration_optimizer.zero_grad(set_to_none=True);loss.backward();calibration_optimizer.step();total += 1
            if clock.checkpoint_due:
                install(float(F.softplus(raw_scale).detach()), float(intercept.detach()));save('failure_calibration')
        a, b = float(F.softplus(raw_scale).detach()), float(intercept.detach())
        install(a, b)
        with torch.no_grad():
            calibration = {'scale': a, 'intercept': b, 'metrics': termination_metrics(
                (a * calibration_logits + b).sigmoid().numpy(), labels[held]), 'updates': CALIBRATION_UPDATES}
        write_json(work / 'calibration_before_external_validation.json', calibration)
        torch.save(calibration_optimizer.state_dict(), work / 'calibration_optimizer.pt')
        save('factorized_complete', True)
        validation = validation_suite(dataset.load('validation'), cfg)
        stats = {'method': plan['method'], 'updates': total,
            'arrival': json.loads((work / 'arrival_training.json').read_text()),
            'arrival_internal': diagnostic['selected'], 'failure_internal': best['record'],
            'failure_calibration': calibration,
            'failure_labels': 'task termination excluding actual success events; includes collisions/offroad',
            'failure_fit_episode_ids': [ep.metadata['episode_id'] for i, ep in enumerate(train) if i not in held_indices]}
        report = evaluate_model(model, validation, normalizer, cfg, 'cpu', termination_fit=stats)
        passed, reason = imagination_gate(report, cfg)
        save('validated_factorized', True)
        # Store fitting diagnostics in the validated payload without changing weights.
        final = torch.load(output, map_location='cpu', weights_only=False)
        final.update(termination_fit=stats, validation_episode_ids=[ep.metadata['episode_id'] for ep in validation])
        temporary = output.with_suffix('.pt.tmp');torch.save(final, temporary);temporary.replace(output)
        write_json(output.with_suffix('.validation.json'), report)
        write_json(work / 'outcome.json', {'gate_passed': passed, 'gate_reason': reason})
        clock.record_checkpoint(output, 'factorized_reported', {'model_updates': source['model_updates'],
            'termination_updates': source.get('termination_updates', 0) + total}, force=True)
        if passed:
            temporary = (root / 'world_model.pt').with_suffix('.pt.tmp')
            shutil.copyfile(output, temporary);temporary.replace(root / 'world_model.pt')
            write_json(root / 'world_model.validation.json', report)
            write_json(experiment_config_path(root), cfg.to_dict())
        print(f'Factorized gate: {passed} ({reason})', flush=True)
        return passed
    finally:
        clock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--diagnose-arrivals', action='store_true')
    args = parser.parse_args()
    if args.diagnose_arrivals:
        diagnose_arrivals(args.experiment)
    else:
        raise SystemExit(0 if run(args.experiment) else 2)

"""Cross-fitted event ensemble, calibrated exclusively on training episodes."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import Config, experiment_config_path
from .data import Dataset, write_json
from .deadline import TrainingClock
from .model import mlp
from .runner import validation_suite
from .training import (choose_device, clear_termination_moments, evaluate_model,
                       imagination_gate, load_checkpoint, rng_state, save_checkpoint,
                       seed_everything, termination_examples, termination_metrics)


def install_ensemble(destination, members):
    """Embed an average of smaller standardized ELU MLPs into the original MLP."""
    if sum(head[0].out_features for head, _, _ in members) > destination[0].out_features:
        raise ValueError("Ensemble exceeds the existing readout capacity")
    with torch.no_grad():
        for parameter in destination.parameters():
            parameter.zero_()
        start = 0
        for head, center, scale in members:
            width = head[0].out_features
            block = slice(start, start + width)
            destination[0].weight[block].copy_(head[0].weight / scale)
            destination[0].bias[block].copy_(head[0].bias - (head[0].weight / scale) @ center)
            destination[2].weight[block, block].copy_(head[2].weight)
            destination[2].bias[block].copy_(head[2].bias)
            destination[4].weight[:, block].copy_(head[4].weight / len(members))
            destination[4].bias.add_(head[4].bias / len(members))
            start += width


def episode_folds(episodes, count, seed):
    if any(ep.metadata['split'] != 'train' for ep in episodes):
        raise ValueError('Calibration accepts training episodes only')
    rng, groups = np.random.default_rng(seed), {}
    for index, episode in enumerate(episodes):
        groups.setdefault(episode.metadata.get('terminal_reason', 'unspecified'), []).append(index)
    assigned = np.empty(len(episodes), dtype=np.int64)
    for group in groups.values():
        assigned[rng.permutation(group)] = np.arange(len(group)) % count
    return np.concatenate([np.full(len(ep), assigned[i], dtype=np.int64)
                           for i, ep in enumerate(episodes)]), assigned


def run(experiment):
    root = Path(experiment)
    work = root / 'termination_calibration'
    work.mkdir(exist_ok=True)
    if (work / 'plan.json').exists():
        raise FileExistsError('Cross-fit recovery already started; inspect its artifacts')
    cfg = Config.load(experiment_config_path(root))
    clock = TrainingClock(root / 'training_clock.json');clock.start()
    try:
        if clock.expired:
            raise RuntimeError('The shared training deadline has expired')
        seed_everything(cfg.training.seed, cfg.training.deterministic_algorithms)
        device = choose_device(cfg.training.device)
        payload, model, normalizer, joint_optimizer = load_checkpoint(root / 'world_model.pt', cfg, device)
        if model.termination_factorized:
            raise ValueError('Use factorized_termination to preserve separately labelled event semantics')
        previous = json.loads((root / 'termination_refit/selected_before_external_validation.json').read_text())
        spec, updates = previous['spec'], previous['updates']
        if spec['hidden'] * 5 > cfg.model.hidden:
            raise ValueError('Selected five-member ensemble does not fit the existing MLP')
        train = Dataset(root / 'data', cfg).load('train')
        masks, assignments = episode_folds(train, 5, cfg.training.seed + 6100)
        plan = {'method': 'fivefold-natural-calibration-v1', 'spec': spec, 'updates_per_fold': updates,
            'folds': 5, 'batch_size': 1024, 'calibration_updates': 1000,
            'calibration_learning_rate': 0.03, 'readout_learning_rate': cfg.model.learning_rate,
            'source_sha256': hashlib.sha256((root / 'world_model.pt').read_bytes()).hexdigest(),
            'fold_episode_ids': {str(i): [ep.metadata['episode_id'] for ep, fold in zip(train, assignments) if fold == i]
                                 for i in range(5)},
            'calibration': 'positive logit scale and intercept; natural-frequency out-of-fold BCE',
            'external_validation_used_for_selection_or_calibration': False,
            'limitation': 'RSSM representation was fitted on all training data; head predictions are out of fold'}
        write_json(work / 'plan.json', plan)
        inputs, labels = termination_examples(model, train, normalizer, cfg, device)
        features = torch.as_tensor(inputs, device=device)
        oof = np.empty(len(inputs), dtype=np.float32)
        members, offsets, fold_states, total = [], [], [], 0
        output = work / 'world_model_calibrated.pt'
        generator = np.random.default_rng(cfg.training.seed + 6200)
        def save(stage, scale=1.0, shift=0.0):
            install_ensemble(model.termination_head, members)
            with torch.no_grad():
                model.termination_head[4].weight.mul_(scale)
                model.termination_head[4].bias.mul_(scale)
                model.termination_logit_offset.fill_(float(np.mean(offsets)) * scale + shift)
            save_checkpoint(output, cfg, model, normalizer, joint_optimizer,
                payload['model_version'] + 1, payload['model_updates'], kind='world_model',
                source_checkpoint=str(root / 'termination_refit/baseline.pt'),
                termination_updates=payload.get('termination_updates', 0) + total,
                training_clock=clock.summary(), fit_stage=stage,
                termination_calibration_plan=plan, rng=rng_state(generator),
                training_episode_ids=[ep.metadata['episode_id'] for ep in train])
            clock.record_checkpoint(output, stage, {'model_updates': payload['model_updates'],
                'termination_updates': payload.get('termination_updates', 0) + total}, force=True)
        for fold in range(5):
            torch.manual_seed(cfg.training.seed + 6300 + fold)
            generator = np.random.default_rng(cfg.training.seed + 6400 + fold)
            fit = np.flatnonzero(masks != fold)
            held = np.flatnonzero(masks == fold)
            center = features[fit].mean(0)
            scale = features[fit].std(0, correction=0).clamp_min(1e-3)
            x = (features - center) / scale
            positive, negative = fit[labels[fit]], fit[~labels[fit]]
            if not len(positive) or not len(negative) or not len(held):
                raise ValueError('Insufficient support in a cross-fitting fold')
            offset = float(np.log(len(positive) / len(negative)))
            head = mlp(inputs.shape[1], 1, spec['hidden']).to(device)
            optimizer = torch.optim.Adam(head.parameters(), lr=cfg.model.learning_rate,
                                         weight_decay=spec['weight_decay'])
            members.append((head, center, scale));offsets.append(offset)
            for update in range(updates):
                if clock.expired:
                    save('deadline')
                    raise RuntimeError('Shared deadline reached during cross-fitting')
                indices = np.concatenate([generator.choice(positive, 512), generator.choice(negative, 512)])
                batch = x[indices]
                batch = batch + torch.randn_like(batch) * spec['noise']
                target = torch.as_tensor(labels[indices], device=device, dtype=torch.float32)
                loss = nn.functional.binary_cross_entropy_with_logits(head(batch).squeeze(-1), target)
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), cfg.model.gradient_clip, error_if_nonfinite=True)
                optimizer.step();total += 1
                if clock.checkpoint_due:
                    save('termination_crossfit')
            with torch.no_grad():
                logits = torch.cat([head(chunk).squeeze(-1) + offset for chunk in x[held].split(4096)])
                oof[held] = logits.cpu().numpy()
            fold_states.append({'head': head.state_dict(), 'center': center.cpu(), 'scale': scale.cpu(),
                'optimizer': optimizer.state_dict(), 'rng': rng_state(generator), 'logit_offset': offset})
            print(f'Cross-fit fold {fold + 1}/5: {termination_metrics(torch.sigmoid(logits).cpu().numpy(), labels[held])}', flush=True)
        np.savez_compressed(work / 'out_of_fold_predictions.npz', logits=oof, labels=labels, fold=masks)
        torch.save(fold_states, work / 'fold_readouts.pt')
        # Fit probabilities to natural prevalence; no balancing, threshold selection,
        # validation observations, or validation labels enter these 2 scalar parameters.
        logits = torch.as_tensor(oof, device=device)
        target = torch.as_tensor(labels, device=device, dtype=torch.float32)
        raw_scale = nn.Parameter(torch.tensor(float(np.log(np.expm1(1.0))), device=device))
        shift = nn.Parameter(torch.zeros((), device=device))
        calibration_optimizer = torch.optim.Adam([raw_scale, shift], lr=0.03)
        for update in range(1000):
            a = nn.functional.softplus(raw_scale)
            if clock.expired:
                save('deadline', float(a.detach()), float(shift.detach()))
                raise RuntimeError('Shared deadline reached during natural-frequency calibration')
            loss = nn.functional.binary_cross_entropy_with_logits(a * logits + shift, target)
            calibration_optimizer.zero_grad(set_to_none=True);loss.backward();calibration_optimizer.step()
            total += 1
            if clock.checkpoint_due:
                save('termination_calibration', float(nn.functional.softplus(raw_scale).detach()), float(shift.detach()))
        a, b = float(nn.functional.softplus(raw_scale).detach()), float(shift.detach())
        with torch.no_grad():
            before = termination_metrics(logits.sigmoid().cpu().numpy(), labels)
            after = termination_metrics((a * logits + b).sigmoid().cpu().numpy(), labels)
        calibration = {'scale': a, 'intercept': b, 'before': before, 'after': after,
            'optimizer': calibration_optimizer.state_dict(), 'updates': 1000}
        torch.save(calibration, work / 'calibration_state.pt')
        write_json(work / 'calibration_before_external_validation.json', {k:v for k,v in calibration.items() if k != 'optimizer'})
        save('calibrated_crossfit_complete', a, b)
        clear_termination_moments(joint_optimizer, model)
        validation = validation_suite(Dataset(root / 'data', cfg).load('validation'), cfg)
        stats = {'method': plan['method'], 'updates': total, 'plan': plan,
                 'natural_out_of_fold_metrics': after, 'calibration_scale': a, 'calibration_intercept': b}
        report = evaluate_model(model, validation, normalizer, cfg, device, termination_fit=stats)
        passed, reason = imagination_gate(report, cfg)
        # Save final validated payload, retaining the exact same already-fitted weights.
        final = torch.load(output, map_location='cpu', weights_only=False)
        final.update(validation=report, termination_fit=stats, model_optimizer=joint_optimizer.state_dict())
        temporary = output.with_suffix('.pt.tmp');torch.save(final, temporary);temporary.replace(output)
        write_json(output.with_suffix('.validation.json'), report)
        write_json(work / 'outcome.json', {'gate_passed': passed, 'gate_reason': reason})
        clock.record_checkpoint(output, 'validated_calibrated_readout', {'model_updates': payload['model_updates'],
            'termination_updates': payload.get('termination_updates', 0) + total}, force=True)
        if passed:
            temporary = (root / 'world_model.pt').with_suffix('.pt.tmp')
            shutil.copyfile(output, temporary);temporary.replace(root / 'world_model.pt')
            write_json(root / 'world_model.validation.json', report)
        print(f'Calibrated cross-fit gate: {passed} ({reason})', flush=True)
        return passed
    finally:
        clock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    raise SystemExit(0 if run(parser.parse_args().experiment) else 2)

"""Independent frozen-representation probe; PAD uses weighted 0-1 errors only."""
import hashlib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def group_probe_split(domains, groups, seed=19):
    """60/20/20 by whole groups within each domain, fixed without feature scores."""
    d, g = np.asarray(domains), np.asarray(groups)
    assignment = {}
    for domain in (0,1):
        own = set(g[d==domain])
        if own & set(g[d!=domain]): raise ValueError('A group crosses source/target')
        ordered = sorted(own, key=lambda v:hashlib.sha256(f'{seed}:{v}'.encode()).hexdigest())
        if len(ordered) < 3: raise ValueError('Need at least three groups per domain for probe splitting')
        n_train = min(len(ordered)-2, max(1,int(.6*len(ordered))))
        n_val = min(len(ordered)-n_train-1,max(1,int(.2*len(ordered))))
        for i, group in enumerate(ordered):
            assignment[group] = 'train' if i<n_train else ('validation' if i<n_train+n_val else 'evaluation')
    return np.asarray([assignment[v] for v in g])


def weighted_error(domains, prediction, weights):
    errors, effective, mass = {}, {}, {}
    for domain in (0,1):
        take = domains==domain; w = weights[take]
        if not len(w) or w.sum() <= 0: raise ValueError('Missing domain support for error rate')
        errors[str(domain)] = float(np.sum(w*(prediction[take]!=domain))/w.sum())
        effective[str(domain)] = float(w.sum()**2/np.sum(w*w))
        mass[str(domain)] = float(w.sum())
    epsilon = .5*(errors['0']+errors['1'])
    raw = 2*(1-2*epsilon)
    return dict(epsilon_bal=epsilon, domain_error=errors, effective_samples=effective, mass=mass,
                raw_pad=raw, scheduling_pad=max(0.,raw),
                status='above_chance_error' if epsilon>.5 else 'PASS')


def run_probe(features, domains, groups, splits, roles, weights=None, *, seed=19, minimum_effective=2., minimum_mass=.1):
    """roles are source_train/source_validation/target_adapt, never final_evaluation.

    Group assignment is supplied/frozen before fitting. Only validation decides
    prediction direction; evaluation direction is never flipped after scoring.
    `weights` can be detached soft conditional membership (not target truth).
    """
    x = features.detach().cpu().numpy() if torch.is_tensor(features) else np.asarray(features)
    x = np.asarray(x,dtype=np.float64)
    d, g, split, role = map(np.asarray,(domains,groups,splits,roles))
    w = np.ones(len(d)) if weights is None else (weights.detach().cpu().numpy() if torch.is_tensor(weights) else np.asarray(weights))
    w = np.asarray(w,dtype=float)
    if x.ndim!=2 or len(x)!=len(d) or any(a.shape!=(len(d),) for a in (d,g,split,role,w)):
        raise ValueError('Probe inputs have inconsistent dimensions')
    if not np.isfinite(x).all() or not np.isfinite(w).all() or (w<0).any() or not set(d)<={0,1}:
        raise ValueError('Invalid probe features/domain labels/weights')
    if set(split)!={'train','validation','evaluation'}:
        raise ValueError('Probe requires fixed train/validation/evaluation splits')
    if not set(role)<={'source_train','source_validation','target_adapt'}:
        raise ValueError('Probe cannot consume final test data')
    if np.any((role=='target_adapt')!=(d==0)): raise ValueError('source=1 / target=0 role mismatch')
    for group in set(g):
        take = g==group
        if len(set(split[take]))!=1 or len(set(d[take]))!=1:
            raise ValueError('Probe group leakage across splits/domains')
    support = {}
    for part in ('train','validation','evaluation'):
        for domain in (0,1):
            v = w[(split==part)&(d==domain)]
            ess = float(v.sum()**2/(np.sum(v*v))) if np.any(v>0) else 0.
            support[f'{part}/{domain}'] = dict(rows=int((v>0).sum()),mass=float(v.sum()),effective=ess)
    if minimum_effective < 1 or minimum_mass <= 0:
        raise ValueError('Positive probe support thresholds required')
    if any(v['effective']+1e-8<minimum_effective or v['mass']<minimum_mass for v in support.values()):
        return dict(status='UNSUPPORTED',raw_pad=None,scheduling_pad=None,support=support,
                    reason='insufficient effective mass in a grouped probe split')
    train, val, test = (split==p for p in ('train','validation','evaluation'))
    # Domain-mass normalization is unrelated to task class balance.
    fit_weight = w[train].copy()
    for domain in (0,1):
        take = d[train]==domain
        fit_weight[take] *= train.sum()/(2*fit_weight[take].sum())
    scaler = StandardScaler().fit(x[train],sample_weight=fit_weight)
    classifier = LogisticRegression(C=1.,solver='lbfgs',max_iter=500,random_state=seed)
    classifier.fit(scaler.transform(x[train]),d[train],sample_weight=fit_weight)
    val_pred = classifier.predict(scaler.transform(x[val]))
    validation = weighted_error(d[val],val_pred,w[val])
    flipped = validation['epsilon_bal']>.5
    evaluation_pred = classifier.predict(scaler.transform(x[test]))
    if flipped: evaluation_pred = 1-evaluation_pred
    result = weighted_error(d[test],evaluation_pred,w[test])
    result.update(support=support, direction_flipped_on_validation=flipped,
                  validation_unflipped=validation,
                  coefficients=classifier.coef_.tolist(),intercept=classifier.intercept_.tolist(),
                  scaler_mean=scaler.mean_.tolist(),scaler_scale=scaler.scale_.tolist(),
                  seed=seed, fixed_C=1., evaluation_predictions=[dict(index=int(i),group=str(g[i]),
                  domain=int(d[i]),prediction=int(p),weight=float(w[i])) for i,p in zip(np.where(test)[0],evaluation_pred)],
                  group_splits={str(v):str(split[np.where(g==v)[0][0]]) for v in set(g)})
    return result


def probe_alignment(source, target, source_groups, target_groups, source_members, target_members,
                    seed=19, minimum_effective=2.):
    """Probe exactly the common pattern/condition regions, with reported coverage.

    Freeze a single group partition for both global and local probes. Distance
    aggregation averages qualified regions; missing region estimates are explicit.
    This uses training/adaptation features only, not final target test features.
    """
    from .adaptation import pattern_ids
    groups = np.asarray(list(source_groups)+list(target_groups))
    domains = np.r_[np.ones(len(source_groups),dtype=int),np.zeros(len(target_groups),dtype=int)]
    roles = np.asarray(['source_train']*len(source_groups)+['target_adapt']*len(target_groups))
    splits = group_probe_split(domains,groups,seed)
    patterns = torch.cat([pattern_ids(source['availability']),pattern_ids(target['availability'])]).cpu().numpy()
    members = torch.cat([source_members.detach(),target_members.detach()]).cpu().numpy()
    global_features = torch.cat([source['z_adv'],target['z_adv']]).detach()
    local_features = torch.cat([source['psi'],target['psi']]).detach()
    report = dict(global_regions=[],local_regions=[],group_splits=dict(zip(groups.tolist(),splits.tolist())))
    for pattern in sorted(set(patterns)-{0}):
        keep = (patterns==pattern).astype(float)
        kwargs = dict(domains=domains,groups=groups,splits=splits,roles=roles,seed=seed,minimum_effective=minimum_effective)
        result = run_probe(global_features,weights=keep,**kwargs)
        report['global_regions'].append(dict(pattern=int(pattern),probe=result))
        for k in range(members.shape[1]):
            result = run_probe(local_features,weights=keep*members[:,k],**kwargs)
            report['local_regions'].append(dict(pattern=int(pattern),condition=k,probe=result))
    global_valid = [r['probe']['scheduling_pad'] for r in report['global_regions'] if r['probe']['scheduling_pad'] is not None]
    local_valid = [r['probe']['scheduling_pad'] for r in report['local_regions'] if r['probe']['scheduling_pad'] is not None]
    report.update(global_distance=float(np.mean(global_valid)) if global_valid else None,local_distances=local_valid,
                  global_qualified=len(global_valid),local_qualified=len(local_valid),
                  aggregation='mean of qualified pattern/condition regions; unsupported regions are not zeros')
    return report

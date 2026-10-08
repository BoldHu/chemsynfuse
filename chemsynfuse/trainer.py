"""Tensor-only downstream training, EMA selection and adaptive domain alignment."""
import copy
from dataclasses import dataclass
import math
import torch
from torch.nn import functional as F
from sklearn.metrics import f1_score
from .adaptation import ConditionalAdapter, EMATeacher, OmegaScheduler, grl_schedule
from .domain_probe import probe_alignment


@dataclass
class TrainingConfig:
    steps: int = 2000
    warmup: int = 400
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    ema_decay: float = .99
    omega_initial: float = .5
    omega_decay: float = .9
    adversarial_max: float = 1.
    validation_every: int = 25
    probe_every: int = 100
    seed: int = 11

    def __post_init__(self):
        if not 0 <= self.warmup < self.steps or min(self.validation_every, self.probe_every) < 1:
            raise ValueError('Invalid training schedule')
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError('Positive finite learning rate required')


class Trainer:
    """Accepts prepared tensors; target labels are never accepted.

    For regression, pass RegressionNeighborhoods initialized with externally
    source-fitted mean, scale, anchors, bandwidth, minimum and maximum.
    Source labels passed to step/fit remain in original endpoint units.
    """
    def __init__(self, model, config=None, neighborhoods=None):
        self.model = model
        self.config = config or TrainingConfig()
        self.neighborhoods = neighborhoods
        if model.kind == 'regression' and neighborhoods is None:
            raise ValueError('Source-fitted regression neighborhood state required')
        conditions = len(neighborhoods.state['anchors']) if neighborhoods else model.classes
        self.adapter = ConditionalAdapter(model.fusion.dim, conditions).to(next(model.parameters()).device)
        self.teacher = EMATeacher(model, self.config.ema_decay)
        self.schedule = OmegaScheduler(self.config.omega_initial, self.config.omega_decay)
        self.parameters = list(model.parameters()) + list(self.adapter.parameters())
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.config.learning_rate,
                                          weight_decay=self.config.weight_decay)
        self.steps_done = 0

    def members(self, prediction):
        return self.neighborhoods.members(prediction.flatten())[0] if self.neighborhoods else prediction.softmax(-1).detach()

    def source_members(self, labels):
        if self.neighborhoods:
            return self.neighborhoods.members(self.neighborhoods.standardize(labels))[0]
        return F.one_hot(labels.long(), self.model.classes).float()

    @staticmethod
    def check_features(batch):
        allowed = {'values', 'states', 'available', 'types', 'sections', 'categories',
                   'molecules', 'molecule_meta', 'molecule_mask', 'molecule_summary'}
        if set(batch) != allowed:
            raise ValueError('Only the documented feature tensors are accepted; labels are separate')
        if not batch['available'].any(1).logical_or(batch['molecule_mask'].any(1)).all():
            raise ValueError('Every record needs at least one available input')

    def step(self, source, source_labels, target):
        self.check_features(source)
        self.check_features(target)
        c = self.config
        if self.steps_done >= c.steps:
            raise ValueError('Configured training budget exhausted')
        active = self.steps_done >= c.warmup
        if self.steps_done == c.warmup:
            self.teacher.model.load_state_dict(self.model.state_dict())
        strength = grl_schedule((self.steps_done + 1 - c.warmup) / (c.steps - c.warmup), c.adversarial_max) if active else 0.
        self.model.train()
        self.adapter.train()
        self.optimizer.zero_grad(set_to_none=True)
        so = self.model.fusion(source, adversarial=active, stop_gate_grad=True)
        sp = self.model.head(so['fused'])
        if self.neighborhoods:
            y = self.neighborhoods.standardize(source_labels.flatten())
            supervised = 2 * F.huber_loss(sp.flatten(), y, delta=1.)
        else:
            supervised = F.cross_entropy(sp, source_labels.long())
        domain = supervised * 0
        info = {}
        if active:
            to = self.model.fusion(target, adversarial=True, stop_gate_grad=True)
            with torch.no_grad():
                qt = self.members(self.teacher(target))
            domain, info = self.adapter(so, to, self.source_members(source_labels), qt,
                                       lambda_adv=strength, omega_local=self.schedule.omega_local)
        loss = supervised + domain
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training objective')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1.)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient')
        self.optimizer.step()
        self.teacher.update(self.model)
        self.steps_done += 1
        return dict(step=self.steps_done, supervised_loss=float(supervised.detach()),
                    domain_loss=float(domain.detach()), **info)

    @torch.no_grad()
    def update_omega(self, source, source_labels, target, source_groups, target_groups):
        """Use source-training and target-adaptation features, never target evaluation."""
        self.check_features(source)
        self.check_features(target)
        was_training = self.model.training
        self.model.eval()
        try:
            so = self.model.fusion(source, adversarial=True)
            to = self.model.fusion(target, adversarial=True)
            report = probe_alignment(so, to, source_groups, target_groups,
                                     self.source_members(source_labels), self.members(self.teacher(target)),
                                     seed=self.config.seed)
            if report['global_distance'] is not None and report['local_distances']:
                self.schedule.update(report['global_distance'], report['local_distances'])
            return report
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def validation_score(self, loader):
        predictions, labels = [], []
        for batch, y in loader:
            self.check_features(batch)
            predictions.append(self.teacher(batch).detach().cpu())
            labels.append(y.detach().cpu())
        if not predictions:
            raise ValueError('Source validation loader is empty')
        p, y = torch.cat(predictions), torch.cat(labels)
        if self.neighborhoods:
            p = p.flatten() * self.neighborhoods.state['scale'] + self.neighborhoods.state['mean']
            return float((p - y.flatten()).square().mean().sqrt())
        return -float(f1_score(y.numpy(), p.argmax(-1).numpy(), average='macro',
                              labels=list(range(self.model.classes)), zero_division=0))

    def fit(self, source_loader, target_loader, source_validation_loader, probe_callback):
        """Return the best source-validation EMA state and scalar training history.

        Loaders supply device-ready (features, labels) for source and features
        only for target. probe_callback(trainer) calls update_omega with a fixed
        original-source/target-adaptation subset and disjoint group identifiers.
        """
        def repeat(loader):
            while True:
                seen = False
                for item in loader:
                    seen = True
                    yield item
                if not seen:
                    raise ValueError('Training loader is empty')
        source, target = repeat(source_loader), repeat(target_loader)
        best, state, history = math.inf, None, []
        while self.steps_done < self.config.steps:
            if self.steps_done >= self.config.warmup and (self.steps_done-self.config.warmup) % self.config.probe_every == 0:
                if self.steps_done == self.config.warmup:
                    self.teacher.model.load_state_dict(self.model.state_dict())
                probe_callback(self)
            batch, y = next(source)
            row = self.step(batch, y, next(target))
            if self.steps_done % self.config.validation_every == 0 or self.steps_done == self.config.steps:
                score = self.validation_score(source_validation_loader)
                row['source_validation'] = score
                if score < best:
                    best = score
                    state = copy.deepcopy(self.teacher.model.state_dict())
            history.append(row)
        return state, history

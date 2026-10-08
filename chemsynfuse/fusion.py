"""Typed table encoding and hierarchical molecular/source fusion."""
import torch
from torch import nn
from torch.nn import functional as F
from .constants import STATES
from .table import SparseFeatureExtractor

def masked_softmax(logits, available, dim):
    """Exact zero for unavailable entries and all-empty rows, finite backward."""
    safe = logits.masked_fill(~available, torch.finfo(logits.dtype).min)
    weights = torch.softmax(safe, dim=dim) * available.to(logits.dtype)
    return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class TypedMSAE(nn.Module):
    def __init__(self, fields, categories, dim=32, heads=4, layers=1, dropout=0.):
        super().__init__()
        # Reuse the actual original modules, including its 2048-wide FFN.
        self.base = SparseFeatureExtractor(
            fields, dim, heads, layers, dim * 2, dim, dropout, token_dropout=0.)
        # Disable nested tensor conversion: explicit masks support holes/all-empty.
        self.base.encoder.enable_nested_tensor = False
        self.type_emb = nn.Embedding(6, dim)
        self.state_emb = nn.Embedding(len(STATES), dim)
        self.category_emb = nn.Embedding(categories, dim, padding_idx=0)
        self.dim = dim

    def forward(self, batch):
        values, states, available = batch['values'], batch['states'], batch['available']
        if values.shape[1] != self.base.index_emb.num_embeddings:
            raise ValueError('Field schema dimension changed')
        ids = torch.arange(values.shape[1], device=values.device)
        # Explicit structural absence uses a state token; it has no magnitude.
        numeric_present = available & (states != STATES.index('not_added')) & (batch['types'] % 2 == 0)
        magnitude = self.base.value_proj(values.unsqueeze(-1)) * numeric_present.unsqueeze(-1)
        tokens = (self.base.index_emb(ids)[None] + magnitude + self.type_emb(batch['types'])[None]
                  + self.state_emb(states) + self.category_emb(batch['categories']))
        tokens = tokens.masked_fill(~available.unsqueeze(-1), 0.)
        safe = available.clone()
        safe[~safe.any(1), 0] = True
        encoded = self.base.encoder(tokens, src_key_padding_mask=~safe)
        outputs, masks, observations = [], [], []
        for section in (0, 1):
            member = batch['sections'] == section
            mask = available & member[None]
            present = mask.any(1)
            pool_mask = mask.clone()
            pool_mask[~present, 0] = True
            pooled = self.base.pool(encoded, ~pool_mask)
            z = self.base.mlp(self.base.norm(pooled))
            outputs.append(z.masked_fill(~present[:, None], 0.))
            masks.append(present)
            obs = F.one_hot(states, len(STATES)).float() * member[None, :, None]
            observations.append(obs.sum(1) / member.sum().clamp_min(1))
        return torch.stack(outputs, 1), torch.stack(masks, 1), torch.cat(observations, -1)


class HierarchicalFusion(nn.Module):

    def __init__(self, schema, molecule_dim=256, dim=32, heads=4, layers=1,
                 dropout=0.):
        super().__init__()
        state = schema.state
        self.table = TypedMSAE(len(state['fields']), len(state['categories']), dim, heads, layers, dropout)
        self.query = nn.Sequential(nn.Linear(2 * dim + 2 * len(STATES), dim), nn.Tanh())
        self.molecule_phi = nn.Sequential(nn.Linear(molecule_dim + schema.metadata_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.molecule_key = nn.Linear(dim, dim, bias=False)
        self.query_key = nn.Linear(dim, dim, bias=False)
        self.alpha_score = nn.Linear(dim, 1, bias=False)
        self.molecule_rho = nn.Sequential(nn.Linear(dim + schema.summary_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.source_projections = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim)) for _ in range(3)])
        self.source_gate = nn.Sequential(nn.Linear(3 * dim + 3, dim), nn.GELU(), nn.Linear(dim, 3 * dim))
        self.norm = nn.LayerNorm(dim)
        self.dim = dim

    def forward(self, batch, *, adversarial=False, stop_gate_grad=True):
        z_table, table_mask, observations = self.table(batch)
        q = self.query(torch.cat([z_table.flatten(1), observations], -1))
        m_mask = batch['molecule_mask']
        v = self.molecule_phi(torch.cat([batch['molecules'], batch['molecule_meta']], -1))
        # PAD is never an extra molecule, regardless of its stored tensor value.
        v = v.masked_fill(~m_mask.unsqueeze(-1), 0.)
        key = self.molecule_key(v)
        key = key + self.query_key(q)[:, None]
        logits = self.alpha_score(torch.tanh(key)).squeeze(-1)
        alpha = masked_softmax(logits, m_mask, 1)
        pool = (alpha.unsqueeze(-1) * v).sum(1)
        z_m = self.molecule_rho(torch.cat([pool, batch['molecule_summary']], -1))
        present = torch.cat([table_mask, m_mask.any(1, keepdim=True)], 1)
        z_m = z_m.masked_fill(~present[:, 2, None], 0.)
        raw = torch.cat([z_table, z_m[:, None]], 1)
        sources = torch.stack([p(raw[:, i]) for i, p in enumerate(self.source_projections)], 1)
        sources = sources.masked_fill(~present[:, :, None], 0.)
        context = torch.cat([sources.flatten(1), present.float()], -1)
        source_logits = self.source_gate(context).reshape(-1, 3, self.dim)
        beta = masked_softmax(source_logits, present[:, :, None], 1)
        contributions = sources * beta
        pre_norm = contributions.sum(1)
        empty = ~present.any(1)
        pre_norm = pre_norm.masked_fill(empty[:, None], 0.)
        fused = self.norm(pre_norm).masked_fill(empty[:, None], 0.)
        result = dict(fused=fused, alpha=alpha, beta=beta, sources=sources,
                    contributions=contributions, pre_norm=pre_norm,
                    availability=present, all_empty=empty, query=q,
                    contribution_additivity=True)
        if adversarial:
            if stop_gate_grad:
                # Reuse table dropout and molecule_phi realizations. Everything
                # recomputed below (rho, projection, LayerNorm) is deterministic.
                adv_pool = (alpha.detach().unsqueeze(-1) * v).sum(1)
                adv_m = self.molecule_rho(torch.cat([adv_pool, batch['molecule_summary']], -1))
                adv_m = adv_m.masked_fill(~present[:, 2, None], 0.)
                adv_m = self.source_projections[2](adv_m).masked_fill(~present[:, 2, None], 0.)
                adv_sources = torch.cat([sources[:, :2], adv_m[:, None]], 1)
                paths = beta.detach() * adv_sources
            else:
                paths = contributions
            adv_sum = paths.sum(1).masked_fill(empty[:, None], 0.)
            result.update(paths=paths, psi=paths.flatten(1), adv_pre_norm=adv_sum,
                          z_adv=self.norm(adv_sum).masked_fill(empty[:, None], 0.))
        return result

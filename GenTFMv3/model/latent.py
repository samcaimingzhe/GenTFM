"""Frozen TabICL row embeddings and schema-conditioned latent flow matching."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch import nn
from .table import Schema, validate_metadata, cat_cardinalities, validate_category_ids

DEFAULT_ENCODER = 'tabicl-classifier-v1-20250208.ckpt'

class FrozenTabICLEncoder(nn.Module):
    def __init__(self, checkpoint, schema: Schema):
        super().__init__()
        from .prior import _ensure_tabicl_importable
        _ensure_tabicl_importable()
        from tabicl._model.tabicl import TabICL
        payload = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, map_location='cpu', weights_only=True)
        config = payload.get('encoder_config', payload.get('config'))
        model = TabICL(**config)
        if 'state_dict' in payload:
            model.load_state_dict(payload['state_dict'], strict=True)
        if getattr(model.col_embedder, 'target_aware', False):
            raise ValueError('Use a non-target-aware pretrained checkpoint (default: TabICL v1). No label bypass is applied.')
        self.col_embedder = model.col_embedder
        self.row_interactor = model.row_interactor
        self.hidden_dim = model.embed_dim * model.row_num_cls
        self.schema = schema
        self.config = config
        self.checkpoint = payload.get('encoder_checkpoint', 'embedded') if isinstance(checkpoint, dict) else str(Path(checkpoint).resolve())
        if 'encoder_state' in payload:
            self.load_state_dict(payload['encoder_state'], strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Keep the pretrained inference path, even inside a trainable parent module.
        return super().train(False)

    def table_input(self, table, meta):
        validate_metadata(meta, self.schema)
        if table.ndim != 2 or table.shape[1] != self.schema.table_dim:
            raise ValueError('Expected [rows, schema.table_dim]')
        n = int(meta.get('tabicl_seq_len', len(table)))
        if not 0 < n <= len(table):
            raise ValueError('Invalid row count')
        table = table[:n]
        nc = int(meta['n_cont'])
        columns = [table[:, :nc]]
        for j, card in enumerate(cat_cardinalities(meta, self.schema.max_cat, self.schema.cat_cardinality)):
            ids = validate_category_ids(table[:, self.schema.category_index(j)], int(card))
            columns.append(ids.float()[:, None])
        x = torch.cat(columns, dim=-1).float()
        if x.shape[1] == 0 or not bool(torch.isfinite(x).all()):
            raise ValueError('Empty/nonfinite table')
        return x

    @torch.no_grad()
    def forward(self, tables, metadata):
        if tables.ndim != 3 or len(metadata) != len(tables):
            raise ValueError('Expected [B, N, F] and one metadata record per table')
        device = next(self.parameters()).device
        rows = []
        for table, meta in zip(tables, metadata):
            x = self.table_input(table, meta).to(device)[None]
            # v1 uses y_train length to choose the column-attention context;
            # its non-target-aware encoder never consumes these placeholder values.
            placeholder = torch.zeros(1, x.shape[1], device=device)
            z = self.row_interactor(self.col_embedder(x, y_train=placeholder, embed_with_test=True))[0]
            if not bool(torch.isfinite(z).all()):
                raise ValueError('Nonfinite pretrained embeddings')
            rows.append(z)
        length = max(len(z) for z in rows)
        out = rows[0].new_zeros(len(rows), length, self.hidden_dim)
        mask = torch.zeros(len(rows), length, dtype=torch.bool, device=device)
        for i, z in enumerate(rows):
            out[i, :len(z)] = z
            mask[i, :len(z)] = True
        return out, mask


def schema_condition(metadata, schema, device):
    """Ordered continuous slots then categorical slots; zero means inactive."""
    result = torch.zeros(len(metadata), schema.max_cont + schema.max_cat, device=device)
    for i, meta in enumerate(metadata):
        validate_metadata(meta, schema)
        result[i, :int(meta['n_cont'])] = 1
        cards = cat_cardinalities(meta, schema.max_cat, schema.cat_cardinality)
        result[i, schema.max_cont:schema.max_cont + len(cards)] = torch.as_tensor(cards, device=device) / schema.cat_cardinality
    return result


class LatentNormalizer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.ones(dim))

    @torch.no_grad()
    def fit(self, batches):
        count, total, squares = 0, None, None
        for z, mask in batches:
            values = z[mask].double()
            if not len(values):
                continue
            count += len(values)
            sums, sq = values.sum(0), values.square().sum(0)
            total = sums if total is None else total + sums
            squares = sq if squares is None else squares + sq
        if count < 2:
            raise ValueError('At least two calibration rows are required')
        mean = total / count
        std = (squares / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
        self.mean.copy_(mean)
        self.std.copy_(std)

    def forward(self, z):
        return (z - self.mean) / self.std

    def inverse(self, z):
        return z * self.std + self.mean


class LatentFlow(nn.Module):
    def __init__(self, latent_dim, condition_dim, width=256, heads=8, layers=4):
        super().__init__()
        if width % heads or width % 2:
            raise ValueError('width must be even and divisible by heads')
        self.config = dict(latent_dim=latent_dim, condition_dim=condition_dim, width=width, heads=heads, layers=layers)
        self.input = nn.Linear(latent_dim, width)
        self.condition = nn.Linear(condition_dim, width)
        self.time = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.register_buffer('frequencies', torch.exp(torch.linspace(0, math.log(1000), width // 2)))
        block = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0, activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.output = nn.Linear(width, latent_dim)

    def forward(self, z, t, condition, row_mask):
        if not bool(row_mask.any(1).all()):
            raise ValueError('Each table must have at least one active row')
        phase = t[:, None] * self.frequencies[None]
        time = self.time(torch.cat([phase.sin(), phase.cos()], -1))
        h = self.input(z) + (time + self.condition(condition))[:, None]
        h = self.transformer(h, src_key_padding_mask=~row_mask)
        return self.output(h).masked_fill(~row_mask[..., None], 0)

    def compute_loss(self, z1, condition, row_mask, *, z0=None, t=None):
        z1 = z1.detach()
        z0 = torch.randn_like(z1) if z0 is None else z0
        t = torch.rand(len(z1), device=z1.device) if t is None else t
        zt = torch.lerp(z0, z1, t[:, None, None]).masked_fill(~row_mask[..., None], 0)
        error = (self(zt, t, condition, row_mask) - (z1 - z0)).square()
        denominator = row_mask.sum(1) * z1.shape[-1]
        return ((error * row_mask[..., None]).sum((1, 2)) / denominator).mean()

    @torch.no_grad()
    def sample(self, condition, row_mask, steps=50):
        if steps < 1:
            raise ValueError('steps must be positive')
        was_training = self.training
        self.eval()
        try:
            z = torch.randn(*row_mask.shape, self.config['latent_dim'], device=condition.device)
            z.masked_fill_(~row_mask[..., None], 0)
            # Heun integration from noise (t=0) to data (t=1).
            for k in range(steps):
                t = condition.new_full((len(z),), k / steps)
                v = self(z, t, condition, row_mask)
                proposal = z + v / steps
                v_next = self(proposal, t + 1 / steps, condition, row_mask)
                z = z + (v + v_next) / (2 * steps)
            return z.masked_fill(~row_mask[..., None], 0)
        finally:
            self.train(was_training)

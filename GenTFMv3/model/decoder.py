"""Mixed-table reconstruction from standardized TabICL row embeddings."""
import torch
from torch import nn
import torch.nn.functional as F
from .table import validate_metadata, cat_cardinalities, validate_category_ids

class TableDecoder(nn.Module):
    def __init__(self, latent_dim, max_cont=32, max_cat=8, max_cardinality=12, width=512):
        super().__init__()
        self.config = dict(latent_dim=latent_dim, max_cont=max_cont, max_cat=max_cat,
                           max_cardinality=max_cardinality, width=width)
        self.body = nn.Sequential(nn.Linear(latent_dim + max_cont + max_cat, width), nn.GELU(),
                                  nn.Linear(width, width), nn.GELU())
        self.cont_head = nn.Linear(width, max_cont)
        self.cat_head = nn.Linear(width, max_cat * max_cardinality)

    def forward(self, z, condition):
        c = condition[:, None].expand(-1, z.shape[1], -1)
        h = self.body(torch.cat([z, c], -1))
        return self.cont_head(h), self.cat_head(h).reshape(*z.shape[:2], self.config['max_cat'], self.config['max_cardinality'])

    def compute_loss(self, z, condition, tables, metadata, schema, row_mask, cat_weight=1.):
        if cat_weight < 0 or not bool(row_mask.any(1).all()):
            raise ValueError('Nonnegative category weight and nonempty tables required')
        cont, logits = self(z.detach(), condition)
        num_losses, cat_losses, accuracies = [], [], []
        for b, meta in enumerate(metadata):
            validate_metadata(meta, schema)
            valid = row_mask[b]
            x = tables[b, :z.shape[1]][valid]
            nc = int(meta['n_cont'])
            observed = torch.ones_like(x[:, :nc], dtype=torch.bool)
            num = (cont[b, valid, :nc] - x[:, :nc]).square()
            num_losses.append(num.masked_fill(~observed, 0).sum() / observed.sum().clamp_min(1))
            ce, acc = [], []
            for j, card in enumerate(cat_cardinalities(meta, schema.max_cat, schema.cat_cardinality)):
                target = validate_category_ids(x[:, schema.category_index(j)], int(card))
                scores = logits[b, valid, j, :int(card)]
                ce.append(F.cross_entropy(scores, target))
                acc.append((scores.argmax(-1) == target).float().mean())
            cat_losses.append(torch.stack(ce).mean() if ce else logits[b].sum() * 0)
            if acc:
                accuracies.append(torch.stack(acc).mean())
        num, cat = torch.stack(num_losses).mean(), torch.stack(cat_losses).mean()
        loss = num + cat_weight * cat
        return loss, dict(continuous_mse=num.detach(), categorical_ce=cat.detach(),
                          categorical_accuracy=torch.stack(accuracies).mean() if accuracies else z.new_tensor(0.))

    @torch.no_grad()
    def reconstruct(self, z, condition, metadata, schema, row_mask, sample_categories=False):
        cont, logits = self(z, condition)
        result = z.new_zeros(*z.shape[:2], schema.table_dim)
        for b, meta in enumerate(metadata):
            validate_metadata(meta, schema)
            valid = row_mask[b]
            nc = int(meta['n_cont'])
            result[b, :, :nc] = cont[b, :, :nc]
            for j, card in enumerate(cat_cardinalities(meta, schema.max_cat, schema.cat_cardinality)):
                scores = logits[b, :, j, :int(card)]
                ids = torch.multinomial(scores.softmax(-1), 1).squeeze(-1) if sample_categories else scores.argmax(-1)
                result[b, :, schema.category_index(j)] = ids.float()
            result[b, ~valid] = 0
        return result

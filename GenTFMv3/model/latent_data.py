"""Draw whole tables independently and pad rows for latent training."""
import torch

def draw_batch(prior, encoder, batch_size, rows):
    items, metas = [], []
    for _ in range(batch_size):
        x, meta = prior.sample_batch(1, rows, return_metadata=True)
        items.append(x[0])
        metas.append(meta[0])
    n = max(map(len, items))
    tables = items[0].new_zeros(batch_size, n, prior.schema.table_dim)
    for i, x in enumerate(items):
        tables[i, :len(x)] = x
    z, mask = encoder(tables, metas)
    return tables, z, mask, metas

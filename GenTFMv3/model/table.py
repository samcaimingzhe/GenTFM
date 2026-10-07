"""Raw numeric/category-ID table contract; no binary or onehot codecs."""
from dataclasses import dataclass
import numpy as np
import torch

@dataclass(frozen=True)
class Schema:
    max_cont: int = 32
    max_cat: int = 8
    cat_cardinality: int = 12
    schema_version: str = 'raw_ids_v1'

    def __post_init__(self):
        if self.schema_version != 'raw_ids_v1':
            raise ValueError('Expected raw_ids_v1; legacy codec checkpoints must be retrained')
        if self.max_cont < 0 or self.max_cat < 0 or self.cat_cardinality < 2:
            raise ValueError('Invalid schema limits')

    @property
    def table_dim(self):
        return self.max_cont + self.max_cat

    def category_index(self, j):
        if not 0 <= j < self.max_cat:
            raise ValueError('Category outside schema')
        return self.max_cont + j

    def metadata_fields(self):
        return dict(schema_version=self.schema_version, no_missingness=True)


def cat_cardinalities(metadata, max_cat, cat_cardinality):
    n = int(metadata['n_cat'])
    cards = np.asarray(metadata['cat_cardinalities'])
    if not 0 <= n <= max_cat or cards.shape != (n,) or not np.isfinite(cards).all():
        raise ValueError('Invalid cardinalities')
    if np.any(cards != cards.astype(np.int64)) or np.any(cards < 1) or np.any(cards > cat_cardinality):
        raise ValueError('Invalid category cardinality')
    return cards.astype(np.int64)


def validate_metadata(metadata, schema):
    if metadata.get('schema_version') != schema.schema_version:
        raise ValueError('Expected raw_ids_v1 metadata')
    if not 0 <= int(metadata['n_cont']) <= schema.max_cont:
        raise ValueError('Invalid continuous column count')
    cat_cardinalities(metadata, schema.max_cat, schema.cat_cardinality)
    if not metadata.get('no_missingness', False):
        raise ValueError('Current raw adapter requires observed tables')


def validate_category_ids(ids, cardinality):
    if not bool(torch.isfinite(ids).all()) or bool(((ids != ids.long()) | (ids < 0) | (ids >= cardinality)).any()):
        raise ValueError('Category IDs must be integers within field cardinality')
    return ids.long()

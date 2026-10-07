"""GenTFMv3: pretrained TabICL embeddings and latent flow matching."""
from .table import Schema
from .prior import PriorConfig, TabICLPriorEngine
from .latent import FrozenTabICLEncoder, LatentFlow, LatentNormalizer, schema_condition
__version__ = '3.0.0'
__all__ = ['Schema', 'PriorConfig', 'TabICLPriorEngine', 'FrozenTabICLEncoder',
           'LatentFlow', 'LatentNormalizer', 'schema_condition']

from .decoder import TableDecoder
__all__.append("TableDecoder")

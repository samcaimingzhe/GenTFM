"""Fixed-width onehot/binary codecs shared by all GenTFM v1.2 components."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math
import numpy as np
import torch
import torch.nn.functional as F

@dataclass(frozen=True)
class Schema:
    max_cont: int = 32
    max_cat: int = 8
    cat_cardinality: int = 12
    cat_encoding: str = 'onehot'
    binary_bit_order: str = 'msb_first'
    schema_version: str | None = None

    def __post_init__(self):
        if self.max_cont < 0 or self.max_cat < 0 or self.cat_cardinality < 2:
            raise ValueError('Invalid schema limits')
        if self.cat_encoding not in {'onehot', 'binary'} or self.binary_bit_order != 'msb_first':
            raise ValueError('Supported codecs: onehot/binary, msb_first only')
        version = f'mixed_{self.cat_encoding}_v1'
        if self.schema_version is not None and self.schema_version != version:
            raise ValueError('schema_version conflicts with cat_encoding')
        object.__setattr__(self, 'schema_version', version)

    @property
    def cat_width(self):
        return self.cat_cardinality if self.cat_encoding == 'onehot' else math.ceil(math.log2(self.cat_cardinality))
    @property
    def encoded_dim(self):
        return 2 * self.max_cont + self.max_cat * self.cat_width
    @property
    def cat_start(self):
        return self.max_cont
    @property
    def mask_start(self):
        return self.max_cont + self.max_cat * self.cat_width
    def as_tuple(self):
        """Legacy size tuple; callers must ALSO pass codec_kwargs()."""
        return self.max_cont, self.max_cat, self.cat_cardinality
    def codec_kwargs(self):
        return dict(cat_encoding=self.cat_encoding, binary_bit_order=self.binary_bit_order)
    def metadata_fields(self):
        return dict(**self.codec_kwargs(), schema_version=self.schema_version)
    def category_slice(self, j):
        if not 0 <= j < self.max_cat:
            raise ValueError(f'Category slot {j} outside schema')
        start = self.cat_start + j * self.cat_width
        return slice(start, start + self.cat_width)


def encoded_dim(max_cont, max_cat, cat_cardinality, cat_encoding='onehot', binary_bit_order='msb_first'):
    return Schema(max_cont,max_cat,cat_cardinality,cat_encoding,binary_bit_order).encoded_dim

def slices(max_cont, max_cat, cat_cardinality, cat_encoding='onehot', binary_bit_order='msb_first'):
    s=Schema(max_cont,max_cat,cat_cardinality,cat_encoding,binary_bit_order)
    return dict(cat_start=s.cat_start, mask_start=s.mask_start, cat_width=s.cat_width)

def validate_metadata(metadata, schema):
    for key, expected in schema.metadata_fields().items():
        # Metadata without codec fields is legacy onehot data.
        actual=metadata.get(key, 'onehot' if key=='cat_encoding' else expected)
        if actual != expected:
            raise ValueError(f'{key}: expected {expected}, got {actual}')
    if schema.cat_encoding == 'binary' and metadata.get('cat_cardinalities') is None:
        raise ValueError('Binary metadata requires explicit cat_cardinalities')
    nc,nk=int(metadata['n_cont']),int(metadata['n_cat'])
    if not (0<=nc<=schema.max_cont and 0<=nk<=schema.max_cat):
        raise ValueError('Active column counts exceed schema')
    cat_cardinalities(metadata,schema.max_cat,schema.cat_cardinality)

def cat_cardinalities(metadata, max_cat, cat_cardinality):
    n=int(metadata['n_cat'])
    if not 0<=n<=max_cat: raise ValueError('Invalid n_cat')
    raw=metadata.get('cat_cardinalities')
    if raw is not None:
        values=np.asarray(raw)
        if not np.all(np.isfinite(values)) or np.any(values != values.astype(np.int64)):
            raise ValueError('Cardinalities must be finite integers')
    cards=np.full(n,cat_cardinality,dtype=np.int64) if raw is None else np.asarray(raw,dtype=np.int64)
    if cards.shape!=(n,) or np.any(cards<1) or np.any(cards>cat_cardinality):
        raise ValueError('cat_cardinalities must give one valid cardinality per active field')
    return cards

def label_info(metadata):
    aliases={'cat':'categorical','categorical':'categorical','classification':'categorical',
             'cont':'continuous','continuous':'continuous','regression':'continuous','numerical':'continuous','numeric':'continuous'}
    kinds=[]
    for key in ('label_type','target_task','task_type','task'):
        value=metadata.get(key)
        if value is not None:
            if str(value).lower() not in aliases: raise ValueError(f'Unknown {key}: {value}')
            kinds.append(aliases[str(value).lower()])
    if len(set(kinds))>1: raise ValueError('Contradictory label/task metadata')
    ci,ni=metadata.get('label_cat_index'),metadata.get('label_cont_index')
    if ci is not None and ni is not None: raise ValueError('Both label indices are set')
    kind=kinds[0] if kinds else ('categorical' if ci is not None else 'continuous' if ni is not None else None)
    if kind is None: raise ValueError('Explicit label/task metadata is required')
    if (kind=='categorical' and ni is not None) or (kind=='continuous' and ci is not None):
        raise ValueError('Label index conflicts with task')
    index=int(ci if ci is not None else int(metadata['n_cat'])-1) if kind=='categorical' else int(ni if ni is not None else 0)
    limit=int(metadata['n_cat'] if kind=='categorical' else metadata['n_cont'])
    if not 0<=index<limit: raise ValueError('Label index outside active columns')
    return kind,index

def category_ids_to_bits(ids, width):
    if isinstance(ids,torch.Tensor):
        if not bool(torch.isfinite(ids).all()) or bool((ids != ids.long()).any()): raise ValueError('Category IDs must be integers')
        if bool(((ids<0)|(ids>=2**width)).any()): raise ValueError('ID outside bit width')
        shifts=torch.arange(width-1,-1,-1,device=ids.device)
        return ((ids.long().unsqueeze(-1)>>shifts)&1).float()
    ids=np.asarray(ids)
    if not np.all(np.isfinite(ids)) or np.any(ids!=ids.astype(np.int64)) or np.any((ids<0)|(ids>=2**width)):
        raise ValueError('Category IDs must be integers within bit width')
    return ((ids.astype(np.int64)[...,None]>>np.arange(width-1,-1,-1))&1).astype(np.float32)

def category_bits_to_ids(bits, width=None):
    width=bits.shape[-1] if width is None else width
    if bits.shape[-1]!=width: raise ValueError('Wrong bit width')
    if isinstance(bits,torch.Tensor):
        if bool(((bits!=0)&(bits!=1)).any()): raise ValueError('Expected exact 0/1 bits')
        weights=2**torch.arange(width-1,-1,-1,device=bits.device)
        return (bits.long()*weights).sum(-1)
    bits=np.asarray(bits)
    if np.any((bits!=0)&(bits!=1)): raise ValueError('Expected exact 0/1 bits')
    return (bits.astype(np.int64)*2**np.arange(width-1,-1,-1)).sum(-1)

def encode_category(ids, schema):
    if schema.cat_encoding=='binary': return category_ids_to_bits(ids,schema.cat_width)
    if isinstance(ids,torch.Tensor):
        if not bool(torch.isfinite(ids).all()) or bool(((ids != ids.long()) | (ids<0) | (ids>=schema.cat_cardinality)).any()):
            raise ValueError('Invalid categorical ID')
        return F.one_hot(ids.long(),schema.cat_cardinality).float()
    ids=np.asarray(ids)
    if not np.all(np.isfinite(ids)) or np.any(ids != ids.astype(np.int64)) or np.any((ids<0)|(ids>=schema.cat_cardinality)):
        raise ValueError('Invalid categorical ID')
    return np.eye(schema.cat_cardinality,dtype=np.float32)[np.asarray(ids,dtype=np.int64)]

def decode_category(block, schema, cardinality):
    ids=category_bits_to_ids(block) if schema.cat_encoding=='binary' else block.argmax(-1)
    invalid=(ids<0)|(ids>=int(cardinality))
    if bool(invalid.any()): raise ValueError('Invalid categorical code for field cardinality')
    return ids

def mixed_feature_mask(metadata, max_cont, max_cat, cat_cardinality, cat_encoding='onehot', binary_bit_order='msb_first'):
    s=Schema(max_cont,max_cat,cat_cardinality,cat_encoding,binary_bit_order)
    validate_metadata(metadata,s)
    mask=np.zeros(s.encoded_dim,dtype=bool)
    nc,nk=int(metadata['n_cont']),int(metadata['n_cat'])
    mask[:nc]=True
    mask[s.mask_start:s.mask_start+nc]=True
    for j,c in enumerate(cat_cardinalities(metadata,max_cat,cat_cardinality)):
        block=s.category_slice(j)
        if cat_encoding=='onehot': mask[block.start:block.start+int(c)]=True
        else: mask[block.stop-max(1,math.ceil(math.log2(int(c)))):block.stop]=True
    return mask

def batch_feature_mask(metadata, max_cont, max_cat, cat_cardinality, device, cat_encoding='onehot', binary_bit_order='msb_first'):
    return torch.as_tensor(np.stack([mixed_feature_mask(m,max_cont,max_cat,cat_cardinality,cat_encoding,binary_bit_order) for m in metadata]),dtype=torch.bool,device=device)

def sanitize_mixed_encoded(encoded, metadata, max_cont, max_cat, cat_cardinality, cat_encoding='onehot', binary_bit_order='msb_first'):
    s=Schema(max_cont,max_cat,cat_cardinality,cat_encoding,binary_bit_order)
    validate_metadata(metadata,s)
    x=np.asarray(encoded,dtype=np.float32)
    if x.ndim!=2 or x.shape[1]!=s.encoded_dim or not np.isfinite(x).all(): raise ValueError('Invalid encoded matrix shape or nonfinite values')
    nc=int(metadata['n_cont'])
    out=np.zeros_like(x)
    obs=np.ones((len(x),nc),dtype=np.float32) if metadata.get('force_observed_mask') or metadata.get('no_missingness') else (x[:,s.mask_start:s.mask_start+nc]>=.5).astype(np.float32)
    out[:,:nc]=x[:,:nc]*obs
    out[:,s.mask_start:s.mask_start+nc]=obs
    for j,c in enumerate(cat_cardinalities(metadata,max_cat,cat_cardinality)):
        block=s.category_slice(j)
        if cat_encoding=='onehot': ids=x[:,block.start:block.start+int(c)].argmax(-1)
        else:
            codes=category_ids_to_bits(np.arange(c),s.cat_width)
            # Nearest legal code; argmin deterministically resolves ties by lowest ID.
            ids=((x[:,block][:,None,:]-codes[None,:,:])**2).sum(-1).argmin(-1)
        out[:,block]=encode_category(ids,s)
    return out

def decode_components(encoded, metadata, max_cont, max_cat, cat_cardinality, cat_encoding='onehot', binary_bit_order='msb_first'):
    s=Schema(max_cont,max_cat,cat_cardinality,cat_encoding,binary_bit_order)
    x=sanitize_mixed_encoded(encoded,metadata,*s.as_tuple(),**s.codec_kwargs())
    nc,nk=int(metadata['n_cont']),int(metadata['n_cat'])
    cats=np.zeros((len(x),nk),dtype=np.int64)
    for j,c in enumerate(cat_cardinalities(metadata,max_cat,cat_cardinality)):
        cats[:,j]=decode_category(x[:,s.category_slice(j)],s,c)
    return dict(encoded=x,cont=x[:,:nc],obs=x[:,s.mask_start:s.mask_start+nc],cats=cats)

def encode_components(cont, cats, metadata, max_cont, max_cat, cat_cardinality, cat_encoding='onehot', binary_bit_order='msb_first', obs=None):
    s=Schema(max_cont,max_cat,cat_cardinality,cat_encoding,binary_bit_order)
    validate_metadata(metadata,s)
    cont=np.asarray(cont,dtype=np.float32)
    cats=np.asarray(cats)
    nc,nk=int(metadata['n_cont']),int(metadata['n_cat'])
    if cont.ndim!=2 or cont.shape[1]!=nc or cats.shape!=(len(cont),nk): raise ValueError('Component shape mismatch')
    out=np.zeros((len(cont),s.encoded_dim),dtype=np.float32)
    out[:,:nc]=cont
    out[:,s.mask_start:s.mask_start+nc]=1 if obs is None else np.asarray(obs)
    for j,c in enumerate(cat_cardinalities(metadata,max_cat,cat_cardinality)):
        ids=cats[:,j]
        if not np.all(np.isfinite(ids)) or np.any(ids!=ids.astype(np.int64)) or np.any((ids<0)|(ids>=c)): raise ValueError('Invalid category ID')
        out[:,s.category_slice(j)]=encode_category(ids,s)
    return sanitize_mixed_encoded(out,metadata,*s.as_tuple(),**s.codec_kwargs())

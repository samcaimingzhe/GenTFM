import torch
from torch import Tensor, nn

if __package__:
    from .ColEmb import MultiheadAttentionBlock
else:
    from ColEmb import MultiheadAttentionBlock


class RowInteraction(nn.Module):
    """Mix valid columns within each row while preserving cell representations."""
    def __init__(
        self, embed_dim, num_blocks, nhead,
        dim_feedforward, max_features,
        dropout=0.0, norm_first=True,
    ):
        super().__init__()
        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")
        if max_features < 1:
            raise ValueError("max_features must be positive")
        if nhead <= 0 or embed_dim <= 0 or embed_dim % nhead != 0:
            raise ValueError("embed_dim must be positive and divisible by nhead")

        self.embed_dim = embed_dim
        self.feature_embedding = nn.Embedding(max_features, embed_dim)
        nn.init.trunc_normal_(self.feature_embedding.weight, std=0.02)

        self.blocks = nn.ModuleList([
            MultiheadAttentionBlock(
                d_model=embed_dim, nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout, norm_first=norm_first,
            )
            for _ in range(num_blocks)
        ])
        self.out_ln = (
            nn.LayerNorm(embed_dim) if norm_first else nn.Identity()
        )

    def forward(self, embeddings: Tensor, feature_mask: Tensor) -> Tensor:
        if embeddings.ndim != 4:
            raise ValueError("embeddings must have shape (B, K, D, E)")
        B, K, D, E = embeddings.shape
        if K == 0:
            raise ValueError("embeddings must contain at least one row")
        if E != self.embed_dim:
            raise ValueError("embedding dimension must match embed_dim")
        if D > self.feature_embedding.num_embeddings:
            raise ValueError("D must not exceed max_features")
        if feature_mask.shape != (B, D):
            raise ValueError("feature_mask must have shape (B, D)")
        if feature_mask.dtype != torch.bool:
            raise TypeError("feature_mask must be boolean")
        if feature_mask.device != embeddings.device:
            raise ValueError("embeddings and feature_mask must be on the same device")

        positions = torch.arange(D, device=embeddings.device)
        features = embeddings + self.feature_embedding(positions).to(embeddings.dtype)
        features = features.masked_fill(~feature_mask[:, None, :, None], 0.0)

        x = features.reshape(B * K, D, E)

        key_mask = (~feature_mask)[:, None, :].expand(-1, K, -1)
        key_mask = key_mask.reshape(B * K, D)

        # Skip rows from tables with no valid keys to avoid all-masked attention.
        valid_rows = (~key_mask).any(dim=1).nonzero(as_tuple=True)[0]
        if valid_rows.numel() == 0:
            return x.reshape(B, K, D, E)

        active = x.index_select(0, valid_rows)
        active_mask = key_mask.index_select(0, valid_rows)
        for block in self.blocks:
            active = active.masked_fill(active_mask[:, :, None], 0.0)
            active = block(active, active, active, key_padding_mask=active_mask)

        active = self.out_ln(active).masked_fill(active_mask[:, :, None], 0.0)
        x = x.index_copy(0, valid_rows, active)
        return x.reshape(B, K, D, E)

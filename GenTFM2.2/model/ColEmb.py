import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class MultiheadAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        norm_first: bool = True,
    ):
        super().__init__()

        self.norm_first = norm_first

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)

        self.dropout_attn = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        if self.norm_first:
            q = self.norm_q(query)
            k = self.norm_kv(key)
            v = self.norm_kv(value)
        else:
            q, k, v = query, key, value

        attn_out, _ = self.attn(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        x = query + self.dropout_attn(attn_out)
        if self.norm_first:
            x = x + self.ffn(self.norm_ffn(x))
        else:
            x = self.norm_q(x)
            x = self.norm_ffn(x + self.ffn(x))

        return x



class InducedSelfAttentionBlock(nn.Module):
    """Process embedded columns of shape (N_columns, K, d_model)."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        norm_first: bool = True,
    ):
        super().__init__()

        self.multihead_attn1 = MultiheadAttentionBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=norm_first,
        )
        self.multihead_attn2 = MultiheadAttentionBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            norm_first=norm_first,
        )

        self.ind_vectors = nn.Parameter(
            torch.empty(num_inds, d_model)
        )
        nn.init.trunc_normal_(self.ind_vectors, std=0.02)

    def forward(self, col: Tensor) -> Tensor:
        ind_vectors = self.ind_vectors.unsqueeze(0).expand(
            col.shape[0], -1, -1
        )

        hidden = self.multihead_attn1(ind_vectors, col, col)
        return self.multihead_attn2(col, hidden, hidden)


class SetTransformer(nn.Module):
    """Encode each valid column independently across its rows."""

    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int = 16,
        dropout: float = 0.0,
        norm_first: bool = True,
        recompute: bool = False,
    ):
        super().__init__()

        if nhead <= 0 or d_model <= 0 or d_model % nhead != 0:
            raise ValueError(
                "d_model and nhead must be positive, "
                "and d_model must be divisible by nhead"
            )
        if num_blocks < 1 or num_inds < 1:
            raise ValueError("num_blocks and num_inds must be positive")

        self.d_model = d_model
        self.recompute = recompute

        self.cell_embedding = nn.Linear(1, d_model)

        self.blocks = nn.ModuleList([
            InducedSelfAttentionBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_inds=num_inds,
                dropout=dropout,
                norm_first=norm_first,
            )
            for _ in range(num_blocks)
        ])

    def forward(
        self,
        data: Tensor,
        feature_mask: Tensor,
    ) -> Tensor:
        # data: (B, K, D); feature_mask: (B, D)
        if data.ndim != 3:
            raise ValueError("data must have shape (B, K, D)")

        B, K, D = data.shape

        if K == 0:
            raise ValueError("data must contain at least one row")
        if feature_mask.shape != (B, D):
            raise ValueError("feature_mask must have shape (B, D)")
        if feature_mask.dtype != torch.bool:
            raise TypeError("feature_mask must be boolean")
        if feature_mask.device != data.device:
            raise ValueError(
                "data and feature_mask must be on the same device"
            )

        # Select valid columns once, before embedding and all blocks.
        col = data.transpose(1, 2)[feature_mask]  # (N_valid, K)

        if col.shape[0] == 0:
            return data.new_zeros(B, D, K, self.d_model)

        col = self.cell_embedding(col.unsqueeze(-1))
        # col: (N_valid, K, d_model)

        for block in self.blocks:
            if self.recompute and self.training:
                col = checkpoint(block, col, use_reentrant=False)
            else:
                col = block(col)

        out = col.new_zeros(B, D, K, self.d_model)
        out[feature_mask] = col
        return out

class ColEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_blocks: int,
        nhead: int,
        dim_feedforward: int,
        num_inds: int,
        dropout: float = 0.0,
        norm_first: bool = True,
        affine: bool = False,
        recompute: bool = False,
    ):
        super().__init__()

        self.affine = affine

        # Use our SetTransformer: scalar embedding + valid-column attention.
        self.tf_col = SetTransformer(
            num_blocks=num_blocks,
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_inds=num_inds,
            dropout=dropout,
            norm_first=norm_first,
            recompute=recompute,
        )

        if affine:
            self.out_w = nn.Linear(embed_dim, embed_dim)
            self.out_b = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        data: Tensor,
        feature_mask: Tensor,
    ) -> Tensor:
        # data: (B, K, D); feature_mask: (B, D)
        col = self.tf_col(data, feature_mask)
        # col: (B, D, K, embed_dim)

        if self.affine:
            weights = self.out_w(col)
            biases = self.out_b(col)

            values = data.transpose(1, 2).masked_fill(
                ~feature_mask[:, :, None], 0.0
            ).unsqueeze(-1)
            col = values * weights + biases

        # Linear biases can make padded columns nonzero.
        col = col.masked_fill(
            ~feature_mask[:, :, None, None], 0.0
        )

        return col.transpose(1, 2).contiguous()
        # (B, K, D, embed_dim)

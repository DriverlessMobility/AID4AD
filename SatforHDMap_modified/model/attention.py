import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Dropout, Softmax, Linear, LayerNorm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Fusion_Atten(nn.Module):
    def __init__(self,
                 bev_channels, prior_channels, hidden_c,
                 img_size=(200, 400), patch_size=(10, 10),
                 decoder_layers=3, dropout=0.1,
                 num_heads=8, mlp_ratio=4, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm
                 ):
        super(Fusion_Atten, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size[0]
        self.grid_size = (int(img_size[0] / patch_size[0]), int(img_size[1] / patch_size[1]))
        self.n_patches = self.grid_size[0] * self.grid_size[1]
        self.bev_channels = bev_channels
        self.prior_channels = prior_channels
        self.hidden_c = hidden_c
        self.drop_out = dropout

        self.patch_embedding = PatchEmbed(bev_in_channels=self.bev_channels, prior_in_channels=self.prior_channels, out_channels=self.hidden_c)
        self.decoder_layers = decoder_layers
        self.decoder = nn.ModuleList([
            TransformerDecoderLayer(dim=hidden_c, input_resolution=self.grid_size,
                                 num_heads=num_heads, 
                                 mlp_ratio=mlp_ratio,
                                 drop=drop, 
                                 norm_layer=norm_layer)
            for i in range(self.decoder_layers)])
        self.expand = nn.Linear(self.hidden_c, (self.patch_size**2)*self.bev_channels, bias=False)
        self.drop = Dropout(self.drop_out)

    def forward(self, bev_features, prior_features):
        bev_embedding, prior_embedding = self.patch_embedding(bev_features, prior_features)
        query_feat = bev_embedding
        for i in range(self.decoder_layers):
            query_feat = self.decoder[i](query_feat, prior_embedding)

        bsz, n_patch, hidden = query_feat.size()  # (B, H/patch_size, W/patch_size, c_hidden)
        query_feat = self.expand(query_feat)  # (B, H/patch_size * W/patch_size， patch_size**2*c_bev)
        query_feat = self.drop(query_feat)

        x = query_feat.permute(0, 2, 1)
        x = x.contiguous().view(bsz, self.bev_channels, self.img_size[0],
                                self.img_size[1])  # (B, H, W, c_bev)
        
        return torch.cat([x, prior_features], dim=1)


class PatchEmbed(nn.Module):
    def __init__(self, bev_in_channels, prior_in_channels, out_channels, img_size=(200,400), patch_size=(10, 10), dropout_rate=0.1):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size[0]

        self.grid_size = (int(img_size[0] / patch_size[0]), int(img_size[1] / patch_size[1]))
        self.n_patches = self.grid_size[0] * self.grid_size[1]

        self.bev_patch_embedding = Conv2d(in_channels=bev_in_channels,
                                          out_channels=out_channels,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)
        self.prior_patch_embedding = Conv2d(in_channels=prior_in_channels,
                                            out_channels=out_channels,
                                            kernel_size=self.patch_size,
                                            stride=self.patch_size)

        # To do: 仿照Transfusion，使用PositionEmbeddingLearned，替代可学习参数
        self.position_embedding_bev = nn.Parameter(torch.zeros([1, self.n_patches, out_channels]))
        self.position_embedding_prior = nn.Parameter(torch.zeros([1, self.n_patches, out_channels]))

        self.dropout = Dropout(dropout_rate)

        # x   [B, C, H, W]
    def forward(self, bev_feature, prior_feature):
        bev_feature = self.bev_patch_embedding(bev_feature)  # [B, C_hidden, gird_size[0], grid_size[1]]
        bev_feature = bev_feature.flatten(2).transpose(-1, -2)  # [B, n_patches, hidden]

        prior_feature = self.prior_patch_embedding(prior_feature)
        prior_feature = prior_feature.flatten(2).transpose(-1, -2)

        bev_embedding = bev_feature + self.position_embedding_bev
        bev_embedding = self.dropout(bev_embedding)

        prior_embedding = prior_feature + self.position_embedding_prior
        prior_embedding = self.dropout(prior_embedding)

        return bev_embedding, prior_embedding


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerDecoderLayer(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, y, attn_mask=None):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)  # B,H*W,C

        y = self.norm1(y)

        x = self.cross_attn(x, y, y, attn_mask=attn_mask)[0]

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

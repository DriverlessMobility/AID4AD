import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
import warnings
from torch.nn import Conv2d, Dropout, Softmax, Linear, LayerNorm
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from timm.models.layers import DropPath

class FusionDeTrans(nn.Module):
    def __init__(self,
                 bev_channels, prior_channels, hidden_c,
                 img_size=(200, 400), patch_size=(10, 10),
                 decoder_layers=2, dropout=0.1,
                 ):
        super(FusionDeTrans, self).__init__()
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
            DeformableTransformerDecoderLayer(dim=self.hidden_c)
            for i in range(self.decoder_layers)])
        self.expand = nn.Linear(self.hidden_c, (self.patch_size**2)*self.bev_channels, bias=False)
        self.drop = Dropout(self.drop_out)

    def forward(self, bev_features, prior_features):
        b, _, _, _ = bev_features.shape
        bev_embedding, prior_embedding = self.patch_embedding(bev_features, prior_features)
        query_feat = bev_embedding
        reference_points = torch.zeros((self.grid_size[0], self.grid_size[1], 2))
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                reference_points[x][y][0] = x
                reference_points[x][y][1] = y
        reference_points = torch.unsqueeze(reference_points, 0)
        reference_points = reference_points.repeat(b, 1, 1, 1)
        reference_points = reference_points.view(b, -1, 2)
        reference_points = torch.unsqueeze(reference_points, 2)
        reference_points = reference_points.to(bev_features)  # (b, len, 1, 2)

        spatial_shapes = []
        for i in range(1):
            spatial_shape = (self.grid_size[0], self.grid_size[1])
            spatial_shapes.append(spatial_shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=query_feat.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        for i in range(self.decoder_layers):
            query_feat = self.decoder[i](query_feat, reference_points, prior_embedding, spatial_shapes, level_start_index)

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

        self.norm_bev = LayerNorm(out_channels)
        self.norm_prior = LayerNorm(out_channels)

        self.dropout = Dropout(dropout_rate)

        # x   [B, C, H, W]
    def forward(self, bev_feature, prior_feature):
        bev_feature = self.bev_patch_embedding(bev_feature)  # [B, C_hidden, gird_size[0], grid_size[1]]
        bev_feature = bev_feature.flatten(2).transpose(-1, -2)  # [B, n_patches, hidden]

        prior_feature = self.prior_patch_embedding(prior_feature)
        prior_feature = prior_feature.flatten(2).transpose(-1, -2)

        bev_embedding = bev_feature + self.position_embedding_bev
        bev_embedding = self.dropout(bev_embedding)
        bev_embedding = self.norm_bev(bev_embedding)

        prior_embedding = prior_feature + self.position_embedding_prior
        prior_embedding = self.dropout(prior_embedding)
        prior_embedding = self.norm_prior(prior_embedding)

        return bev_embedding, prior_embedding


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, dim, n_levels=1, n_points=8, num_heads=8,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(DeformableTransformerDecoderLayer, self).__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.cross_attn = MSDeformAttn(dim, n_levels, num_heads, n_points)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, tgt, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        shortcut = tgt
        tgt = self.norm1(tgt)
        src = self.norm1(src)

        tgt = self.cross_attn(tgt, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask)

        tgt = shortcut + self.drop_path(tgt)
        tgt = tgt + self.drop_path(self.mlp(self.norm2(tgt)))

        return tgt


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


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MultiScaleDeformableAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        return output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0
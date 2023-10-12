import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch import Tensor
import numpy as np
import sys 
import os
import math
from timm.models.layers import trunc_normal_
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from layers import furthest_point_sample, gather_operation, ball_query, grouping_operation, three_interpolation
from typing import List

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Conv1D(nn.Module):
    def __init__(self, channel_list):
        super().__init__()

        layers = [[nn.Conv1d(channel_list[i], channel_list[i + 1], 1, bias=False),
                   nn.BatchNorm1d(channel_list[i + 1]), nn.LeakyReLU(0.2)] for i in range(len(channel_list) - 1)]
        # flatten out the pairs
        layers = [item for sublist in layers for item in sublist]
        self.conv = nn.Sequential(*layers)

    def forward(self, inp):
        return self.conv(inp)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x) + x


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.spatial_op = nn.Linear(3, dim_head, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, xyzs, features):
        b, l, n, _, h = *features.shape, self.heads

        norm_features = self.norm(features)
        qkv = self.to_qkv(norm_features).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b l n (h d) -> b h (l n) d', h = h), qkv)                             # [b, h, m, d]

        xyzs_flatten = rearrange(xyzs, 'b l n d -> b (l n) d')                                                      # [b, m, 3]

        delta_xyzs = torch.unsqueeze(input=xyzs_flatten, dim=1) - torch.unsqueeze(input=xyzs_flatten, dim=2)        # [b, m, m, 3]

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale                                             # [b, h, m, m]
        attn = dots.softmax(dim=-1)

        v = einsum('b h i j, b h j d -> b h i d', attn, v)                                                          # [b, h, m, d]

        attn = torch.unsqueeze(input=attn, dim=4)                                                                   # [b, h, m, m, 1]
        delta_xyzs = torch.unsqueeze(input=delta_xyzs, dim=1)                                                       # [b, 1, m, m, 3]
        delta_xyzs = torch.sum(input=attn*delta_xyzs, dim=3, keepdim=False)                                         # [b, h, m, 3]

        displacement_features = self.spatial_op(delta_xyzs)                                                         # [b, h, m, d]

        out = v + displacement_features
        out = rearrange(out, 'b h m d -> b m (h d)')
        out =  self.to_out(out)
        out = rearrange(out, 'b (l n) d -> b l n d', l=l, n=n)
        return out + features


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))
    def forward(self, xyzs, features):
        for attn, ff in self.layers:
            features = attn(xyzs, features)
            features = ff(features)
        return features


class P4DConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mlp_planes: List[int],
                 mlp_batch_norm: List[bool],
                 mlp_activation: List[bool],
                 spatial_kernel_size: [float, int],
                 spatial_stride: int,
                 temporal_kernel_size: int,
                 temporal_stride: int = 1,
                 temporal_padding: [int, int] = [0, 0],
                 temporal_padding_mode: str = 'replicate',
                 operator: str = 'addition',
                 spatial_pooling: str = 'max',
                 temporal_pooling: str = 'sum',
                 bias: bool = False):

        super().__init__()

        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_activation = mlp_activation

        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.temporal_padding_mode = temporal_padding_mode

        self.operator = operator
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling

        conv_d = [nn.Conv2d(in_channels=4, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
        if mlp_batch_norm[0]:
            conv_d.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
        if mlp_activation[0]:
            conv_d.append(nn.ReLU(inplace=True))
        self.conv_d = nn.Sequential(*conv_d)

        if in_planes != 0:
            conv_f = [nn.Conv2d(in_channels=in_planes, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
            if mlp_batch_norm[0]:
                conv_f.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
            if mlp_activation[0]:
                conv_f.append(nn.ReLU(inplace=True))
            self.conv_f = nn.Sequential(*conv_f)

        mlp = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=mlp_planes[i]))
            if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)


    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            xyzs: torch.Tensor
                 (B, T, N, 3) tensor of sequence of the xyz coordinates
            features: torch.Tensor
                 (B, T, C, N) tensor of sequence of the features
        """
        device = xyzs.get_device()

        nframes = xyzs.size(1)
        npoints = xyzs.size(2)

        assert (self.temporal_kernel_size % 2 == 1), "P4DConv: Temporal kernel size should be odd!"
        assert ((nframes + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride == 0), "P4DConv: Temporal length error!"

        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]

        if self.temporal_padding_mode == 'zeros':
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]
        else:
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]

        if self.in_planes != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

            if self.temporal_padding_mode == 'zeros':
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
            else:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]

        new_xyzs = []
        new_features = []
        for t in range(self.temporal_kernel_size//2, len(xyzs)-self.temporal_kernel_size//2, self.temporal_stride):                 # temporal anchor frames
            # spatial anchor point subsampling by FPS
            anchor_idx = furthest_point_sample(xyzs[t], npoints//self.spatial_stride)                               # (B, N//self.spatial_stride)
            anchor_xyz_flipped = gather_operation(xyzs[t].transpose(1, 2).contiguous(), anchor_idx)                 # (B, 3, N//self.spatial_stride)
            anchor_xyz_expanded = torch.unsqueeze(anchor_xyz_flipped, 3)                                                            # (B, 3, N//spatial_stride, 1)
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()                                                            # (B, N//spatial_stride, 3)

            new_feature = []
            for i in range(t-self.temporal_kernel_size//2, t+self.temporal_kernel_size//2+1):
                neighbor_xyz = xyzs[i]

                idx = ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)

                neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()                                                    # (B, 3, N)
                neighbor_xyz_grouped = grouping_operation(neighbor_xyz_flipped, idx)                                # (B, 3, N//spatial_stride, k)

                xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded                                                       # (B, 3, N//spatial_stride, k)
                t_displacement = torch.ones((xyz_displacement.size()[0], 1, xyz_displacement.size()[2], xyz_displacement.size()[3]), dtype=torch.float32, device=device) * (i-t)
                displacement = torch.cat(tensors=(xyz_displacement, t_displacement), dim=1, out=None)                               # (B, 4, N//spatial_stride, k)
                displacement = self.conv_d(displacement)

                if self.in_planes != 0:
                    neighbor_feature_grouped = grouping_operation(features[i], idx)                                 # (B, in_planes, N//spatial_stride, k)
                    feature = self.conv_f(neighbor_feature_grouped)
                    if self.operator == '+':
                        feature = feature + displacement
                    else:
                        feature = feature * displacement
                else:
                    feature = displacement

                feature = self.mlp(feature)
                if self.spatial_pooling == 'max':
                    feature = torch.max(input=feature, dim=-1, keepdim=False)[0]                                                        # (B, out_planes, n)
                elif self.spatial_pooling == 'sum':
                    feature = torch.sum(input=feature, dim=-1, keepdim=False)
                else:
                    feature = torch.mean(input=feature, dim=-1, keepdim=False)

                new_feature.append(feature)
            new_feature = torch.stack(tensors=new_feature, dim=1)
            if self.temporal_pooling == 'max':
                new_feature = torch.max(input=new_feature, dim=1, keepdim=False)[0]
            elif self.temporal_pooling == 'sum':
                new_feature = torch.sum(input=new_feature, dim=1, keepdim=False)
            else:
                new_feature = torch.mean(input=new_feature, dim=1, keepdim=False)
            new_xyzs.append(anchor_xyz)
            new_features.append(new_feature)

        new_xyzs = torch.stack(tensors=new_xyzs, dim=1)
        new_features = torch.stack(tensors=new_features, dim=1)

        return new_xyzs, new_features


class LatentAttn(nn.Module):
    def __init__(self, width, length, head=8, ratio=1., num_basis=12, depth=1):
        super().__init__()
        self.width = width
        self.num_basis = num_basis
        self.ratio = ratio
        self.length = length

        # basis
        self.modes_list = (1.0 / float(num_basis)) * torch.tensor([i for i in range(num_basis)],
                                                                  dtype=torch.float).cuda()
        self.weights = nn.Parameter(
            (1 / (width)) * torch.rand(width, self.num_basis * 2, dtype=torch.float))
        # latent
        self.head = head
        self.depth = depth
        self.mask_token = nn.Parameter(
            (1 / (width)) * torch.rand(1, width, 1, dtype=torch.float))
        trunc_normal_(self.mask_token, std=.02)
        # self.pos_emb = nn.Embedding(length, self.width)
        self.encoder_attn = nn.Conv1d(self.width, self.width * 3, kernel_size=1, stride=1)
        self.decoder_attn = nn.Conv1d(self.width, self.width, kernel_size=1, stride=1)
        # self.drop = nn.Dropout(0.)
        self.fc = Conv1D([self.width, self.width, 3])
        self.softmax = nn.Softmax(dim=-1)

    def self_attn(self, q, k, v):
        # q,k,v: B H L C/H
        dim = q.shape[-1]
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k) * (dim ** 0.5))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)

    def encoder(self, x):
        # x: B C N
        B, C, N = x.shape

        x_tmp = self.encoder_attn(x)
        x_tmp = x_tmp.view(B, 3, C, N).permute(1, 0, 2, 3).contiguous()
        x_tmp = x_tmp.view(3, B, self.head, C // self.head, N).permute(0, 1, 2, 4, 3).contiguous()  # 3, B, h, N, C/h

        attn = self.self_attn(x_tmp[0], x_tmp[1], x_tmp[2])
        attn = attn.permute(0, 1, 3, 2).contiguous().view(B, C, -1)

        # transition
        attn_modes = self.get_basis(attn)
        attn = self.compl_mul2d(attn_modes, self.weights) + x

        return attn

    def decoder(self, x, attn):
        # x: B C N
        B, C, N = x.shape
        x_init = x
        attn = attn.view(B, self.head, C // self.head, self.length).permute(0, 1, 3, 2).contiguous()
        x_que = self.decoder_attn(x).view(B, self.head, C // self.head, -1).permute(0, 1, 3, 2).contiguous()

        x = self.self_attn(x_que, attn, attn)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C, N) + x_init

        return x

    def get_basis(self, x):
        # x: B C N
        x_sin = torch.sin(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        x_cos = torch.cos(self.modes_list[None, None, None, :] * x[:, :, :, None] * math.pi)
        return torch.cat([x_sin, x_cos], dim=-1)

    def compl_mul2d(self, input, weights):
        return torch.einsum("bilm,im->bil", input, weights)

    def _mask(self, x):
        with torch.no_grad():
            B, C, N = x.shape
            len_keep = int(N * (1 - self.ratio))

            assert len_keep < 1   # mask all

            if len_keep == 0:
                x_masked = self.mask_token.repeat(B, 1, N)
                mask = torch.ones([B, N], dtype=torch.long, device=x.device)
                return x_masked, mask

            noise = torch.rand(B, N, device=x.device)  # noise in [0, 1]

            index_sort = torch.argsort(noise, dim=1)
            index_restore = torch.argsort(index_sort, dim=1)

            index_keep = index_sort[:, :len_keep]
            x_keep = torch.gather(x, dim=-1, index=index_keep.unsqueeze(1).repeat(1, C, 1))

            mask_token = self.mask_token.repeat(B, 1, N - len_keep)

            # generate the binary mask: 0 is keep, 1 is remove
            mask = torch.ones([B, N], dtype=torch.long, device=x.device)
            mask[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask = torch.gather(mask, dim=1, index=index_restore)

            x_masked = torch.cat([x_keep, mask_token], dim=-1)
            x_masked = torch.gather(x_masked, dim=-1, index=index_restore.unsqueeze(1).repeat(1, C, 1))

        return x_masked, mask

    def forward_feat(self, xt, x_masked):
        for i in range(self.depth):
            # encoder
            xt = self.encoder(xt)  # b, c, token
            # decoder
            x_masked = self.decoder(x_masked, xt)
        return x_masked

    def forward(self, xt, xn):
        # xt -> b, c, t  xn -> b, c, n
        B, C, L = xt.shape
        t = torch.arange(L, dtype=torch.long, device=xt.device).unsqueeze(0).repeat(B, 1)
        # xt = xt + self.pos_emb(t).transpose(1, 2)

        x_masked, _ = self._mask(xn)
        xn_pre = self.forward_feat(xt, x_masked)

        return xn_pre.contiguous(), self.fc(xn_pre).transpose(1, 2).contiguous()


class PSTTransformer(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()
        self.pretrain = args.pretrain
        (radius, nsamples, spatial_stride, temporal_kernel_size,
         temporal_stride, dim, depth, heads, dim_head, mlp_dim, dropout1, dropout2) = \
            (args.radius, args.nsamples, args.spatial_stride, args.temporal_kernel_size,
             args.temporal_stride, args.dim, args.depth, args.heads, args.dim_head, args.mlp_dim, args.dropout1, args.dropout2)

        self.tube_embedding = P4DConv(in_planes=0, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride, temporal_padding=[1, 0],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        if self.pretrain:
            self.latentattn = LatentAttn(dim, args.clip_len // temporal_stride, head=2, num_basis=12, depth=3)
            self.loss = nn.CrossEntropyLoss()
        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout2),
                nn.Linear(mlp_dim, num_classes),
            )

    def forward(self, input, videoname):  # [B, L, N, 3]
        device = input.get_device()

        output_dict = {'pre_loss': 0}

        xyzs, features = self.tube_embedding(input)  # [B, L, n, 3], [B, L, C, n]
        features = features.permute(0, 1, 3, 2)
        output = self.transformer(xyzs, features)  # B, L, N, C
        if self.pretrain:
            feat = output.permute(0, 3, 1, 2)  # batch, c, t, n
            xt = feat.max(-1)[0]  # -> batch, c, t
            xn = feat.max(-2)[0]  # -> batch, c, n
            batch, c, n = xn.shape

            xyz = xyzs.permute(0, 3, 1, 2)  # batch, 3, t, n
            xyz_gt = xyz.view(batch, 3, -1).transpose(1, 2).contiguous()  # -> batch, tn, 3

            xn_pre, xyz_pre = self.latentattn(xt, xn)  # batch, c, n  batch, 3, n

            feat_pre = three_interpolation(xyz_gt, xyz_pre, xn_pre).view(batch, c, -1, n)  # batch, c, t, n
            feat_pre = feat_pre.max(-2)[0].transpose(1, 2).contiguous().view(-1, c)
            feat_pre = F.normalize(feat_pre, dim=-1)  # batch*n, c

            feat_tgt = feat.permute(0, 2, 3, 1).contiguous().view(-1, c)  # batch*tn, c
            feat_tgt = F.normalize(feat_tgt, dim=-1)  # batch*tn, c

            xn = xn.transpose(1, 2).contiguous().view(-1, c)
            xn = F.normalize(xn, dim=-1)  # batch*n, c

            feat_tgt = torch.cat([xn, feat_tgt], dim=0)  # batch*(n+1)

            # InfoNCE loss
            temporal_similarity = torch.matmul(feat_pre, feat_tgt.transpose(0, 1)) / 0.01  # batch*n, batch*t(n+1)
            target_similarity = torch.arange(temporal_similarity.shape[0], device=temporal_similarity.device)
            loss = self.loss(temporal_similarity, target_similarity)

            output_dict['temporal_similarity'] = temporal_similarity
            output_dict['target_similarity'] = target_similarity
            output_dict['pre_loss'] = loss

        if not self.pretrain:
            output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
            output = torch.max(input=output, dim=1, keepdim=False, out=None)[0]
            output = self.classifier(output)

        output_dict['logit'] = output

        return output_dict


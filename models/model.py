import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from einops import rearrange, reduce, repeat
from math import sqrt
from torchvision.models.resnet import resnet34

from .base import BaseModel
from .model_util import copy_states, CropParameters, CropSize
from .submodules import *
from .model_util import *
from .unet import *
from myutils.vis_events.matplotlib_plot_events import *
from models.DCNv2.dcn_v2 import DCN_sep


class FeatsExtract(BaseModel):
    def __init__(self, basech=16, norm=None, activation='relu'):
        super().__init__()

        self.convblock = nn.ModuleList([
            ConvLayer(in_channels=basech, out_channels=2*basech, kernel_size=3, stride=2, padding=1, norm=norm, activation=activation),

            ConvLayer(in_channels=2*basech, out_channels=4*basech, kernel_size=3, stride=2, padding=1, norm=norm, activation=activation),

            ConvLayer(in_channels=4*basech, out_channels=8*basech, kernel_size=3, stride=2, padding=1, norm=norm, activation=activation),
        ])

    def forward(self, input):
        """
        input: torch.tensor, Bx2xHxW
        return: torch.tensor, BxCxHxW
        """
        x = input

        out_list = []
        for i in range(len(self.convblock)):
            x = self.convblock[i](x)
            out_list.append(x)
        out_list = out_list[::-1]

        return out_list


class TimePropagation(BaseModel):
    def __init__(self, basech=16, norm=None, activation='relu', 
                            has_ltc=True, has_gtc=True, gtc_frozen=False):
        super().__init__()
        self.has_ltc = has_ltc
        self.has_gtc = has_gtc
        self.gtc_frozen = gtc_frozen
        assert has_ltc or has_gtc

        # local time correlation
        if has_ltc:
            self.pred_map = nn.Sequential(
                ConvLayer(in_channels=2*basech, out_channels=basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation),
                ConvLayer(in_channels=basech, out_channels=1, kernel_size=3, stride=1, padding=1, norm=norm, activation='sigmoid'),
            )
            self.local_fusion = nn.Sequential(
                ResidualBlock(in_channels=3*basech, out_channels=3*basech, norm=norm),
                ConvLayer(in_channels=3*basech, out_channels=basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=None)
            )

        # global time correlation
        if has_gtc:
            self.lstm = RecurrentConvLayer(in_channels=basech, out_channels=basech, kernel_size=3, stride=1, padding=1, recurrent_block_type='convgru', norm=norm, activation=activation)
            self.global_fusion = ConvLayer(in_channels=2*basech, out_channels=basech, kernel_size=1, stride=1, padding=0, norm=norm, activation=activation)
        self.states = [None] * 2

    def reset_states(self):
        self.states = [None] * 2

    def local_time_corre(self, feat0, feat1, feat2):
        """
        feat0, feat1, feat2: torch.tensor, BxCxHxW
        return: torch.tensor, BxCxHxW
        """
        map0 = self.pred_map(torch.cat([feat0, feat1], dim=1)) # Bx1xHxW
        map1 = self.pred_map(torch.cat([feat1, feat2], dim=1)) # Bx1xHxW
        feat0_mapped = feat0 * map0
        feat2_mapped = feat2 * map1
        output = self.local_fusion(torch.cat([feat0_mapped, feat1, feat2_mapped], dim=1))
        output += feat1

        return output

    def global_time_corre(self, input):
        """
        input: torch.tensor, BxNxCxHxW
        return: torch.tensor, BxNxCxHxW
        """
        B, N, C, H, W = input.size()
        rev_idx = list(reversed(range(N)))
        input_rev = input[:, rev_idx, ...]

        x_list = []
        rev_list = []
        state = self.states[0]
        state_rev = self.states[1]
        for i in range(N):
            if self.gtc_frozen:
                state, state_rev = None, None
            x, state = self.lstm(input[:, i, ...], state)
            rev, state_rev = self.lstm(input_rev[:, i, ...], state_rev)
            x_list.append(x)
            rev_list.append(rev)
        if self.gtc_frozen:
            state, state_rev = None, None
        self.states[0] = state
        self.states[1] = state_rev

        x = torch.stack(x_list, dim=1) # BxNxCxHxW 
        rev = torch.stack(rev_list, dim=1) # BxNxCxHxW
        rev = rev[:, rev_idx, ...] # BxNxCxHxW

        output = torch.cat([x, rev], dim=2).view(B*N, -1, H, W) # BNx2CxHxW
        output = self.global_fusion(output)
        output = output.view(B, -1, C, H, W) # BxNxCxHxW

        return output

    def forward(self, input):
        """
        input: torch.tensor, BxNxCxHxW
        return: torch.tensor, BxNxCxHxW
        """
        B, N, C, H, W = input.size()

        if self.has_ltc:
            feat_list = []
            for i in range(N):
                if i == 0:
                    idx = [0, 0, 1]
                elif i == N - 1:
                    idx = [N -2, N - 1, N - 1]
                else:
                    idx = [i - 1, i, i + 1]

                feat_list.append(self.local_time_corre(input[:, idx[0], ...], input[:, idx[1], ...], input[:, idx[2], ...]))
            feats = torch.stack(feat_list, dim=1)
        else:
            feats = input

        if self.has_gtc:
            feats = self.global_time_corre(feats)

        feats = feats + input

        return feats


class STFusion(BaseModel):
    def __init__(self, basech=16, num_frame=3, norm=None, activation='relu', 
                    has_dcnatten=True, has_scaleaggre=True):
        super().__init__()
        self.has_dcnatten = has_dcnatten
        self.has_scaleaggre = has_scaleaggre
        assert has_dcnatten or has_scaleaggre
        self.num_frame = num_frame
        self.mid_idx = (self.num_frame - 1) // 2
        assert (num_frame + 1) % 2 == 0
        assert num_frame >= 3

        if has_dcnatten:
            self.offset = nn.Sequential(
                ConvLayer(in_channels=2*basech, out_channels=basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation),
                ConvLayer(in_channels=basech, out_channels=basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=None),
            )  
            self.dcn = DCN_sep(basech, basech, 3, stride=1, padding=1, dilation=1, deformable_groups=8)
            self.activation = nn.ReLU()

            self.convblock = nn.Sequential(
                ConvLayer(in_channels=2*basech, out_channels=basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation),
                ConvLayer(in_channels=basech, out_channels=basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=None),
            )
            self.kernel = ConvLayer(in_channels=basech, out_channels=2, kernel_size=1, stride=1, padding=0, norm=norm, activation='sigmoid')
            self.fc = nn.Sequential(
                MLP(input_dim=basech, hidden_dim=basech//2, output_dim=2*basech, num_layers=2),
                nn.Sigmoid()
            )
            self.dcn_fusion = nn.Sequential(
                ConvLayer(in_channels=2*basech, out_channels=basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation),
                ConvLayer(in_channels=basech, out_channels=basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=None),
            )

        self.dense_fusion = nn.Sequential(
            ConvLayer(in_channels=num_frame*basech, out_channels=basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation),
            ConvLayer(in_channels=basech, out_channels=basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=None),
        )

        if has_scaleaggre:
            self.attens = nn.ModuleList([
                ConvLayer(in_channels=basech, out_channels=1, kernel_size=3, stride=1, padding=1, norm=norm, activation='sigmoid'),
                ConvLayer(in_channels=basech//2, out_channels=1, kernel_size=3, stride=1, padding=1, norm=norm, activation='sigmoid'),
                ConvLayer(in_channels=basech//4, out_channels=1, kernel_size=3, stride=1, padding=1, norm=norm, activation='sigmoid'),
            ])

        self.recons = nn.ModuleList([
            UpsampleConvLayer(in_channels=basech, out_channels=basech//2, kernel_size=3, stride=1, padding=1, norm=norm),
            UpsampleConvLayer(in_channels=basech//2, out_channels=basech//4, kernel_size=3, stride=1, padding=1, norm=norm),
            UpsampleConvLayer(in_channels=basech//4, out_channels=basech//8, kernel_size=3, stride=1, padding=1, norm=norm),
        ])

    def fuse(self, feat0, feat1):
        """
        feat0: torch.tensor, BxCxHxW
        feat1: torch.tensor, BxCxHxW
        return: torch.tensor, BxCxHxW
        """
        B, C, H, W = feat0.size()

        offset = self.offset(torch.cat([feat0, feat1], dim=1))
        feat0_aligned = self.activation(self.dcn(feat0, offset))

        feat = self.convblock(torch.cat([feat0_aligned, feat1], dim=1))
        spatial_kernel = self.kernel(feat) # Bx2xHxW
        channel_kernel = self.fc(feat.view(B, C, H*W).transpose(1, 2).max(1, keepdim=True)[0]) # Bx1x2C
        channel_kernel = channel_kernel.transpose(1, 2).unsqueeze(-1) # Bx2Cx1x1

        y0 = feat0_aligned * spatial_kernel[:, [0], ...]
        y0 = y0 * channel_kernel[:, :C, ...]
        y1 = feat1 * spatial_kernel[:, [1], ...]
        y1 = y1 * channel_kernel[:, C:, ...]

        out = self.dcn_fusion(torch.cat([y0, y1], dim=1))

        return out

    def dense_fuse(self, input):
        """
        input: torch.tensor, BxNxCxHxW
        return: torch.tensor, BxCxHxW
        """
        if self.has_dcnatten:
            out_list = []
            idxes = list(range(self.mid_idx)) + list(range(self.mid_idx+1, self.num_frame))
            for i in idxes:
                out_list.append(self.fuse(input[:, i, ...].contiguous(), input[:, self.mid_idx, ...].contiguous()))

            out_list.append(input[:, self.mid_idx, ...].contiguous())
            out = torch.cat(out_list, dim=1) # BxNCxHxW
        else:
            out = input.view(input.size(0), -1, input.size(-2), input.size(-1)) # BxNCxHxW

        out = self.dense_fusion(out)

        return out

    def scale_aggre(self, input, feats, scale_idx):
        """
        input: torch.tensor, BxCxHxW
        feats: torch.tensor, BxNxCxHxW
        return: torch.tensor, BxC1x2Hx2W
        """
        if self.has_scaleaggre:
            B, N, C, H, W = feats.size()
            feats = feats.view(B*N, -1, H, W) # BNxCxHxW
            atten = self.attens[scale_idx](feats) # BNx1xHxW
            feats = feats * atten # BNxCxHxW
            feats = feats.view(B, N, -1, H, W) # BxNxCxHxW
            feats = feats.mean(1) # BxCxHxW
            # out = torch.cat([input, feats], dim=1) # Bx2CxHxW
            out = input + feats # BxCxHxW
        else:
            out = input # BxCxHxW

        out = self.recons[scale_idx](out) # BxC1x2Hx2W 

        return out

    def forward(self, input, feats_list):
        """
        input: torch.tensor, BxNxCxHxW
        feats_list: [torch.tensor], BxNxCxHxW, BxNxC1x2Hx2W, ...
        return: torch.tensor, BxC3x8Hx8W
        """
        B, N, C, H, W = input.size()
        x = input
        assert x.size(1) == self.num_frame

        x = self.dense_fuse(x) # BxCxHxW

        for idx, feats in enumerate(feats_list):
            feats = feats.view(B, N, -1, feats.size(-2), feats.size(-1))
            x = self.scale_aggre(x, feats, idx)

        return x


class DeepRecurrNet(BaseModel):
    def __init__(self, inch=2, basech=16, num_frame=3, norm=None, activation='relu', 
                has_ltc=True, has_gtc=True, gtc_frozen=False,
                has_dcnatten=True, has_scaleaggre=True):
        super().__init__()
        self.down_scale = 8

        self.head = ConvLayer(in_channels=inch, out_channels=basech, kernel_size=3, stride=1, padding=1, norm=norm, activation=activation)

        self.feat_extract = FeatsExtract(basech=basech, norm=norm, activation=activation)
        self.time_propagate = TimePropagation(basech=self.down_scale*basech, norm=norm, activation=activation,
                                has_ltc=has_ltc, has_gtc=has_gtc, gtc_frozen=gtc_frozen)
        self.spacetime_fuse = STFusion(basech=self.down_scale*basech, num_frame=num_frame, norm=norm, activation=activation,
                                has_dcnatten=has_dcnatten, has_scaleaggre=has_scaleaggre)
        
        self.tail = ConvLayer(in_channels=basech, out_channels=inch, kernel_size=3, stride=1, padding=1, norm=norm, activation='relu')

    def reset_states(self):
        self.time_propagate.reset_states()

    def forward(self, input):
        """
        input: torch.tensor, BxNx2xHxW
        return: torch.tensor, Bx2xHxW
        """
        B, N, C, H, W = input.size()

        # pad input
        x = input
        factor = {'h': 8, 'w': 8}
        need_crop = (H % factor['h'] != 0) or (W % factor['w'] != 0)
        pad_crop = CropSize(W, H, factor) if need_crop else None
        if need_crop and pad_crop:
            x = pad_crop.pad(x)

        x = x.view(B*N, -1, x.size(-2), x.size(-1))
        x = self.head(x)  
        feats_list = self.feat_extract(x)
        x = feats_list[0]

        x = x.view(B, N, -1, x.size(-2), x.size(-1))
        x = self.time_propagate(x)
        x = self.spacetime_fuse(x, feats_list)
        x = self.tail(x)

        # crop output
        if need_crop and pad_crop:
            x = pad_crop.crop(x)
            x = x.contiguous()

        return x






# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

from net.BasicBlock import UNetEncoder, UNetDecoder, UNetUpSamplingBlock, UNetConvBlock, ConvNormRelu, UNetUpSamplingBlock
from net.Network_SwinT import SwinTransformer


class TFusion(nn.Module):
    def __init__(self, embedding_dim=1024, feature_size=8):
        super(TFusion, self).__init__()
        self.embedding_dim = embedding_dim
        self.d_model = self.embedding_dim
        self.modalities_num = 4
        self.patch_dim = min(feature_size, 8)
        self.scale_factor = feature_size // self.patch_dim

        self.fusion_block = SwinTransformer(
            patch_size=4,
            in_chans=embedding_dim,
            embed_dim=embedding_dim,
            window_size=8,
            layer = 1)

        self.conv = nn.Conv2d(self.embedding_dim * 2, self.embedding_dim, kernel_size=3, stride=1, padding=1)
        self.relu = nn.LeakyReLU(0.01)


    def forward(self, all_content, m_d, pre_am):
        n_modality = len(all_content)

        # concat with dim = 2 -> H from (B, C, H, W)
        concated_content = None
        H = all_content[0].size(2)
        for i in range(n_modality):
            if concated_content is None:
                concated_content = all_content[i]
            else:
                concated_content = torch.cat([concated_content, all_content[i]], dim=2)

        out0 = self.fusion_block(concated_content)
        out = out0[-1]


        a_m = None
        for i in range(n_modality):
            atten_mapi = out[:, :, H*i:H*(i+1), :]
            if pre_am is not None:
                atten_mapi = self.conv(torch.cat([atten_mapi, pre_am[i]],dim=1))

            atten_mapi1 = atten_mapi.unsqueeze(dim=0)
            if a_m == None:
                a_m = atten_mapi1
            else:
                a_m = torch.cat([a_m, atten_mapi1], dim=0)

        a_m2 = F.softmax(a_m, dim=0)
        torch.cuda.empty_cache()
        res = self.atten(all_content, a_m2, n_modality)

        return res, a_m

    def atten(self, all_content, atten_map, n_modality):
        output = None
        for i in range(n_modality):
            a_map = atten_map[i, :, :, :, :]
            assert all_content[i].shape == a_map.shape, 'all_content and a_m cannot match!!'
            if output == None:
                output = all_content[i] * a_map
            else:
                output += all_content[i] * a_map
        return output

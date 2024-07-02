import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from fightingcv_attention.attention.ShuffleAttention import *

class CLIP_Adapter(nn.Module):
    # map_num :M_Blk_num
    def __init__(self, in_ch, mid_ch, out_ch, G_ch, CLIP_ch, 
                 cond_dim, k, s, p, map_num, CLIP, hidden_size):
        super(CLIP_Adapter, self).__init__()
        self.CLIP_ch = CLIP_ch
        self.FBlocks = nn.ModuleList([])
        self.FBlocks.append(M_Block_RAT(in_ch, mid_ch, out_ch, cond_dim, k, s, p, hidden_size))
        for i in range(map_num-1):
            self.FBlocks.append(M_Block_RAT(out_ch, mid_ch, out_ch, cond_dim, k, s, p, hidden_size))
            
        self.conv_fuse = nn.Conv2d(out_ch, CLIP_ch, 5, 1, 2)
        # self.toclip = nn.Sequential(
        #     nn.ConvTranspose2d(512, 256, 3, 2, 1, 1),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.2, inplace=True),
        #     nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
        #     nn.BatchNorm2d(64),
        #     nn.Sigmoid(),
        # )
        # self.CLIP = CLIP
        self.conv = nn.Conv2d(512, G_ch, 5, 1, 2)

    def forward(self,out,c):
        for FBlock in self.FBlocks:
            out = FBlock(out,c)
        fuse_feat = self.conv_fuse(out)
        # in_feat = self.toclip(fuse_feat)
        # map_feat = self.CLIP.encode_image(in_feat, gen=True)
        # return self.conv(fuse_feat+0.1*map_feat)
        return self.conv(fuse_feat)

class NetG(nn.Module):
    def __init__(self, ngf, nz, cond_dim, imsize, CLIP):
        super(NetG, self).__init__()
        self.ngf = ngf
        # build CLIP Mapper
        self.code_sz, self.code_ch, self.mid_ch = 8, 64, 32
        self.CLIP_ch = 512
        self.fc_code = nn.Linear(nz, self.code_sz*self.code_sz*self.code_ch)
        self.mapping = CLIP_Adapter(self.code_ch, self.mid_ch, 
                                    self.code_ch, ngf*8, self.CLIP_ch, 
                                    cond_dim+nz, 3, 1, 1, 4, 
                                    CLIP, hidden_size=64
                                    )
        # build GBlocks
        self.GBlocks = nn.ModuleList([])
        in_out_pairs = list(get_G_in_out_chs(ngf, imsize))
        imsize = 4
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            if idx<(len(in_out_pairs)-1):
                imsize = imsize*2
            else:
                imsize = 256
            self.GBlocks.append(G_Block(cond_dim+nz, in_ch, out_ch, imsize))
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2,inplace=True),
            nn.Conv2d(out_ch, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, noise, c): # x=noise, c=ent_emb
        with torch.cuda.amp.autocast():
            cond = torch.cat((noise, c), dim=1)
            out = self.mapping(self.fc_code(noise).view(noise.size(0), 
                                                        self.code_ch, 
                                                        self.code_sz, 
                                                        self.code_sz), 
                                                        cond)
            # fuse text and visual features
            for GBlock in self.GBlocks:
                out = GBlock(out, cond)
            # convert to RGB image
            out = self.to_rgb(out)
        return out

class NetD(nn.Module):
    def __init__(self):
        super(NetD, self).__init__()
        self.DBlocks = nn.ModuleList([
            D_Block(512, 512, 3, 1, 1, res=True, CLIP_feat=True),
            D_Block(512, 512, 3, 1, 1, res=True, CLIP_feat=True),
            D_Block(512, 512, 3, 1, 1, res=True, CLIP_feat=True),
        ])
        
        self.main = D_Block(512, 512, 3, 1, 1, res=True, CLIP_feat=False)

        self.process = nn.ModuleList([
            nn.Sequential(
                Conv(128, 256, 3, 2, 1),
                Conv(256, 512, 3, 2, 1)
            ),
            nn.Sequential(
                Conv(256, 512, 3, 2, 1)
            ),
            nn.Identity(),
            nn.Identity()
        ])

    def forward(self, h):
        with torch.cuda.amp.autocast():
            r = []
            for i in range(len(self.process)):
                r.append(self.process[i](h[i]))
            h = torch.stack(r, dim=1)
            out = h[:,0]
            for idx in range(len(self.DBlocks)):
                out = self.DBlocks[idx](out, h[:,idx+1])
            out = self.main(out)
        return out, h

class NetC(nn.Module):
    def __init__(self, cond_dim):
        super(NetC, self).__init__()
        self.cond_dim = cond_dim
        self.joint_conv = nn.Sequential(
            nn.Conv2d(512+512, 128, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
        )

    def forward(self, out, cond):
        with torch.cuda.amp.autocast():
            cond = cond.view(-1, self.cond_dim, 1, 1)
            cond = cond.repeat(1, 1, 8, 8)
            h_c_code = torch.cat((out, cond), 1)
            out = self.joint_conv(h_c_code)
        return out

class M_Block(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, cond_dim, k, s, p):
        super(M_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, k, s, p)
        self.fuse1 = DFBLK(cond_dim, mid_ch)
        self.conv2 = Conv_shuffle(mid_ch, out_ch, k, s, p)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        self.learnable_sc = in_ch != out_ch
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, text):
        h = self.conv1(h)
        h = self.fuse1(h, text)
        h = self.conv2(h)
        h = self.fuse2(h, text)
        return h

    def forward(self, h, c):
        return self.shortcut(h) + self.residual(h, c)

class M_Block_RAT(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, cond_dim, k, s, p, hidden_size):
        super(M_Block_RAT, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, k, s, p)
        self.fuse1 = DFBLK_rnn(cond_dim, mid_ch, hidden_size)
        self.conv2 = Conv_shuffle(mid_ch, out_ch, k, s, p)# SA
        # self.conv2 = nn.Conv2d(mid_ch, out_ch, k, s, p)# 
        self.fuse2 = DFBLK_rnn(cond_dim, out_ch, hidden_size)
        self.learnable_sc = in_ch != out_ch
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, text):
        h = self.conv1(h)
        h = self.fuse1(h, text)
        h = self.conv2(h)
        # h = ShuffleAttention(h)
        h = self.fuse2(h, text)
        return h

    def forward(self, h, c):
        return self.shortcut(h) + self.residual(h, c)

class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, imsize):
        super(G_Block, self).__init__()
        self.imsize = imsize
        self.learnable_sc = in_ch != out_ch 
        self.c1 = Conv_shuffle(in_ch, out_ch, 3, 1, 1)
        self.c2 = Conv_shuffle(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)
        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch,out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
        return x

    def residual(self, h, y):
        h = self.fuse1(h, y)
        h = self.c1(h)
        h = self.fuse2(h, y)
        h = self.c2(h)
        return h

    def forward(self, h, y):
        h = F.interpolate(h, size=(self.imsize, self.imsize))
        return self.shortcut(h) + self.residual(h, y)

class D_Block(nn.Module):
    def __init__(self, fin, fout, k, s, p, res, CLIP_feat):
        super(D_Block, self).__init__()
        self.res, self.CLIP_feat = res, CLIP_feat
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        if self.res==True:
            self.gamma = nn.Parameter(torch.zeros(1))
        if self.CLIP_feat==True:
            self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, CLIP_feat=None):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if (self.res==True)and(self.CLIP_feat==True):
            return x + self.gamma*res + self.beta*CLIP_feat
        elif (self.res==True)and(self.CLIP_feat!=True):
            return x + self.gamma*res
        elif (self.res!=True)and(self.CLIP_feat==True):
            return x + self.beta*CLIP_feat
        else:
            return x

# normal DF
class DFBLK(nn.Module):
    def __init__(self, cond_dim, in_ch):
        super(DFBLK, self).__init__()
        self.affine0 = Affine(cond_dim, in_ch)
        self.affine1 = Affine(cond_dim, in_ch)

    def forward(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return h


# rnn DF
class DFBLK_rnn(nn.Module):
    def __init__(self, cond_dim, in_ch, hidden_size):
        super(DFBLK_rnn, self).__init__()
        self.affine0 = Affine_rnn(cond_dim, in_ch, hidden_size)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.affine1 = Affine_rnn(cond_dim, in_ch, hidden_size)

    def forward(self, x, y=None):
        h = self.affine0(x, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        h = self.affine1(h, y)
        h = nn.LeakyReLU(0.2,inplace=True)(h)
        return h

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# CAT
class Affine(nn.Module):
    def __init__(self, cond_dim, num_features=2):
        super(Affine, self).__init__()

        self.fc_gamma = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self.fc_beta = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(cond_dim, num_features)),
            ('relu1',nn.ReLU(inplace=True)),
            ('linear2',nn.Linear(num_features, num_features)),
            ]))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma.linear2.weight.data)
        nn.init.ones_(self.fc_gamma.linear2.bias.data)
        nn.init.zeros_(self.fc_beta.linear2.weight.data)
        nn.init.zeros_(self.fc_beta.linear2.bias.data)

    def forward(self, x, y=None):
        weight = self.fc_gamma(y)
        bias = self.fc_beta(y)        

        # torch.Size([x,y])
        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias

# RAT
class Affine_rnn(nn.Module):
    def __init__(self, cond_dim, num_features, rnn_hidden_size):
        super(Affine_rnn, self).__init__()

        self.rnn = nn.LSTM(cond_dim, rnn_hidden_size, batch_first=True)
        self.fc_gamma = nn.Sequential(nn.Linear(rnn_hidden_size, num_features),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(num_features, num_features))
        self.fc_beta = nn.Sequential(nn.Linear(rnn_hidden_size, num_features),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(num_features, num_features))
        self._initialize()

    def _initialize(self):
        nn.init.zeros_(self.fc_gamma[-1].weight.data)
        nn.init.ones_(self.fc_gamma[-1].bias.data)
        nn.init.zeros_(self.fc_beta[-1].weight.data)
        nn.init.zeros_(self.fc_beta[-1].bias.data)

    def forward(self, x, y=None):
        if y is not None:
            _, (h, c) = self.rnn(y.unsqueeze(0))
        else:
            _, (h, c) = self.rnn(torch.zeros(1, x.size(0), 
                                             self.rnn.hidden_size).to(x.device))

        weight = self.fc_gamma(h.squeeze(0))
        bias = self.fc_beta(h.squeeze(0))

        if weight.dim() == 1:
            weight = weight.unsqueeze(0)
        if bias.dim() == 1:
            bias = bias.unsqueeze(0)

        size = x.size()
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)
        return weight * x + bias

def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs

def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs

# Conv2d with BN
class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.2, inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
 
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
    
# SA Att
class Conv_shuffle(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  
        super(Conv_shuffle, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.2, inplace=True) if act is True else \
                  (act if isinstance(act, nn.Module) else nn.Identity())
        self.se = ShuffleAttention(channel=c2)

    def forward(self, x):
        return self.se(self.act(self.bn(self.conv(x))))
    
    # def fuseforward(self, x):
    #     return self.se(self.act(self.conv(x)))
    
import torch
import torch.nn as nn
from einops import rearrange
import math
import torch.nn.functional as F

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, bias=False,padding=1)

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=True)

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""


class CMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Priority on encoding

        ## Initial values
        self.f_qr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_kr = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sve = nn.Parameter(torch.tensor(0.1),  requires_grad=False)
        self.f_sv = nn.Parameter(torch.tensor(1.0),  requires_grad=False)


        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H), [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2, self.kernel_size, self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)


        # multiply by factors
        qr = torch.mul(qr, self.f_qr)
        kr = torch.mul(kr, self.f_kr)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)

        # multiply by factors
        sv = torch.mul(sv, self.f_sv)
        sve = torch.mul(sve, self.f_sve)

        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output
    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class Attention(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        # heads = inp // dim_head
        heads = 8
        # print("heads:", heads)
        project_out = not (heads == 1 and dim_head == inp)

        self.ih, self.iw = image_size

        self.heads = heads
        self.scale = dim_head ** -0.5

        # parameter table of relative position bias
        self.relative_bias_table = nn.Parameter(
            torch.zeros((2 * self.ih - 1) * (2 * self.iw - 1), heads))

        coords = torch.meshgrid([torch.arange(self.ih), torch.arange(self.iw)])  # , indexing='ij')
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[0] += self.ih - 1
        relative_coords[1] += self.iw - 1
        relative_coords[0] *= 2 * self.iw - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.attend = nn.Softmax(dim=-1)

        self.to_out = nn.Sequential(
            nn.Conv2d(inp, oup, 1),
            nn.Dropout2d(dropout, inplace=True)
        ) if project_out else nn.Identity()

        self.projQ = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 1, 1, groups=1, bias=False),
            nn.GroupNorm(1, inp, eps=1e-6),
            nn.GELU(),
        )
        self.projK = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 1, 1, groups=1, bias=False),
            nn.GroupNorm(1, inp, eps=1e-6),
            nn.GELU()
        )

        self.projV = nn.Sequential(
            nn.Conv2d(inp, inp, 3, 1, 1, groups=1, bias=False),
            nn.GroupNorm(1, inp, eps=1e-6),
            nn.GELU()
        )

    def forward(self, x):
        q = self.projQ(x)
        k = self.projK(x)
        v = self.projV(x)

        q = rearrange(q, 'b c ih iw -> b (ih iw) c')
        k = rearrange(k, 'b c ih iw -> b (ih iw) c')
        v = rearrange(v, 'b c ih iw -> b (ih iw) c')

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Use "gather" for more efficiency on GPUs
        relative_bias = self.relative_bias_table.gather(
            0, self.relative_index.repeat(1, self.heads))
        relative_bias = rearrange(
            relative_bias, '(h w) c -> 1 c h w', h=self.ih * self.iw, w=self.ih * self.iw)
        dots = dots + relative_bias

        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = rearrange(out, 'b (ih iw) c -> b c ih iw', ih=self.ih, iw=self.iw)
        out = self.to_out(out)
        return out




class Transformer(nn.Module):
    def __init__(self, inp, oup, image_size, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        # hidden_dim = int(inp * 4)

        self.ih, self.iw = image_size

        self.attn = Attention(inp, oup, image_size, heads, dim_head, dropout)
        self.mlp = CMLP(inp, 4 * oup, drop=dropout)
        self.norm = nn.GroupNorm(1, oup, eps=1e-6)


    def forward(self, TRANS):
        TRANS = self.norm(TRANS)
        TRANS = self.attn(TRANS) + TRANS
        TRANS = self.norm(TRANS)
        TRANS = self.mlp(TRANS) + TRANS
        return TRANS





class Swin_axial(nn.Module):
    def __init__(self, inp, oup, scale_size):
        super().__init__()
        #(ih , iw ) =image_size

        self.scale_size = scale_size

        patch_12_11 = 11

        self.transformer1 = Transformer(inp, oup, (patch_12_11 * self.scale_size, patch_12_11 * self.scale_size ))
        self.transformer2 = Transformer(inp, oup,  (patch_12_11 * self.scale_size , patch_12_11 * self.scale_size ))
        self.transformer3 = Transformer(inp, oup,  (patch_12_11 * self.scale_size , patch_12_11 * self.scale_size ))
        self.transformer4 = Transformer(inp, oup,  (patch_12_11 * self.scale_size , patch_12_11 * self.scale_size ))

        self.conv_down = nn.Conv2d(inp, inp, kernel_size=3, stride=1, bias=True, padding=1, groups=1)

        #self.diagonal_block1 = AxialAttention(inp, oup, groups=4,kernel_size=12 * self.scale_size,stride=1)       # 小主
        #self.diagonal_block2 = AxialAttention(inp, oup, groups=4, kernel_size=12 * self.scale_size+1,stride=1)    # 小副
        self.hight_block = AxialAttention(inp, oup, groups=4, kernel_size=patch_12_11 * self.scale_size)
        self.width_block = AxialAttention(inp, oup, groups=4, kernel_size=patch_12_11 * self.scale_size, stride=1,width=True)
        self.diagonal_block3 = AxialAttention(inp, oup, groups=4, kernel_size=patch_12_11*2 * self.scale_size, stride=1)     # 大主
        self.diagonal_block4 = AxialAttention(inp, oup, groups=4, kernel_size=patch_12_11*2 * self.scale_size + 1, stride=1) # 大副

        self.hight_block1 = AxialAttention(inp, inp, groups=4, kernel_size=patch_12_11*2 * self.scale_size)


    def forward(self, TRANS):
        #print(TRANS.shape)
        TRANS = rearrange(TRANS, 'b c (p1 h) (p2 w) -> b (p1 p2) c h w ', p1=2, p2=2)
        #print(TRANS.shape)

        TRANS1, TRANS2, TRANS3, TRANS4 = TRANS.unbind(1)

        #TRANS1 = self.transformer1(TRANS1)
        # 副对角线
        # TRANS1 = F.pad(TRANS1, (0, 1, 0, 0), value=0)
        # TRANS1 = TRANS1.reshape(*TRANS1.size()[:-2], TRANS1.size(-1), TRANS1.size(-2))
        # TRANS1 = self.diagonal_block2(TRANS1)
        # TRANS1 = TRANS1.reshape(*TRANS1.size()[:-2], TRANS1.size(-1), TRANS1.size(-2))
        # TRANS1 = F.pad(TRANS1, (0, -1, 0, 0), value=0)
        #TRANS1 = self.width_block(TRANS1) + self.hight_block(TRANS1)

        TRANS1_1 = TRANS1
        TRANS1 = self.hight_block(TRANS1)
        TRANS1 = self.width_block(TRANS1)
        TRANS1 = TRANS1 + TRANS1_1
        TRANS1 = self.conv_down(TRANS1)

        #TRANS2 = self.transformer2(TRANS2)
        # 主对角线
        # TRANS2 = F.pad(TRANS2, (0, 0, 0, 1), value=0)
        # TRANS2 = TRANS2.reshape(*TRANS2.size()[:-2], TRANS2.size(-1), TRANS2.size(-2))
        # TRANS2 = self.diagonal_block1(TRANS2)
        # TRANS2 = TRANS2.reshape(*TRANS2.size()[:-2], TRANS2.size(-1), TRANS2.size(-2))
        # TRANS2 = F.pad(TRANS2, (0, 0, 0, -1), value=0)

        #TRANS2 = self.width_block(TRANS2) + self.hight_block(TRANS2)
        TRANS2_1 = TRANS2
        TRANS2 = self.hight_block(TRANS2)
        TRANS2 = self.width_block(TRANS2)
        TRANS2 = TRANS2 + TRANS2_1
        TRANS2 = self.conv_down(TRANS2)

        #TRANS3 = self.transformer3(TRANS3)
        # 主对角线
        # TRANS3 = F.pad(TRANS3, (0, 0, 0, 1), value=0)
        # TRANS3 = TRANS3.reshape(*TRANS3.size()[:-2], TRANS3.size(-1), TRANS3.size(-2))
        # TRANS3 = self.diagonal_block1(TRANS3)
        # TRANS3 = TRANS3.reshape(*TRANS3.size()[:-2], TRANS3.size(-1), TRANS3.size(-2))
        # TRANS3 = F.pad(TRANS3, (0, 0, 0, -1), value=0)

        #TRANS3 = self.width_block(self.hight_block(TRANS3)) + TRANS3
        TRANS3_1 = TRANS3
        TRANS3 = self.hight_block(TRANS3)
        TRANS3 = self.width_block(TRANS3)
        TRANS3 = TRANS3 + TRANS3_1
        TRANS3 = self.conv_down(TRANS3)

        #TRANS4 = self.transformer4(TRANS4)
        # 副对角线
        # TRANS4 = F.pad(TRANS4, (0, 1, 0, 0), value=0)
        # TRANS4 = TRANS4.reshape(*TRANS4.size()[:-2], TRANS4.size(-1), TRANS4.size(-2))
        # TRANS4 = self.diagonal_block2(TRANS4)
        # TRANS4 = TRANS4.reshape(*TRANS4.size()[:-2], TRANS4.size(-1), TRANS4.size(-2))
        # TRANS4 = F.pad(TRANS4, (0, -1, 0, 0), value=0)

        #TRANS4 = self.width_block(TRANS4) + self.hight_block(TRANS4)
        TRANS4_1 = TRANS4
        TRANS4 = self.hight_block(TRANS4)
        TRANS4 = self.width_block(TRANS4)
        TRANS4 = TRANS4 + TRANS4_1
        TRANS4 = self.conv_down(TRANS4)

        TRANS1 = TRANS1.unsqueeze(1)
        TRANS2 = TRANS2.unsqueeze(1)
        TRANS3 = TRANS3.unsqueeze(1)
        TRANS4 = TRANS4.unsqueeze(1)

        TRANS = torch.cat([TRANS1, TRANS2, TRANS3, TRANS4], dim=1)

        TRANS = rearrange(TRANS, 'b (p1 p2) c h w -> b c (p1 h) (p2 w) ', p1=2, p2=2)

        TRANS_1 = TRANS
        # 拼起来再做一次主对角线和副对角线
        # Main diagonal axial attention

        TRANS = F.pad(TRANS, (0, 0, 0, 1), value=0)

        TRANS = TRANS.reshape(*TRANS.size()[:-2], TRANS.size(-1), TRANS.size(-2))

        TRANS = self.diagonal_block3(TRANS)
        TRANS = TRANS.reshape(*TRANS.size()[:-2], TRANS.size(-1), TRANS.size(-2))

        TRANS = F.pad(TRANS, (0, 0, 0, -1), value=0)


        # Counter diagonal axial attention
        TRANS = F.pad(TRANS, (0, 1, 0, 0), value=0)
        TRANS = TRANS.reshape(*TRANS.size()[:-2], TRANS.size(-1), TRANS.size(-2))
        TRANS = self.diagonal_block4(TRANS)
        TRANS = TRANS.reshape(*TRANS.size()[:-2], TRANS.size(-1), TRANS.size(-2))
        TRANS = F.pad(TRANS, (0, -1, 0, 0), value=0)

        TRANS = TRANS + TRANS_1
        TRANS = self.conv_down(TRANS)


        return TRANS





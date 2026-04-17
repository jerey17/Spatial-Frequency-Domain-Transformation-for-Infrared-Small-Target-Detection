import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import numbers
from einops import rearrange
from thop import profile
import time


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def Up_sample(src,scale):
    src =  F.interpolate(src,scale_factor=scale,mode='bilinear',align_corners=True)
    
    return src

class REBNCONV(nn.Module):
    def __init__(self,in_ch,out_ch,dirate):
        super(REBNCONV,self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self,x):

        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout

class MultiRBC(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(MultiRBC,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        return hx1d + hxin

class ChannelAttention(nn.Module):
    def __init__(self,in_channel,ratio = 3):
        super(ChannelAttention,self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channel,in_channel // ratio,kernel_size=1,bias=False)
        self.fc2 = nn.Conv2d(in_channel // ratio ,in_channel,kernel_size=1,bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        b, c, h, w = x.size()
        max_pool_out = self.fc2(self.fc1(self.max_pool(x)))
        avg_pool_out = self.fc2(self.fc1(self.avg_pool(x)))
        out = max_pool_out + avg_pool_out
        return self.sigmoid(out)
    
    
class SpacialAttention(nn.Module):
    def __init__(self,kernel_size = 7):
        super(SpacialAttention,self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2,1,kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        max_pool_out , _ = torch.max(x,dim=1,keepdim=True)
        avg_pool_out = torch.mean(x,dim=1,keepdim=True)
        out = torch.cat([max_pool_out,avg_pool_out],dim=1)
        out = self.conv1(out)
        return self.sigmoid(out)
    
    
class Res_Block(nn.Module):
    def __init__(self,in_channels,out_channels, stride=1):
        super(Res_Block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,stride=stride,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None
            
        self.ca = ChannelAttention(out_channels)
        self.sa = SpacialAttention()
        
            
    def forward(self,x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
            
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        out += residual
        out = self.relu(out)
        return out
    
    
class FDSA(nn.Module):
    def __init__(self, dim, bias):
        super(FDSA, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output
    


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        
        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features,
                                   bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features,
                                   bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1_3 = self.relu3(self.dwconv3x3(x1))
        x2_5 = self.relu5(self.dwconv5x5(x2))
        x = F.gelu(x1_3) * x2_5
        x = self.project_out(x)
        return x
    
    
class TransformerLayer(nn.Module):
    def __init__(self,dim,ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias',att=False):
       super(TransformerLayer, self).__init__()

       self.att = att
       if self.att:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = FDSA(dim, bias)

       self.norm2 = LayerNorm(dim, LayerNorm_type)
       self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.att:
            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x
    

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
    

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()
    
    
def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(CBN(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(CBN(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class CBN(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(CBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class Fuse(nn.Module):
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)
        self.nConvs = _make_nConv(in_channels=(F_g+F_x),out_channels=F_g,nb_Conv=1,activation='ReLU')

        

    def forward(self, g, x):
        avg_pool_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g) / 2.0
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        x = torch.cat([g,out],dim=1)
        return self.nConvs(x)

# class Fuse(nn.Module):
#     def __init__(self, F_g, F_x):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels=F_x,out_channels=F_g,kernel_size=1)
        
#     def forward(self,g,x):
#         out = g + self.conv(x)
#         return out

class DeconvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(DeconvUpsample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    
    def forward(self, x):
        return self.deconv(x)


class SFDTNet(nn.Module):
    def __init__(self,in_channels,deep_supervision,ffn_expansion=2.66,bias=False,block = Res_Block,mode='train'):
        super(SFDTNet,self).__init__()
        
        block = Res_Block
        param_channels = [16, 32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.mode = mode
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2,2)
        self.bn = nn.BatchNorm2d(16)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.down = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        
        self.inconv = nn.Conv2d(in_channels=in_channels, out_channels=param_channels[0],kernel_size=3, padding=1)
        self.down_encoder1 = self._make_layer(param_channels[0], param_channels[1], block=block)
        self.down_encoder2 = self._make_layer(param_channels[1], param_channels[2], block=block)
        self.down_encoder3 = self._make_layer(param_channels[2], param_channels[3], block=block)
        self.down_encoder4 = self._make_layer(param_channels[3], param_channels[4], block=block)
        self.mid_layer = MultiRBC(param_channels[4], param_channels[5]//2, param_channels[4])
        
        self.Upsample_4 = DeconvUpsample(in_channels=param_channels[4],out_channels=param_channels[3])
        self.Upsample_3 = DeconvUpsample(in_channels=param_channels[3],out_channels=param_channels[2])
        self.Upsample_2 = DeconvUpsample(in_channels=param_channels[2],out_channels=param_channels[1])
        self.Upsample_1 = DeconvUpsample(in_channels=param_channels[1],out_channels=param_channels[0])
        
        self.fuse4 = Fuse(param_channels[3],param_channels[4])
        self.fuse3 = Fuse(param_channels[2],param_channels[3])
        self.fuse2 = Fuse(param_channels[1],param_channels[2])
        self.fuse1 = Fuse(param_channels[0],param_channels[1])

        
        self.decoder4 = TransformerLayer(param_channels[3],att=True)
        self.decoder3 = TransformerLayer(param_channels[2],att=True)
        self.decoder2 = TransformerLayer(param_channels[1],att=True)
        self.decoder1 = TransformerLayer(param_channels[0],att=True)
        
        self.outconv = nn.Conv2d(param_channels[0],1,1)
        
        if self.deep_supervision:

            self.conv4 = nn.Conv2d(param_channels[3], 1, 1)
            self.conv3 = nn.Conv2d(param_channels[2], 1, 1)
            self.conv2 = nn.Conv2d(param_channels[1], 1, 1)
            self.conv1 = nn.Conv2d(param_channels[0], 1, 1)
            self.conv0 = nn.Conv2d(4, 1, 1)
        
        
        
    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []        
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num-1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)


    def forward(self,x):
        x1 = self.down_encoder1(self.relu(self.bn((self.inconv(x)))))
        x2 = self.down_encoder2(self.pool(x1))
        x3 = self.down_encoder3(self.pool(x2))
        x4 = self.down_encoder4(self.pool(x3))
        x5 = self.mid_layer(self.pool(x4))
        
        x_4up = self.decoder4(self.Upsample_4(x5))
        x4_fuse = self.fuse4(x_4up,x4)
        x3_up = self.decoder3(self.Upsample_3(x4_fuse))
        x3_fuse = self.fuse3(x3_up,x3)
        x2_up = self.decoder2(self.Upsample_2(x3_fuse))
        x2_fuse = self.fuse2(x2_up,x2)
        x1_up = self.decoder1(self.Upsample_1(x2_fuse))
        x1_fuse = self.fuse1(x1_up,x1)
        
        output = self.outconv(x1_fuse)
        
        if self.deep_supervision:
            gt_4 = Up_sample(self.conv4(x4_fuse),8)
            gt_3 = Up_sample(self.conv3(x3_fuse),4)
            gt_2 = Up_sample(self.conv2(x2_fuse),2)
            gt_1 = self.conv1(x1_fuse)
            
            out = self.conv0(torch.cat([gt_4,gt_3,gt_2, gt_1],dim=1))

            if self.mode == 'train':
                return (torch.sigmoid(gt_4), torch.sigmoid(gt_3), torch.sigmoid(gt_2), torch.sigmoid(gt_1),torch.sigmoid(output), torch.sigmoid(out))
            else:
                return torch.sigmoid(output)
        
        
        else:
            return torch.sigmoid(output)

        
        
if __name__ == '__main__':
    size = 256
    data = torch.arange(0, size * size * 2 * 3)
    data = data / (size * size * 2 * 3)
    data = data.reshape(2, 3, int(size), int(size))
    device = torch.device("cpu")
    data = data.to(device)
    model = SFDTNet(in_channels=3,deep_supervision=False).to(device)
    print('# model_restoration parameters: %.2f M' % (sum(param.numel() for param in model.parameters()) / 1e6))
    # torch.cuda.synchronize()
    # start = time.time()
    out = model(data)
    # torch.cuda.synchronize()
    # end = time.time()
    print((out))
    print(out.shape)
    flops, params = profile(model, (data,))

    print("-" * 50)
    print('FLOPs = ' + str(flops / 1000 ** 3) + ' G')
    print('Params = ' + str(params / 1000 ** 2) + ' M')
    
    # torch.cuda.synchronize()
    # end = time.time()
    # print('infer_time:', end-start)
        
        
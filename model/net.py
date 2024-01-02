import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from model.vit import ViT
from model.CFI import *
from model.CFR import *

class MViTLD(nn.Module):
    def __init__(self, num_classes, input_channels, block, num_blocks, nb_filter):
        super(MViTLD, self).__init__()
        self.stem = nn.Sequential(
            nn.BatchNorm2d(3,affine=False),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2) 
        self.pool4 = nn.MaxPool2d(4, 4) 
        self.pool8 = nn.MaxPool2d(8, 8)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True) 
        self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True) 

        self.sigmoid = nn.Sigmoid()
        self.nolocal256 = NonLocalBlock(256)
        

        self.cfi4 = Cross_Feature_Interaction(in_high_channels=nb_filter[4],in_low_channels=nb_filter[3],out_channels=nb_filter[3])
        self.cfi3 = Cross_Feature_Interaction(in_high_channels=nb_filter[3],in_low_channels=nb_filter[2],out_channels=nb_filter[2])
        self.cfi2 = Cross_Feature_Interaction(in_high_channels=nb_filter[2],in_low_channels=nb_filter[1],out_channels=nb_filter[1])
        self.cfi1 = Cross_Feature_Interaction(in_high_channels=nb_filter[1],in_low_channels=nb_filter[0],out_channels=nb_filter[0])

        self.conv256 = self._make_layer(block, nb_filter[4],   nb_filter[3], 2)
        self.conv128 = self._make_layer(block, nb_filter[3],   nb_filter[2], 2)
        
        self.convout = self._make_layer(block, nb_filter[3],   nb_filter[4], 2)
        self.conv64 = self._make_layer(block, nb_filter[2],   nb_filter[1], 2)
        self.conv32 = self._make_layer(block, nb_filter[1],   nb_filter[0], 2)
        
        self.conv0_0 = self._make_layer(block, input_channels, nb_filter[0])
  
        self.conv1_0 = self._make_layer(block, 16,   nb_filter[1], num_blocks[0])

        self.conv2_0 = self._make_layer(block,  nb_filter[1],   nb_filter[2], num_blocks[1])

        self.conv3_0 = self._make_layer(block, nb_filter[2],   nb_filter[3], num_blocks[2])

        self.conv4_0 = self._make_layer(block, nb_filter[3],   nb_filter[4], num_blocks[3])

        self.conv5 = self._make_layer(block, 16+nb_filter[1] + nb_filter[2] + nb_filter[3]+ nb_filter[4],   nb_filter[4])

        self.vit5 = ViT(img_dim=256, in_channels=nb_filter[0], embedding_dim=nb_filter[4], head_num=2, mlp_dim=32 * 32,
                        block_num=1, patch_dim=8, classification=False, num_classes=1)

        self.vit4 = ViT(img_dim=128, in_channels=nb_filter[1], embedding_dim=nb_filter[4], head_num=2, mlp_dim=32 * 32,
                        block_num=1, patch_dim=4, classification=False, num_classes=1)
        self.vit3 = ViT(img_dim=64, in_channels=nb_filter[2], embedding_dim=nb_filter[4], head_num=2, mlp_dim=32 * 32,
                        block_num=1, patch_dim=2, classification=False, num_classes=1)
        self.vit2 = ViT(img_dim=32, in_channels=nb_filter[3], embedding_dim=nb_filter[4],head_num=2, mlp_dim=32 * 32,
                        block_num=1, patch_dim=1, classification=False,num_classes=1)
        
        self.conv3_1_1 = self._make_layer(block, nb_filter[0] + nb_filter[1] + nb_filter[2] + nb_filter[3] + nb_filter[4],
                                        nb_filter[4])
        self.conv3_1 = self._make_layer(block, nb_filter[3] + nb_filter[4], nb_filter[3])

        self.conv2_2 = self._make_layer(block, nb_filter[2] + nb_filter[3], nb_filter[2])
  
        self.conv1_3 = self._make_layer(block, nb_filter[1] + nb_filter[2], nb_filter[1])
   
        self.conv0_4 = self._make_layer(block, nb_filter[0] + nb_filter[1], nb_filter[0])

        self.final3 = nn.Conv2d(nb_filter[3], num_classes, kernel_size=3,padding=1)

        self.final2 = nn.Conv2d(nb_filter[2], num_classes, kernel_size=3,padding=1)

        self.final1 = nn.Conv2d(nb_filter[1], num_classes, kernel_size=3,padding=1)

        self.final0 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=3,padding=1)

        self.finalfuse = nn.Conv2d(5,num_classes,kernel_size=1)

        self.cfr = CFR(1,64)

    def _make_layer(self, block, input_channels, output_channels, num_blocks=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(num_blocks-1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
    
        input = self.stem(input)  

        x0_0 = self.conv0_0(input) 

        x1_0 = self.conv1_0(self.pool(x0_0))
       
        x2_0 = self.conv2_0(self.pool(x1_0)) 
      
        x3_0 = self.conv3_0(self.pool(x2_0)) 

        out = self.conv4_0(x3_0) 
        
        outtemp = torch.cat([self.pool8(x0_0),self.pool4(x1_0),self.pool(x2_0),x3_0,out],1)

        out = self.conv5(outtemp)

        v2 = rearrange(self.vit2(x3_0), "b (x y) c -> b c x y", x=32, y=32) 
       
        v3 = rearrange(self.vit3(x2_0), "b (x y) c -> b c x y", x=32, y=32) 
        
        v4 = rearrange(self.vit4(x1_0), "b (x y) c -> b c x y", x=32, y=32)
       
        v5 = rearrange(self.vit5(x0_0), "b (x y) c -> b c x y", x=32, y=32)
        
        out =  v2 + v3 + v4 +v5+ out 

        out = self.nolocal256(out)

        fuse41, fuse42 = self.cfi4(out,x3_0)
        out4 =  self.conv256(torch.cat([fuse41,fuse42],1))

        fuse31, fuse32 = self.cfi3(self.up(out4),x2_0)
        out3 =  self.conv128(torch.cat([fuse31,fuse32],1))  
        fuse21, fuse22 = self.cfi2(self.up(out3),x1_0)
        out2 =  self.conv64(torch.cat([fuse21,fuse22],1))  
        fuse11, fuse12 = self.cfi1(self.up(out2),x0_0)
        
        out1 =  self.conv32(torch.cat([fuse11,fuse12],1)) 
        
        out4 = self.final3(out4)
        out3 = self.final2(out3)
        out2 = self.final1(out2)
        out1 = self.final0(out1)

        out4up = _upsample_like(out4, out1)
        out3up = _upsample_like(out3,out1)
        out2up = _upsample_like(out2,out1)

        out5 = self.cfr(out1)
        out6 = self.finalfuse(torch.cat([out5,out4up,out3up,out2up,out1],1))   

        return [out6,out5,out4up,out3up,out2up,out1]
        # return [F.sigmoid(out6),F.sigmoid(out5),F.sigmoid(out4up),F.sigmoid(out3up),F.sigmoid(out2up),F.sigmoid(out1)]

def _upsample_like(src,tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)
    return src 

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
class Res_CBAM_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super(Res_CBAM_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace = True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
    def forward(self, x):
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
        # out = self.gam(out)
        out += residual
        out = self.relu(out)
        return out

if __name__ == '__main__':
    resUnet = MViTLD(num_classes=1, input_channels=3, block=Res_CBAM_block, num_blocks=[2,2,2,2],nb_filter=[16,32,64,128,256])
    print(sum(p.numel() for p in resUnet.parameters()))
    print(resUnet(torch.rand(2, 3, 256, 256)).shape)

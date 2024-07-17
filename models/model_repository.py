from torch import nn
import torch
from torch.nn import functional as F
from .resnet import resnet18, resnet50, resnet34



class Resnet18_8s(nn.Module):
    def __init__(self, fcdim=256, s8dim=128, s4dim=64, s2dim=32, raw_dim=32):
        super(Resnet18_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                               pretrained=True,
                               output_stride=32,
                               remove_avg_pool_layer=True)


        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        self.up32sto16s=nn.UpsamplingBilinear2d(scale_factor=2)
        # x16s->256
        self.conv16s=nn.Sequential(
            nn.Conv2d(fcdim+256, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.LeakyReLU(0.1,True)
        )
        self.up16sto8s=nn.UpsamplingBilinear2d(scale_factor=2)
        #128
        self.conv16s_uv=nn.Sequential(
             nn.Conv2d(fcdim, fcdim, 3, 1, 1, bias=False),
             nn.BatchNorm2d(fcdim),
             nn.LeakyReLU(0.1,True)
        )
        self.up16sto8s_uv=nn.UpsamplingBilinear2d(scale_factor=2)

        # x8s->128
        self.conv8s=nn.Sequential(
            nn.Conv2d(128+fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up8sto4s=nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv8s_uv=nn.Sequential(
            nn.Conv2d(fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up8sto4s_uv=nn.UpsamplingBilinear2d(scale_factor=2)

        # x4s->64
        self.conv4s=nn.Sequential(
            nn.Conv2d(64+s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up4sto2s=nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv4s_uv=nn.Sequential(
            nn.Conv2d(s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up4sto2s_uv=nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s->64
        self.conv2s=nn.Sequential(
            nn.Conv2d(64+s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )
        self.up2storaw = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2s_uv=nn.Sequential(
            nn.Conv2d(s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1,True)
        )  
        self.up2storaw_uv = nn.UpsamplingBilinear2d(scale_factor=2)

        self.final = nn.Sequential(
            nn.Conv2d(3+s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True)
        )
        self.final_uv = nn.Sequential(
            nn.Conv2d(s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1,True)
        )

        self.uptorawsize=nn.UpsamplingBilinear2d(scale_factor=2)
        self.seg=nn.Conv2d(raw_dim, 2, 1, 1)
        self.uv=nn.Conv2d(raw_dim, 3, 1, 1)
        self.drop = nn.Dropout2d(p=0.15)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)


        x16=self.up32sto16s(xfc)  #256, 16s
        

        fm=self.conv16s(torch.cat([x16,x16s],1))
        fm=self.up16sto8s(fm) 
        #pdb.set_trace()
        fm=self.conv8s(torch.cat([fm,x8s],1))
        fm=self.up8sto4s(fm)
        fm=self.drop(fm)


        fm=self.conv4s(torch.cat([fm,x4s],1))
        fm=self.up4sto2s(fm)
        fm=self.drop(fm)

        fm=self.conv2s(torch.cat([fm,x2s],1))
        fm=self.up2storaw(fm)


        x=self.final(torch.cat([fm,x],1))
        # fm=self.uptorawsize(fm)
        # x=self.drop(fm)

        #uv
        fm=self.conv16s_uv(x16)  # 128
        fm=self.up16sto8s_uv(fm)

        fm=self.conv8s_uv(fm)
        fm=self.up8sto4s_uv(fm)
        fm=self.drop(fm)
                                                
        fm=self.conv4s_uv(fm)
        fm=self.up4sto2s_uv(fm)
        fm=self.drop(fm)
                                                
        fm=self.conv2s_uv(fm)
        fm=self.up2storaw_uv(fm)
                                                
        f_uv=self.final_uv(fm)
        uv=self.uv(f_uv)

        return x, self.seg(x), f_uv, torch.tanh(uv)



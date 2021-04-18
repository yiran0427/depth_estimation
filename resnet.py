import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # encoder
        self.conv1 = self.conv_block(in_dim=3, out_dim=32, kernel=7)
        self.conv2 = self.conv_block(in_dim=32, out_dim=64, kernel=5)
        self.conv3 = self.conv_block(in_dim=64, out_dim=128, kernel=3)
        self.conv4 = self.conv_block(in_dim=128, out_dim=256, kernel=3)
        self.conv5 = self.conv_block(in_dim=256, out_dim=512, kernel=3)
        self.conv6 = self.conv_block(in_dim=512, out_dim=512, kernel=3)
        self.conv7 = self.conv_block(in_dim=512, out_dim=512, kernel=3)
        
        # decoder
        self.upconv7 = self.upconv_block(in_dim=512, out_dim=512)
        self.iconv7 = self.iconv_block(in_dim=1024, out_dim=512)

        self.upconv6 = self.upconv_block(in_dim=512, out_dim=512)
        self.iconv6 = self.iconv_block(in_dim=1024, out_dim=512)
        
        self.upconv5 = self.upconv_block(in_dim=512, out_dim=256)
        self.iconv5 = self.iconv_block(in_dim=512, out_dim=256)

        self.upconv4 = self.upconv_block(in_dim=256, out_dim=128)
        self.iconv4 = self.iconv_block(in_dim=256, out_dim=128)
        self.disp4_layer = self.disp_block(in_dim=128)

        self.upconv3 = self.upconv_block(in_dim=128, out_dim=64)
        self.iconv3 = self.iconv_block(in_dim=130, out_dim=64)
        self.disp3_layer = self.disp_block(in_dim=64)

        self.upconv2 = self.upconv_block(in_dim=64, out_dim=32)
        self.iconv2 = self.iconv_block(in_dim=66, out_dim=32)
        self.disp2_layer = self.disp_block(in_dim=32)

        self.upconv1 = self.upconv_block(in_dim=32, out_dim=16)
        self.iconv1 = self.iconv_block(in_dim=18, out_dim=16)
        self.disp1_layer = self.disp_block(in_dim=16)
        
    def conv_block(self, in_dim, out_dim, kernel):
        layers = []
        layers += [
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel, stride=2),
            nn.BatchNorm2d(out_dim),
            nn.ELU(),
        ]
        layers += [
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ELU(),
        ]
        return nn.Sequential(*layers)
    
    def upconv_block(self, in_dim, out_dim):
        layers = []
        layers += [
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ELU(),
        ]
        return nn.Sequential(*layers)

    def iconv_block(self, in_dim, out_dim):
        layers = []
        layers += [
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1),
            nn.BatchNorm2d(out_dim),
            nn.ELU(),
        ]
        return nn.Sequential(*layers)
    
    def disp_block(self, in_dim):
        layers = []
        layers += [
            nn.Conv2d(in_dim, 2, kernel_size=3, stride=1),
            nn.Sigmoid(),
        ]
        return nn.Sequential(*layers)
        

    def forward(self, x):
        # encoder
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        
        # decoder
        upconv7 = self.upconv7(F.interpolate(conv7, scale_factor=2, mode='bilinear', align_corners=True))
        concat7 = torch.cat([upconv7, conv6], 1)
        iconv7 = self.iconv7(concat7)
        
        upconv6 = self.upconv6(F.interpolate(iconv7, scale_factor=2, mode='bilinear', align_corners=True))
        concat6 = torch.cat([upconv6, conv5], 1)
        iconv6 = self.iconv7(concat6)
        
        upconv5 = self.upconv5(F.interpolate(iconv6, scale_factor=2, mode='bilinear', align_corners=True))
        concat5 = torch.cat([upconv5, conv4], 1)
        iconv5 = self.iconv7(concat5)
        
        upconv4 = self.upconv4(F.interpolate(iconv5, scale_factor=2, mode='bilinear', align_corners=True))
        concat4 = torch.cat([upconv4, conv3], 1)
        iconv4 = self.iconv4(concat4)
        self.disp4 = self.disp4_layer(iconv4)
        udisp4 = F.interpolate(self.disp4, scale_factor=2, mode='bilinear', align_corners=True)
        
        upconv3 = self.upconv3(F.interpolate(iconv4, scale_factor=2, mode='bilinear', align_corners=True))
        concat3 = torch.cat([upconv3, conv2, udisp4], 1)
        iconv3 = self.iconv3(concat3)
        self.disp3 = self.disp3_layer(iconv3)
        
        upconv2 = self.upconv2(F.interpolate(iconv3, scale_factor=2, mode='bilinear', align_corners=True))
        concat2 = torch.cat([upconv2, conv1, udisp3], 1)
        iconv2 = self.iconv2(concat2)
        self.disp2 = self.disp2_layer(iconv2)
        
        upconv1 = self.upconv1(F.interpolate(iconv2, scale_factor=2, mode='bilinear', align_corners=True))
        concat1 = torch.cat([upconv1, udisp2], 1)
        iconv1 = self.iconv1(concat1)
        self.disp1 = self.disp1_layer(iconv1)

        return self.disp1, self.disp2, self.disp3, self.disp4
 
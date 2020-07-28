import torch
import torch.nn as nn
import torch.nn.functional as F
import neural_renderer as nr


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop = 0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.conv1_drop = nn.Dropout2d(drop)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN2 = nn.BatchNorm2d(out_channels)

    def forward(self, x_in):
        x = self.conv1_drop(self.conv1(x_in))
        x = F.relu(self.BN1(x))
        x = self.conv2_drop(self.conv2(x))
        x = F.relu(self.BN2(x))
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, drop=0):
        super(DownBlock, self).__init__()
        P = int((kernel_size-1)/2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.pool = nn.MaxPool2d((2, 2))
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN = nn.BatchNorm2d(out_channels)

    def forward(self, x_in):
        x1 = self.conv2_drop(self.conv2(self.conv1_drop(self.conv1(x_in))))
        x1_pool = F.relu(self.BN(self.pool(x1)))
        return x1, x1_pool


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, act=None, drop=0):
        super(UpBlock, self).__init__()
        P = int((kernel_size-1)/2)
        self.Upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv1_drop = nn.Dropout2d(drop)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=P)
        self.conv2_drop = nn.Dropout2d(drop)
        self.BN = nn.BatchNorm2d(out_channels)
        self.act = act
    
    def forward(self, x_in, x_up, act):
        x = self.Upsample(x_in)
        x_cat = torch.cat((x,x_up),1)
        x1 = self.conv2_drop(self.conv2(self.conv1_drop(self.conv1(x_cat))))
        if act == 'tanh': 
            return F.tanh(self.BN(x1))
        elif act == 'relu':
            return F.relu(self.BN(x1))

        
class Encoder(nn.Module):
    def __init__(self,AEdim, drop=0):
        super(Encoder, self).__init__()
        a, b, c = int(AEdim/4), int(AEdim/2), AEdim
        self.down1 = DownBlock(3, a, drop=drop)
        self.down2 = DownBlock(a, b, drop=drop)
        self.down3 = DownBlock(b, c, drop=drop)

    def forward(self, x_in):
        x1, x1_pool = self.down1(x_in)
        x2, x2_pool = self.down2(x1_pool)
        x3, x3_pool = self.down3(x2_pool)
        return x1, x2, x3, x3_pool

        
class Decoder(nn.Module):
    def __init__(self, AEdim, drop=0):
        super(Decoder, self).__init__()
        a, b, c = int(AEdim/4), int(AEdim/2), AEdim
        self.up3 = UpBlock(2*c, b, act='relu', drop = drop)
        self.up2 = UpBlock(c, a, act='relu', drop=drop)
        self.up1 = UpBlock(b, 2, act='tanh', drop=drop)

    def forward(self, x1, x2, x3, x_res):
        x4_pool = self.up3(x_res, x3, act='relu')
        x5_pool = self.up2(x4_pool, x2, act='relu')
        x6_pool = self.up1(x5_pool, x1, act='tanh')
        Ix = x6_pool[:, 0, :, :].unsqueeze(dim=1)
        Iy = x6_pool[:, 1, :, :].unsqueeze(dim=1)
        return Ix, Iy
     

class DeepACM(nn.Module):
    def __init__(self, AEdim=512, nP=16, image_size=128, drop=0):
        super(DeepACM, self).__init__()
        self.Ea = Encoder(AEdim, drop=drop)
        self.Dxy = Decoder(AEdim, drop=drop)
        self.res = ResidualBlock(AEdim, AEdim, drop=drop)
        self.nP = nP
        self.texture_size = 2
        self.camera_distance = 1
        self.elevation = 0
        self.azimuth = 0
        self.image_size = image_size
        self.renderer = nr.Renderer(camera_mode='look_at', image_size=image_size, light_intensity_ambient=1,
                                    light_intensity_directional=1, perspective=False)

    def forward(self, I, P, faces):
        x1, x2, x3, x3_res = self.Ea(I)
        x3_res = self.res(x3_res)
        Ix, Iy = self.Dxy(x1, x2, x3, x3_res)
        Pxx = F.grid_sample(Ix, P).transpose(3, 2)
        Pyy = F.grid_sample(Iy, P).transpose(3, 2)
        Pedge = torch.cat((Pxx, Pyy), -1)
        PP = Pedge + P
        z = torch.ones((PP.shape[0], 1, PP.shape[2], 1)).cuda()
        PP = torch.cat((PP, z), 3)
        PP = torch.squeeze(PP, dim=1)
        PP[:, :, 1] = PP[:, :, 1]*-1
        faces = torch.squeeze(faces, dim=1)
        self.renderer.eye = nr.get_points_from_angles(self.camera_distance, self.elevation, self.azimuth)
        mask = self.renderer(PP, faces, mode='silhouettes').unsqueeze(dim=1)
        PP[:, :, 1] = PP[:, :, 1]*-1
        return mask, PP[:, :, 0:2].unsqueeze(dim=1), Ix, Iy















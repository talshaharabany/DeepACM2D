from models.vanilla import *
import neural_renderer as nr
from models.hardnet import *


class final_layer(nn.Module):
    def __init__(self, full_features):
        super(final_layer, self).__init__()
        # self.conv1 = nn.Conv2d(full_features[0], full_features[0], 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(full_features[0], 1, 1, stride=1, padding=0)
        # self.BN1 = nn.BatchNorm2d(full_features[0])
        # self.BN2 = nn.BatchNorm2d(1)

    def forward(self, z, size):
        return F.interpolate(z, size=size, mode='bilinear', align_corners=True)
        # z = F.relu(self.BN1(self.conv1(z)))
        # return F.sigmoid(self.BN2(self.conv2(z)))


class Decoder(nn.Module):
    def __init__(self, full_features, args):
        super(Decoder, self).__init__()
        if int(args['outlayer']) == 2:
            self.up1 = UpBlock(full_features[1] + full_features[0], 1,
                               func='sigmoid', drop=float(args['drop'])).cuda()
        if int(args['outlayer']) == 3:
            self.up1 = UpBlock(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=float(args['drop'])).cuda()
            self.up2 = UpBlock(full_features[1] + full_features[0], 1,
                               func='sigmoid', drop=float(args['drop'])).cuda()
        if int(args['outlayer']) == 4:
            self.up1 = UpBlock(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=float(args['drop'])).cuda()
            self.up2 = UpBlock(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=float(args['drop'])).cuda()
            self.up3 = UpBlock(full_features[1] + full_features[0], 1,
                               func='sigmoid', drop=float(args['drop'])).cuda()
        self.args = args
        self.final = final_layer(full_features)

    def forward(self, x, size):
        if int(self.args['outlayer']) == 2:
            z = self.up1(x[1], x[0])
        if int(self.args['outlayer']) == 3:
            z = self.up1(x[2], x[1])
            z = self.up2(z, x[0])
        if int(self.args['outlayer']) == 4:
            z = self.up1(x[3], x[2])
            z = self.up2(z, x[1])
            z = self.up3(z, x[0])
        return self.final(z, size)


class Segmentation(nn.Module):
    def __init__(self, args):
        super(Segmentation, self).__init__()
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        self.decoder = Decoder(self.backbone.full_features, args)

    def forward(self, I):
        size = I.size()[2:]
        z = self.backbone(I)
        return self.decoder(z, size)















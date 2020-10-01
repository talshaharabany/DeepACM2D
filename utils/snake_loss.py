import torch


class Snakeloss:
    def __init__(self, criterion):
        self.criterion = criterion

    def ballon_loss(self, mask):
        return 1 - torch.mean(mask, dim=[2, 3]).cuda().mean()

    def avg_dis(self, P):
        nP = P.shape[2]
        even1 = P[:, :, 0:nP:2, :]
        odd1 = P[:, :, 1:nP:2, :]
        diff1 = torch.sum((even1-odd1)**2, dim=[2, 3])
        even2 = P[:, :, 2:nP:2, :]
        odd2 = P[:, :, 1:nP-1:2, :]
        diff2 = torch.sum((even2-odd2)**2, dim=[2, 3])
        diff3 = torch.sum((P[:, :, 0, :] - P[:, :, nP-1, :])**2, dim=[1, 2])
        return (1/3)*diff1**0.5+(1/3)*diff2**0.5+(1/3)*diff3**0.5

    def curvature_loss(self, P):
        Pf = P.roll(-1, dims=2)
        Pb = P.roll(1, dims=2)
        K = Pf + Pb - 2 * P
        return K.abs().mean()

    def snake_loss(self, num_of_it, net_out, gt):
        for it in range(num_of_it):
            if it == 0:
                loss = self.criterion(net_out[0][it], gt) + \
                       0.1*self.avg_dis(net_out[1][it]).mean()
            else:
                loss = loss + \
                       self.criterion(net_out[0][it], gt) + \
                       0.1*self.avg_dis(net_out[1][it]).mean()
        return loss


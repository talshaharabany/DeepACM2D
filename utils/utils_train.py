import numpy as np
import torch


def ballon_loss(mask, criterion):
    return criterion(torch.mean(mask, dim=[2, 3]), torch.ones(mask.shape[0], 1).to('cuda:0'))


def avg_dis(P):
    nP = P.shape[2]
    even1 = P[:, :, 0:nP:2, :]
    odd1  = P[:, :, 1:nP:2, :]
    diff1 = torch.sum((even1-odd1)**2, dim=[2, 3])
    even2 = P[:, :, 2:nP:2, :]
    odd2  = P[:, :, 1:nP-1:2, :]
    diff2 = torch.sum((even2-odd2)**2, dim=[2, 3])
    diff3 = torch.sum((P[:, :, 0, :] - P[:, :, nP-1, :])**2, dim=[1, 2])
    return (1/3)*diff1**0.5+(1/3)*diff2**0.5+(1/3)*diff3**0.5


def curvature_loss(P):
    Pf = P.roll(-1, dims=2)
    Pb = P.roll(1, dims=2)
    K = Pf + Pb - 2 * P
    return K.abs().mean()


def train(ds, model, optimizer, criterion, PTrain, faces, epoch, args):
    Maskloss_list = []
    BLoss_list = []
    NNLoss_list = []
    for ix, (_x, _y) in enumerate(ds):
        _x = _x.float().cuda()
        _p = PTrain.float().cuda().clone()
        _y = _y.float().cuda()
        optimizer.zero_grad()
        iter = int(args['DeepIt'])
        for it in range(iter):
            net_out = model(_x, _p, faces)
            _p = net_out[1]
            if it == 0:
                MaskLoss = criterion(net_out[0], _y)
                NNLoss = torch.sum((net_out[1] - net_out[1].roll(1, 2)) ** 2, dim=[2, 3]).mean()
                BLoss = ballon_loss(net_out[0], criterion)
            else:
                MaskLoss = MaskLoss + criterion(net_out[0], _y)
                NNLoss = NNLoss + torch.sum((net_out[1] - net_out[1].roll(1, 2)) ** 2, dim=[2, 3]).mean()
                BLoss = BLoss + ballon_loss(net_out[0], criterion)

        loss = (float(args['wM'])*MaskLoss +\
                float(args['wB'])*BLoss +\
                float(args['wNN'])*NNLoss)/iter

        Maskloss_list.append(MaskLoss.item())
        BLoss_list.append(BLoss.item())
        NNLoss_list.append(NNLoss.item())
        loss.backward()
        optimizer.step()
        return Maskloss_list, BLoss_list, NNLoss_list


def logger_train(epoch, writer, Maskloss_list, BLoss_list, NNLoss_list):
    print('************************************************************************')
    print('Epoch: ' + str(epoch) + ' Mask mean loss: ' + str(np.mean(Maskloss_list)) + ' Mask max loss: ' + str(
        np.max(Maskloss_list)) + ' Mask min loss: ' + str(np.min(Maskloss_list)))
    writer.add_scalar('MaskLoss', np.mean(Maskloss_list), global_step=epoch)
    print('Epoch: ' + str(epoch) + ' B mean  loss: ' + str(np.mean(BLoss_list)) + ' B max loss: ' + str(
        np.max(BLoss_list)) + ' B min loss: ' + str(np.min(BLoss_list)))
    writer.add_scalar('BallonLoss', np.mean(BLoss_list), global_step=epoch)
    print('Epoch: ' + str(epoch) + ' NN mean  loss: ' + str(np.mean(NNLoss_list)) + ' NN max loss: ' + str(
        np.max(NNLoss_list)) + ' NN min loss: ' + str(np.min(NNLoss_list)))
    writer.add_scalar('NNLoss', np.mean(NNLoss_list), global_step=epoch)
    print('************************************************************************')



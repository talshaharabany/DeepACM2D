import numpy as np
import torch
from tqdm import tqdm


def norm_tensor(_x):
    bs = _x.shape[0]
    min_x = _x.view(bs, 3, -1).min(dim=2)[0].view(bs, 3, 1, 1)
    max_x = _x.view(bs, 3, -1).max(dim=2)[0].view(bs, 3, 1, 1)
    return (_x - min_x) / (max_x - min_x + 1e-2)


def norm_input(_x, seg_out, a):
    _x = norm_tensor(_x)
    _x = a * _x + (1 - a) * seg_out
    return _x


def train(ds, model, segnet, optimizer, snake_loss, PTrain, faces, args):
    loss_list = []
    for ix, (_x, _y) in tqdm(enumerate(ds)):
        _x = _x.float().cuda()
        _p = PTrain.float().cuda().clone()
        _y = _y.float().cuda()
        optimizer.zero_grad()
        seg_out = segnet(_x)
        _x = norm_input(_x, seg_out, float(args['a'])).detach()
        num_of_it = int(args['DeepIt'])
        net_out = model(_x, _p, faces, num_of_it)
        loss = snake_loss.snake_loss(num_of_it, net_out, _y)
        loss_list.append(loss.item()/num_of_it)
        loss.backward()
        optimizer.step()
    return loss_list


def logger_train(epoch, writer, loss_list):
    print('************************************************************************')
    print('Epoch: ' + str(epoch) + ' Mask mean loss: ' + str(np.mean(loss_list)) + ' Mask max loss: ' + str(
        np.max(loss_list)) + ' Mask min loss: ' + str(np.min(loss_list)))
    writer.add_scalar('MaskLoss', np.mean(loss_list), global_step=epoch)
    print('************************************************************************')



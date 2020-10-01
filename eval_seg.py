import torch
import numpy as np
from loader.viah_loader import *
from loader.bing_loader import *
from utils.utils_args import *
from utils.utils_eval import *
from utils.utils_train import *
from utils.utils_tri import *
from utils.utils_vis import *
from utils.loss import *
from models.model_seg import *


def eval_ds(ds, model):
    TestDice_list = []
    TestIoU_list = []
    for ix, (_x, _y) in enumerate(ds):
        _x = _x.float().cpu()
        _y = _y.float().cpu()
        Mask = model(_x)
        Mask[Mask >= 0.5] = 1
        Mask[Mask < 0.5] = 0
        (cDice, cIoU) = get_dice_ji(Mask, _y)
        TestDice_list.append(cDice)
        TestIoU_list.append(cIoU)
    Dice = np.mean(TestDice_list)
    IoU = np.mean(TestIoU_list)
    print((Dice, IoU))


def main():
    torch.backends.cudnn.benchmark = True
    args = get_args()
    save_args(args)

    if args['task'] == 'viah':
        PATH = r'results/viah/best/'
        testset = viah_segmentation(ann='test', args=args)
    elif args['task'] == 'bing':
        testset = bing_segmentation(ann='test', args=args)
        PATH = r'results/bing/best/'
    segnet = Segmentation(args)
    segnet1 = torch.load(PATH + 'SEG.pt')
    segnet.load_state_dict(segnet1.state_dict())
    segnet.cpu().eval()
    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=1, drop_last=False)
    eval_ds(ds_val, segnet)

if __name__ == '__main__':
    main()


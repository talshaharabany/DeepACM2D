from torch.utils.data import Dataset
from tensorboardX import SummaryWriter

from models.model import *
from models.model_seg import Segmentation
from loader.viah_loader import *
from loader.bing_loader import *
from utils.utils_args import *
from utils.utils_eval import get_dice_ji, vis_ds
from utils.utils_train import *
from utils.utils_tri import *
from utils.loss import *
from utils.snake_loss import Snakeloss
import random


def eval_ds(ds, model, segnet, PTrain, faces, args):
    model.eval()
    IoU_list = []
    Dice_list = []
    model.eval()
    with torch.no_grad():
        for ix, (_x, _y) in enumerate(ds):
            _x = _x.float().cuda()
            _p = PTrain.float().cuda().clone()
            _y = _y.float().cuda()
            seg_out = segnet(_x).detach()
            _x = norm_input(_x, seg_out, float(args['a']))
            iter = int(args['DeepIt'])
            net_out = model(_x, _p, faces, iter)
            Mask = net_out[0][iter-1]
            cDice, cIoU = get_dice_ji(Mask, _y)
            IoU_list.append(cIoU)
            Dice_list.append(cDice)
        IoU = np.mean(IoU_list)
        Dice = np.mean(Dice_list)
        model.train()
        return Dice, IoU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
args = get_args()

segnet = Segmentation(args)
model = DeepACM(args)


P_test, faces_test = get_poly(int(args['im_size']), int(args['nP']),
                              int(args['Radius']), int(args['im_size']) / 2, int(args['im_size']) / 2)
faces_test = faces_test.unsqueeze(dim=0).unsqueeze(dim=0).cuda()
faces = faces_test.repeat(1, 1, 1, 1).cuda()
PTrain = P_test.repeat(1, 1, 1, 1).cuda()
if args['task'] == 'viah':
    PATH = r'results/viah/best/'
    testset = viah_segmentation(ann='test', args=args)
elif args['task'] == 'bing':
    testset = bing_segmentation(ann='test', args=args)
    PATH = r'results/bing/best/'
ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                     num_workers=1, drop_last=False)

model1 = torch.load(PATH + 'ACM.pt')
model.load_state_dict(model1.state_dict())
model.eval().to(device)
segnet1 = torch.load(PATH + 'SEG.pt')
segnet.load_state_dict(segnet1.state_dict())
segnet.eval().to(device)
vis_ds(ds_val, model, segnet, PTrain, faces, args, num_of_ex=20)
dice, iou = eval_ds(ds_val, model, segnet, P_test, faces_test, args)
print((dice, iou))

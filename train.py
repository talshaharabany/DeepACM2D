from torch.utils.data import Dataset
from tensorboardX import SummaryWriter

from models.model import *
from models.model_seg import Segmentation
from loader.viah_loader import *
from loader.bing_loader import *
from utils.utils_args import *
from utils.utils_eval import *
from utils.utils_train import *
from utils.utils_tri import *
from utils.utils_vis import *
from utils.utils_lr import *
from utils.loss import *
from utils.snake_loss import Snakeloss
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
args = get_args()
save_args(args)
writer = SummaryWriter()

PATH1 = r'results/gpu'+str(args['folder'])+'/'
segnet = Segmentation(args)
segnet1 = torch.load(r'results/mask/model.pt')
segnet.load_state_dict(segnet1.state_dict())
segnet.eval().to(device)

model = DeepACM(args)
model.train().to(device)

P_test, faces_test = get_poly(int(args['im_size']), int(args['nP']),
                              int(args['Radius']), int(args['im_size']) / 2, int(args['im_size']) / 2)
faces_test = faces_test.unsqueeze(dim=0).unsqueeze(dim=0).cuda()
faces = faces_test.repeat(int(args['Batch_size']), 1, 1, 1).cuda()
PTrain = P_test.repeat(int(args['Batch_size']), 1, 1, 1).cuda()
criterion = SoftDiceLoss()
snake_loss = Snakeloss(criterion)
if args['opt'] == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['learning_rate']), weight_decay=float(args['WD']))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args['D_rate']), gamma=0.3)
elif args['opt'] == 'sgd':
    wd_params, non_wd_params = [], []
    for name, param in model.named_parameters():
        if param.dim() == 1:
            non_wd_params.append(param)
        elif param.dim() == 2 or param.dim() == 4:
            wd_params.append(param)
    params_list = [
        {'params': wd_params, },
        {'params': non_wd_params, 'weight_decay': 0},
    ]
    warmup_iters = int(args['WU'])
    optimizer = torch.optim.SGD(params_list,
                                lr=float(args['learning_rate']),
                                weight_decay=float(args['WD']),
                                momentum=0.9)
    max_iter = int(args['WU2'])
    scheduler = WarmupPolyLrScheduler(optimizer,
                                      power=0.9,
                                      max_iter=max_iter,
                                      warmup_iter=warmup_iters,
                                      warmup_ratio=0.001,
                                      warmup='exp',
                                      last_epoch=-1)

if args['task'] == 'viah':
    trainset = viah_segmentation(ann='training', args=args)
    testset = viah_segmentation(ann='test', args=args)
elif args['task'] == 'bing':
    trainset = bing_segmentation(ann='training', is_aug=False, args=args)
    testset = bing_segmentation(ann='test', is_aug=False, args=args)

ds = torch.utils.data.DataLoader(trainset, batch_size=int(args['Batch_size']), shuffle=True,
                                 num_workers=int(args['nW']), drop_last=True)
ds_tri = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False,
                                     num_workers=1, drop_last=False)
ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                     num_workers=1, drop_last=False)
best = 0
best_tri = 1
unet_dir = r'/media/data1/talshah/DeepACM/results/gpu1/use'
PATH1_list = update_net_list(unet_dir)
segnet1 = torch.load(r'/media/data1/talshah/DeepACM/results/gpu1/use/model_241.pt')
segnet.load_state_dict(segnet1.state_dict())
segnet.eval().to(device)
max_iter = int(args['DeepIt'])
for epoch in range(1, int(args['epochs'])):
    args['DeepIt'] = int(epoch/20) + 1
    if args['DeepIt'] > max_iter:
        args['DeepIt'] = max_iter
    loss_list = train(ds, model, segnet, optimizer, snake_loss, PTrain, faces, args)
    logger_train(epoch, writer, loss_list)
    scheduler.step()
    if epoch % 10 == 0:
        CP = None
        best_CP = 0
        for PATH1 in PATH1_list:
            segnet1 = torch.load(unet_dir + '/' + PATH1)
            segnet.load_state_dict(segnet1.state_dict())
            segnet.eval().to(device)
            iou = eval_ds(ds_val, model, segnet, P_test, faces_test, args)
            if best_CP < iou:
                best_CP = iou
                print('best CP: ' + PATH1 + ' with ' + str(best_CP))
                CP = PATH1
        writer.add_scalar('IoU', best_CP, global_step=epoch)
        segnet1 = torch.load(unet_dir + '/' + CP)
        segnet.load_state_dict(segnet1.state_dict())
        segnet.eval().to(device)
        if best_CP > best:
            best = best_CP
            print('best: ' + str(epoch) + ' with ' + str(best))
            torch.save(model, unet_dir + '/' + 'best.pt')












from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm

from models.model_seg import *
from loader.viah_loader import *
from loader.bing_loader import *
from utils.utils_args import *
from utils.utils_eval import get_dice_ji
from utils.utils_lr import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def train(ds, model, optimizer, criterion, scheduler, args):
    loss_list = []
    for ix, (_x, _y) in tqdm(enumerate(ds)):
        _x = _x.float().cuda()
        _y = _y.float().cuda().unsqueeze(dim=1)
        optimizer.zero_grad()
        mask = model(_x)
        loss = criterion(mask, _y)
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        if args['opt'] == 'sgd':
            scheduler.step()
    return loss_list


def eval_ds(ds, model, writer, epoch, PATH1, best, label, args):
    model.eval()
    TestDice_list = []
    TestIoU_list = []
    model.eval()
    for ix, (_x, _y) in enumerate(ds):
        _x = _x.float().cuda()
        _y = _y.float().cuda()
        Mask = model(_x)
        Mask[Mask >= 0.5] = 1
        Mask[Mask < 0.5] = 0
        (cDice, cIoU) = get_dice_ji(Mask, _y)
        TestDice_list.append(cDice)
        TestIoU_list.append(cIoU)
    Dice = np.mean(TestDice_list)
    IoU = np.mean(TestIoU_list)
    print((epoch, Dice, IoU))
    if IoU > 0.89 and label=='test':
        torch.save(model, PATH1 + 'model_' + str(epoch) + '.pt')
        # print('best results: ' + str(IoU))
        # best = IoU
    writer.add_scalar('Dice_' + label, Dice, global_step=epoch)
    writer.add_scalar('IoU_' + label, IoU, global_step=epoch)
    model.train()
    return best, IoU


def main(args, writer):
    PATH1 = r'results/gpu'+str(args['folder'])+'/'
    if bool(args['is_load']):
        model = Segmentation(args)
        model1 = torch.load(PATH1 + 'net_' + args['CP'] + '.pt')
        model.load_state_dict(model1.state_dict())
    else:
        model = Segmentation(args)
    model.train().to(device)

    criterion = nn.BCELoss()
    if args['opt'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(args['learning_rate']),
                                     weight_decay=float(args['WD']))
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
        warmup_iters = 12
        optimizer = torch.optim.SGD(params_list,
                                    lr=float(args['learning_rate']),
                                    weight_decay=float(args['WD']),
                                    momentum=0.9)
        max_iter = int(args['D_rate'])
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
                                         num_workers=0, drop_last=False)
    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=0, drop_last=False)
    best = 0
    best_tri = 1
    for epoch in range(1, int(args['epochs'])):
        loss_list = train(ds, model, optimizer, criterion, scheduler, args)
        print('************************************************************************')
        print('Epoch: ' + str(epoch) + ' Mask mean loss: ' + str(np.mean(loss_list)) + ' Mask max loss: ' + str(
            np.max(loss_list)) + ' Mask min loss: ' + str(np.min(loss_list)))
        writer.add_scalar('MaskLoss', np.mean(loss_list), global_step=epoch)
        print('************************************************************************')
        if args['opt'] == 'adam':
            scheduler.step()

        if epoch % 3 == 1:
            eval_ds(ds_tri, model, writer, epoch, PATH1, best_tri, 'training', args)
            best, _ = eval_ds(ds_val, model, writer, epoch, PATH1, best, 'test', args)

if __name__ == '__main__':
    args = get_args()
    save_args(args)
    writer = SummaryWriter()
    main(args, writer)










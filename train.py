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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
args = get_args()
save_args(args)
writer = SummaryWriter()

PATH = r'results/' + args['task']
segnet = Segmentation(args)
segnet1 = torch.load(PATH + '/best/SEG.pt')
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
optimizer = torch.optim.Adam(model.parameters(), lr=float(args['learning_rate']), weight_decay=float(args['WD']))
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(args['D_rate']), gamma=0.3)
if args['task'] == 'viah':
    trainset = viah_segmentation(ann='training', args=args)
    testset = viah_segmentation(ann='test', args=args)
elif args['task'] == 'bing':
    trainset = bing_segmentation(ann='training', args=args)
    testset = bing_segmentation(ann='test', args=args)

ds = torch.utils.data.DataLoader(trainset, batch_size=int(args['Batch_size']), shuffle=True,
                                 num_workers=int(args['nW']), drop_last=True)
ds_tri = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False,
                                     num_workers=1, drop_last=False)
ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                     num_workers=1, drop_last=False)
best = 0
max_iter = int(args['DeepIt'])
for epoch in range(1, int(args['epochs'])):
    args['DeepIt'] = int(epoch/20) + 1
    if args['DeepIt'] > max_iter:
        args['DeepIt'] = max_iter
    loss_list = train(ds, model, segnet, optimizer, snake_loss, PTrain, faces, args)
    logger_train(epoch, writer, loss_list)
    scheduler.step()
    if epoch % 5 == 2:
        iou = eval_ds(ds_val, model, segnet, P_test, faces_test, args)
        writer.add_scalar('IoU', iou, global_step=epoch)
        if iou > best:
            best = iou
            print('best: ' + str(epoch) + ' with ' + str(best))
            torch.save(model, PATH + '/' + 'ACM_best.pt')












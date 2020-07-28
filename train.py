from torch.utils.data import Dataset
from tensorboardX import SummaryWriter

from models.model import *
from loader.viah_loader import *
from loader.bing_loader import *
from utils.utils_args import *
from utils.utils_eval import *
from utils.utils_train import *
from utils.utils_tri import *
from utils.utils_vis import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
args = get_args()
save_args(args)
writer = SummaryWriter()

PATH1 = r'results/gpu'+str(args['folder'])+'/'
if bool(args['is_load']):
    model = DeepACM(AEdim=int(args['AEdim']), nP=int(args['nP']),
                    image_size=int(args['Idim']), drop=float(args['drop']))
    model1 = torch.load(PATH1 + 'net_' + args['CP'] + '.pt')
    model.load_state_dict(model1.state_dict())
else:
    model = DeepACM(AEdim=int(args['AEdim']), nP=int(args['nP']),
                    image_size=int(args['dim']), drop=float(args['drop']))
model.train().to(device)

P_test, faces_test = get_poly(int(args['dim']), int(args['nP']),
                              int(args['Radius']), int(args['dim']) / 2, int(args['dim']) / 2)
faces_test = faces_test.unsqueeze(dim=0).unsqueeze(dim=0).cuda()
faces = faces_test.repeat(int(args['Batch_size']), 1, 1, 1).cuda()
PTrain = P_test.repeat(int(args['Batch_size']), 1, 1, 1).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(args['learning_rate']))

if args['task'] == 'viah':
    trainset = viah_segmentation(ann='training', is_aug=False, args=args)
    testset = viah_segmentation(ann='test', is_aug=False, args=args)
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
best_tri = 0
for epoch in range(1, int(args['epochs'])):
    Maskloss_list, BLoss_list, NNLoss_list = train(ds, model, optimizer, criterion, PTrain, faces, epoch, args)
    logger_train(epoch, writer, Maskloss_list, BLoss_list, NNLoss_list)
    if epoch % 10 == 1:
        best_tri = eval_ds(ds_tri, model, writer, P_test, faces_test, epoch, PATH1, best_tri, 'training', args)
        best = eval_ds(ds_val, model, writer, P_test, faces_test, epoch, PATH1, best, 'test', args)














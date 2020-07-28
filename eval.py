import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from models.model import *
# from unet.model import *
import os
import argparse
import math
from utils.utils_tri import eval_utils
import cv2
from models.model_TL import *
from PIL import Image
from tqdm import tqdm

def fig2data ( fig ):
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    Copied from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    # draw the renderer
    fig.canvas.draw ()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf 
    
def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.frombytes( "RGBA", ( w ,h ), buf.tostring( ) )
    
def send_image_to_TB(img,P_init,P, Mask,Ix, Iy):
    dim2 = 256
    alpha = 0.5
    fig, ax = plt.subplots(nrows=1, ncols=4,figsize=[10, 10])
    img[:,:,1] = alpha*img[:,:,1] + alpha*Mask
    ax[0].imshow(img)
    ax[0].plot(P_init[0,:]*0.5*dim2+0.5*dim2,P_init[1,:]*0.5*dim2+0.5*dim2,'r--',linewidth=2.0, color=[1, 0, 0])
    ax[0].plot(P[0,:]*0.5*dim2+0.5*dim2,P[1,:]*0.5*dim2+0.5*dim2, 'r--',linewidth=2.0, color=[0, 1, 1])
    ax[1].imshow(Ix, cmap='Greys_r')        
    ax[2].imshow(Iy, cmap='Greys_r')
    ax[3].imshow((Ix**2+Iy**2)**0.5, cmap='Greys_r')
    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    ax[3].axis('off')
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    return np.asarray(fig2img ( fig ))
    
def send_image_to_TB_all(img,P_init,Mask,P,AE,Ix,Iy,GT):
    dim  = Mask.shape[0]
    dim2 = img.shape[0]
    fig, ax = plt.subplots(nrows=3, ncols=3,figsize=[10, 10])
    fig.subplots_adjust(wspace = 0, hspace = 0, left = 0,right = 1 ,bottom = 0,top = 1)
    ax[0,0].imshow(img)
    ax[0,1].plot(P_init[0,:]*0.5*dim2+0.5*dim2,P_init[1,:]*0.5*dim2+0.5*dim2,'r--',linewidth=2.0)
    ax[0,1].plot(P[0,:]*0.5*dim2+0.5*dim2,P[1,:]*0.5*dim2+0.5*dim2,color=[0, 1, 0],linewidth=2.0,marker = '*')
    # ax[0,1].plot(P[0,:]*0.5*dim+0.5*dim,P[1,:]*-0.5*dim+0.5*dim,'go',markersize=2)
    ax[0,1].imshow(img)
    ax[0,2].imshow(AE)
    ax[1,0].imshow(GT)
    ax[1,1].imshow(Mask)
    ax[1,2].plot(P_init[0,:]*0.5*dim+0.5*dim,P_init[1,:]*0.5*dim+0.5*dim,'ro')
    ax[1,2].plot(P[0,:]*0.5*dim+0.5*dim,P[1,:]*0.5*dim+0.5*dim,color=[0, 1, 0],linewidth=2.0,marker = '*')
    # ax[1,2].plot(P[0,:]*0.5*dim+0.5*dim,P[1,:]*-0.5*dim+0.5*dim,'go',markersize=2)
    ax[1,2].imshow(Mask)        
    ax[2,0].imshow(Ix)        
    ax[2,1].imshow(Iy)
    ax[2,2].imshow((Ix**2+Iy**2)**0.5)
    ax[0,0].axis('off')
    ax[0,1].axis('off')
    ax[0,2].axis('off')
    ax[1,0].axis('off')
    ax[1,1].axis('off')
    ax[1,2].axis('off')
    ax[2,0].axis('off')
    ax[2,1].axis('off')
    ax[2,2].axis('off')
    fig.suptitle('IoU: '+ str(cIoU))
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    return np.asarray(fig2img ( fig ))
    
def update_net_list(PATH):
    a = os.listdir(PATH)
    PATH1_list = []
    for i in a: 
        if i[0:3]=='net': PATH1_list.append(i)
    PATH1_list.sort()
    return PATH1_list
    
def get_poly(dim,n,IN,R):
    half_dim = dim / 2        
    half_width = half_dim
    half_height = half_dim
    P = [ np.array([
        half_width + math.floor( math.cos( 2 * math.pi / n * x ) * R ),
        half_height + math.floor( math.sin( 2 * math.pi / n * x ) * R ) ])
        for x in range( 0, n )]        
    train_data = torch.zeros(1,1,n,4).to('cuda:0')
    for i in range(n):
        train_data[0,0,i,0] = torch.tensor((P[i][0]-half_dim)/half_dim).clone()
        train_data[0,0,i,1] = torch.tensor((P[i][1]-half_dim)/half_dim).clone()
    train_data = train_data.repeat(IN,1,1,1)
    vertices = torch.ones((n,3))
    tmp = train_data[0,:,:,0:2].squeeze(dim = 0).squeeze(dim = 0)
    vertices[:,0] = tmp[:,0]
    vertices[:,1] = tmp[:,1]*-1
    tri = Delaunay(vertices[:,0:2].numpy())
    faces = torch.tensor(tri.simplices.copy())
    return (train_data,faces) 
    
def get_dice_ji(predict,target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return (dice,ji)
def get_number(s):
    return s.split('_')[2].split('.')[0]
    
torch.manual_seed(0)
np.random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('-gpu','--gpu', help='Description for foo argument', required=True)
parser.add_argument('-it' ,'--DeepIt', help='Description for foo argument', required=True)
parser.add_argument('-nP' ,'--nP', help='Description for foo argument', required=True)
parser.add_argument('-R'  ,'--R', help='Description for foo argument', required=True)
parser.add_argument('-DS'     ,'--DS'  ,default=1, help='which dataset to use?', required=False)


args    = vars(parser.parse_args())
gpu     = int(args['gpu'])
DeepIt  = int(args['DeepIt'])
nP      = int(args['nP'])
R       = int(args['R'])
DS      = int(args['DS'])

if DS == 1: 
    DataSet = 'Vaih'
    ind = 101
else: 
    DataSet = 'Bing'
    ind = 335

net_dir = DataSet+'/results/gpu'+str(gpu)+'/'	
PATH1   = net_dir+'net_DGVF_G1.pt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net    = torch.load(PATH1)
net.eval()
net.to(device)

XTest_path256  = DataSet+'/tensors/XTest_256.pt'
XTest_path     = DataSet+'/tensors/XTest_128.pt'
TTest_path     = DataSet+'/tensors/TTest_256.pt'
TTest_path128  = DataSet+'/tensors/TTest_128.pt'

print('loading Data.........')
XTest       = torch.load(XTest_path).to(device)
XTest_256   = torch.load(XTest_path256).to(device)
TTest       = torch.load(TTest_path).to(device)
TTest128    = torch.load(TTest_path128).to(device)

IN        = XTest.shape[0]
Idim      = XTest.shape[3]
PTrain,cF = get_poly(Idim,nP,IN,R)
cF        = cF.unsqueeze(dim=0).to(device)

IN     = XTest.shape[0]
uTestDice_list   = np.zeros((IN,1))
TestDice_list    = np.zeros((IN,1))
uTestIoU_list    = np.zeros((IN,1))
TestIoU_list     = np.zeros((IN,1))
WCov_list_val    = np.zeros((IN,1))
FBound_list_val  = np.zeros((IN,1))

is_single = 1
# unet_dir  = r'/media/data1/talshah/UNET/BuildingsSegmentation/results/resnet/128/res34'
unet_dir  = r'/media/data1/talshah/UNET/BuildingsSegmentation/results/UNET/gpu5'
if not is_single : PATH1_list = update_net_list(unet_dir)
else: PATH1_list = ['net_cnn_711.pt']
a = 0.6
best = 0
for inx2,PATH1 in tqdm(enumerate(PATH1_list)):
    unet = torch.load(unet_dir+'/'+PATH1)
    unet.eval()
    unet.to(device)
    for inx3,i in tqdm(enumerate(range(IN))):
        cData        = XTest[i].unsqueeze(dim=0).to(device)
        cData256     = XTest_256[i].unsqueeze(dim=0).to(device)
        cTarget      = TTest[i].unsqueeze(dim=0).to(device)
        cTarget128   = TTest128[i].unsqueeze(dim=0).to(device)
        cP           = PTrain[i,:,:,0:2].unsqueeze(dim=0).to(device)
        unet_out     = unet(cData)
        for it in range(DeepIt):
            MEAN    = cData.view(1,3,-1).mean(2).view(1,3,1,1)
            STDV    = cData.view(1,3,-1).std(2).view(1,3,1,1)
            cData   = (cData-MEAN)/STDV
            net_out = net(cData,cP,cF)
            cData   = (cData*STDV)+MEAN
            cP      = net_out[1]
            # tmp = torch.tensor(cv2.resize(net_out[0].squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy(),(128,128))).unsqueeze(dim=0).unsqueeze(dim=0) 
            # cData   = a*cData + (1-a)*tmp.cuda()
            cData   = a*cData + (1-a)*unet_out
        
        P = net_out[1]
        cP = PTrain[i,:,:,0:2]
        Mask  = net_out[0].squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
        img     = cv2.cvtColor(cData256.squeeze(dim = 0).detach().cpu().numpy().transpose(1,2,0), cv2.COLOR_BGR2RGB)
        P_init  = cP.squeeze(dim = 0).detach().cpu().numpy().transpose(1,0)
        P       = P.squeeze(dim = 0).squeeze(dim = 0).detach().cpu().numpy().transpose(1,0)
        P       = np.concatenate((P,P[:,0:1]),1)
        Ix      = net_out[3].squeeze(dim = 0).squeeze(dim = 0).detach().cpu().numpy()
        Iy      = net_out[4].squeeze(dim = 0).squeeze(dim = 0).detach().cpu().numpy()
        im      = send_image_to_TB(img,P_init,P, Mask,Ix, Iy)
        cv2.imwrite('ICLR/pic'+str(i)+'.jpg',im)
        
        Mask  = net_out[0].squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
        UMask = unet_out.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
        UMask[UMask<0.5]  = 0
        UMask[UMask>=0.5] = 1
        Mask[Mask<0.5]  = 0
        Mask[Mask>=0.5] = 1
        GT = cTarget.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
        GT[GT<0.5]  = 0
        GT[GT>=0.5] = 1
        GT128   = cTarget128.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
        (cDice,cIoU)      = get_dice_ji(Mask,GT)
        (uDice,uIoU)      = get_dice_ji(UMask,GT128)
        WCov   = eval_utils.WCov_metric(Mask, GT)
        Mask[Mask<1] = 0
        FBound = eval_utils.FBound_metric(Mask, GT)
        cP = PTrain[0,:,:,0:2].unsqueeze(dim=0).to(device)
        # print((i+ind,cDice,cIoU))
        TestDice_list[i]    = cDice
        TestIoU_list[i]     = cIoU
        uTestDice_list[i]    = uDice
        uTestIoU_list[i]     = uIoU
        WCov_list_val[i]    = WCov
        FBound_list_val[i]  = FBound

    # Dice   = np.mean(TestDice_list)
    # IoU    = np.mean(TestIoU_list)
    # WCov   = np.mean(WCov_list_val)
    # FBound = np.mean(FBound_list_val)
    if np.mean(TestIoU_list)>best:
        print(str(get_number(PATH1))+': Dice mean: ' +str(np.mean(TestDice_list)) +' Dice max: '+str(np.max(TestDice_list)) +' Dice min: '+str(np.min(TestDice_list)))
        print(str(get_number(PATH1))+': IoU mean: '  +str(np.mean(TestIoU_list))  +' IoU max: ' +str(np.max(TestIoU_list))  +' IoU min: ' +str(np.min(TestIoU_list)))
        best = np.mean(TestIoU_list)
    if is_single:
        print(str(get_number(PATH1))+': WCov mean: '  +str(np.mean(WCov_list_val))  +' WCov max: ' +str(np.max(WCov_list_val))  +' WCov min: ' +str(np.min(WCov_list_val)))
        print(str(get_number(PATH1))+': FBound mean: '  +str(np.mean(FBound_list_val))  +' FBound max: ' +str(np.max(FBound_list_val))  +' FBound min: ' +str(np.min(FBound_list_val)))
        print('*************************************************************************************')
        print(str(get_number(PATH1))+': unet Dice mean: ' +str(np.mean(uTestDice_list)) +' unet Dice max: '+str(np.max(uTestDice_list)) +' unet Dice min: '+str(np.min(uTestDice_list)))
        print(str(get_number(PATH1))+': unet IoU mean: '  +str(np.mean(uTestIoU_list))  +' unet IoU max: ' +str(np.max(uTestIoU_list))  +' unet IoU min: ' +str(np.min(uTestIoU_list)))

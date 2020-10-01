import numpy as np
from skimage.morphology import binary_dilation, disk
import os
import torch
from utils.utils_vis import send_image_to_TB
from utils.utils_train import norm_input
import cv2

def image_norm(img):
    return (img - img.min()) / (img.max() - img.min())


def vis_ds(ds, model, segnet, PTrain, faces, args, num_of_ex=5):
    model.eval()
    for ix, (_x, _y) in enumerate(ds):
        if ix > num_of_ex: break
        _x = _x.float().cuda()
        img = image_norm(_x.squeeze(dim=0).detach().cpu().numpy().transpose(1, 2, 0))
        _p = PTrain.float().cuda().clone()
        _y = _y.float().cuda()
        seg_out = segnet(_x)
        _x = norm_input(_x, seg_out, float(args['a']))
        iter = int(args['DeepIt'])
        net_out = model(_x, _p, faces, iter)
        Mask = net_out[0][iter - 1]
        (_, cIoU) = get_dice_ji(Mask, _y)
        P = net_out[1][iter - 1]
        P_init = PTrain.squeeze().detach().cpu().numpy().transpose(1, 0)
        Mask = Mask.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
        P = P.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy().transpose(1, 0)
        P = np.concatenate((P, P[:, 0:1]), 1)
        Ix = net_out[2].squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
        Iy = net_out[3].squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
        GT = _y.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
        im = cv2.cvtColor(send_image_to_TB(img, P_init, Mask, P, Ix, Iy, GT, cIoU), cv2.COLOR_RGBA2BGR)
        cv2.imwrite('out/' + str(ix) + '.jpg', im)
    model.train()


def eval_ds(ds, model, segnet, PTrain, faces, args):
    model.eval()
    TestIoU_list = []
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
            _, cIoU = get_dice_ji(Mask, _y)
            TestIoU_list.append(cIoU)
        IoU = np.mean(TestIoU_list)
        model.train()
        return IoU


def get_dice_ji(predict, target):
    predict = predict.data.cpu().numpy() + 1
    target = target.data.cpu().numpy() + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def update_net_list(PATH):
    a = os.listdir(PATH)
    PATH1_list = []
    for i in a:
        if i[0:5]=='model': PATH1_list.append(i)
    PATH1_list.sort()
    return PATH1_list


def dice_metric(X, Y):
    return np.sum(X[Y==1])*2.0 / (np.sum(X) + np.sum(Y) + 1e-6)


def IoU_metric(y_pred, y_true):
    intersection = np.sum(y_true * y_pred, axis=None)
    union = np.sum(y_true, axis=None) + np.sum(y_pred, axis=None) - intersection
    if float(union)==0: return 0.0
    else: return float(intersection) / float(union)


def WCov_metric(X, Y):
    A1 = float(np.count_nonzero(X))
    A2 = float(np.count_nonzero(Y))
    if A1>=A2: return A2/A1
    if A2>A1: return A1/A2


def FBound_metric(X, Y):
    tmp1 = db_eval_boundary(X,Y,1)[0]
    tmp2 = db_eval_boundary(X,Y,2)[0]
    tmp3 = db_eval_boundary(X,Y,3)[0]
    tmp4 = db_eval_boundary(X,Y,4)[0]
    tmp5 = db_eval_boundary(X,Y,5)[0]
    return (tmp1+tmp2+tmp3+tmp4+tmp5)/5.0
    
def db_eval_boundary(foreground_mask, gt_mask, bound_th):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.
    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
    Returns:
        F (float): boundaries F-measure
        P (float): boundaries precision
        R (float): boundaries recall
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1

    bound_pix = bound_th if bound_th >= 1 else \
            np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = seg2bmap(foreground_mask);
    gt_boundary = seg2bmap(gt_mask);

    fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
    gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg     = np.sum(fg_boundary)
    n_gt     = np.sum(gt_boundary)

    #% Compute precision and recall
    if n_fg == 0 and  n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0  and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match)/float(n_fg)
        recall    = np.sum(gt_match)/float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2*precision*recall/(precision+recall);

    return F, precision, recall, np.sum(fg_match), n_fg, np.sum(gt_match), n_gt


def seg2bmap(seg,width=None,height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width     : Width of desired bmap  <= seg.shape[1]
        height  :   Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray): Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """
    seg = seg.astype(np.bool)
    seg[seg>0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height

    h,w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
            'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1]    = seg[:,1:]
    s[:-1,:]    = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b        = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+floor((y-1)+height / h)
                    i = 1+floor((x-1)+width  / h)
                    bmap[j,i] = 1;

    return bmap
 
from __future__ import division, print_function

import torch
import numpy as np
from PIL import Image
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import importlib
import cv2
import xlrd
# import dar_package.config


def fig2img(fig):
    """
    Convert a Matplotlib figure to a PIL Image in RGBA format
    Copied from http://www.icare.univ-lille1.fr/tutorials/convert_a_matplotlib_figure
    """
    # put the figure pixmap into a np array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


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


def send_image_to_TB(img, P_init, Mask, P, Ix, Iy, GT, cIoU):
    dim  = Mask.shape[0]
    dim2 = img.shape[0]
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=[10, 10])
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    ax[0, 0].imshow(img)
    ax[0, 1].plot(P_init[0, :]*0.5*dim2+0.5*dim2, P_init[1, :]*0.5*dim2+0.5*dim2, 'r--', linewidth=2.0)
    ax[0, 1].plot(P[0, :]*0.5*dim2+0.5*dim2, P[1, :]*0.5*dim2+0.5*dim2, color=[0, 1, 0], linewidth=2.0, marker='*')
    ax[0, 1].imshow(img)
    ax[0, 2].imshow(img)
    ax[1, 0].imshow(GT)
    ax[1, 1].imshow(Mask)
    ax[1, 2].plot(P_init[0, :]*0.5*dim+0.5*dim, P_init[1, :]*0.5*dim+0.5*dim, 'ro')
    ax[1, 2].plot(P[0, :]*0.5*dim+0.5*dim, P[1, :]*0.5*dim+0.5*dim, color=[0, 1, 0], linewidth=2.0, marker='*')
    ax[1, 2].imshow(Mask)
    ax[2, 0].imshow(Ix)
    ax[2, 1].imshow(Iy)
    ax[2, 2].imshow((Ix**2+Iy**2)**0.5)
    ax[0, 0].axis('off')
    ax[0, 1].axis('off')
    ax[0, 2].axis('off')
    ax[1, 0].axis('off')
    ax[1, 1].axis('off')
    ax[1, 2].axis('off')
    ax[2, 0].axis('off')
    ax[2, 1].axis('off')
    ax[2, 2].axis('off')
    fig.suptitle('IoU: ' + str(cIoU))
    fig.tight_layout(pad=0, w_pad=0, h_pad=0)
    return np.asarray(fig2img(fig))

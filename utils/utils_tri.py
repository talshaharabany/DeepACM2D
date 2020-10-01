import torch
import math
import numpy as np
from scipy.spatial import Delaunay


def get_faces(P):
    N = P.shape[2]*2
    faces = torch.zeros(P.shape[0], N, 3)
    for i in range(P.shape[0]):
        cP = P[i, :, :, :].squeeze(dim=0).squeeze(dim=0)
        tri = Delaunay(cP.detach().cpu().numpy())
        tri = torch.tensor(tri.simplices.copy())
        nP = tri.shape[0]
        last = tri[nP-1, :].unsqueeze(dim=0)
        for j in range(N-nP):
            tri = torch.cat((tri, last), dim=0)
        faces[i, :, :] = tri.unsqueeze(dim=0)
    return faces.type(torch.int32)


def get_poly(dim=128, n=16, R=16, xx=64, yy=64):
    half_dim = dim / 2
    P = [np.array([xx + math.floor(math.cos(2 * math.pi / n * x) * R),
                   yy + math.floor(math.sin(2 * math.pi / n * x) * R)]) for x in range(0, n)]
    train_data = torch.zeros(1, 1, n, 2)
    for i in range(n):
        train_data[0, 0, i, 0] = torch.tensor((P[i][0] - half_dim) / half_dim).clone()
        train_data[0, 0, i, 1] = torch.tensor((P[i][1] - half_dim) / half_dim).clone()
    vertices = torch.ones((n, 3))
    tmp = train_data.squeeze(dim=0).squeeze(dim=0)
    vertices[:, 0] = tmp[:, 0]
    vertices[:, 1] = tmp[:, 1] * -1
    tri = Delaunay(vertices[:, 0:2].numpy())
    faces = torch.tensor(tri.simplices.copy())
    return train_data, faces

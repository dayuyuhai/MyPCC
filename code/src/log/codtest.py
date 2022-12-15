import numpy as np
import torch
import math

# gtocc = np.arange(1, 17).reshape(2, 8)
# print(gtocc)
# mask = np.zeros((8, 8))
# for i in range(7):mask[i+1:,i] = 1 
# print
# maskocc = (gtocc.reshape(-1, 1, 8) * mask.reshape(1, 8, 8))
# print(maskocc.shape)
# print(maskocc)

# maskocc = maskocc.reshape(-1, 8)
# print(maskocc.shape)
# print(maskocc)

# gtocc = np.arange(1, 65).reshape(4, 4, 4)
# print(gtocc)
# print(gtocc[2:, 2:, 2:])

gtocc = np.arange(1, 65).reshape(4, 4, 4)
print(gtocc)
len_ptoc = 8
ptoc_voxel = np.zeros((len_ptoc, len_ptoc, len_ptoc), dtype=np.int8)
ptoc_voxel[:, ::2, ::2, ::2] = gtocc[:, :, :]
ptoc_voxel[1::2, ::2, ::2] = gtocc[:, :, :]
ptoc_voxel[::2, 1::2, ::2] = gtocc[:, :, :]
ptoc_voxel[::2, ::2, 1::2] = gtocc[:, :, :]
ptoc_voxel[1::2, 1::2, ::2] = gtocc[:, :, :]
ptoc_voxel[1::2, ::2, 1::2] = gtocc[:, :, :]
ptoc_voxel[::2, 1::2, 1::2] = gtocc[:, :, :]
ptoc_voxel[1::2, 1::2, 1::2] = gtocc[:, :, :]
print(ptoc_voxel)
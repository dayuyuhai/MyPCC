import os
import torch
import numpy as np 
from utils.fastpc2oct import fastpc2oct, knn

def generate_data(source, target, level):
    files = os.listdir(source)[1:]
    exist = os.listdir(target)
    for i, file in enumerate(files):
        # if file[:-4] + "_" +  str(level) in exist:
        #     continue
        print(i, ":", file) 
        cloud = np.load(source + file)['arr_0'] 
        pc_voxel, octpc, query, voxel, gtocc = fastpc2oct(cloud, level)  
        octpc = torch.FloatTensor(octpc.reshape(1, octpc.shape[0], 3))
        step = octpc.shape[1] // ((int)(octpc.shape[1] * 0.1)) 
        for i in range(octpc.shape[1] // step):
            sample_idx = [i * step] 
            sample_point = octpc[:, sample_idx, :] 
            _, k_idx = knn(sample_point, octpc, 512) 
            sample_pc = octpc[:, k_idx.view(-1), :].view(-1, 1, 512, 3).repeat(1, 8, 1, 1).view(-1, 512, 3).numpy() 
            sample_pcvoxel = pc_voxel[sample_idx, :, :, :, :].reshape(-1, 8, 8, 8, 1).repeat(8,axis=0)
            sample_query = query[sample_idx, :, :].reshape(-1, 1, 3)
            sample_voxel = voxel[sample_idx, :, :, :, :, :].reshape(-1, 4, 4, 4, 1)
            sample_gtocc = gtocc[sample_idx,:].reshape(-1, 1)
            if i == 0:
                print("sample_pqvgtocc:", sample_pcvoxel.shape, sample_pc.shape, sample_query.shape, sample_voxel.shape, sample_gtocc.shape)
            for j, (s_pcv, s_pc, s_query, s_voxel, s_gtocc) in enumerate(zip(sample_pcvoxel, sample_pc, sample_query, sample_voxel, sample_gtocc)):
                # print(s_pc.shape, s_query.shape, s_voxel.shape, s_gtocc.shape)
                if os.path.exists(target + file[:-4] + "_" +  str(level)) is False:
                    os.makedirs(target + file[:-4] + "_" +  str(level))
                np.savez_compressed(target + file[:-4] + "_" +  str(level) + "/" + str(i * 8 + j) + ".npz", pc_voxel=s_pcv, pc=s_pc, query=s_query, voxel=s_voxel, gtocc=s_gtocc)


  


if __name__ == '__main__':
    generate_data('dataset/dense/test/pc/10/', 'dataset/dense/test/octree/', 6)
    # for level in range(4): 
    #     print(9, "test: ", level + 6)
    #     generate_data('dataset/dense/test/pc/9/', 'dataset/dense/test/octree/', 6 + level)
    # for level in range(5): 
    #     print(10, "test: ", level + 7)
    #     generate_data('dataset/dense/test/pc/10/', 'dataset/dense/test/octree/', 7 + level)
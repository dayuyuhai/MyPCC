import numpy as np
import torch
import math
import os
def knn(query, pc, k):#query:Bxmx3 pc:Bxnx3
    query = query.view(query.shape[0], query.shape[1], 1, query.shape[2])
    pc = pc.view(pc.shape[0], 1, pc.shape[1], pc.shape[2])
    dist = torch.sum((query - pc)**2, dim=-1)
 
    weight, idx = dist.topk(k=k, dim=-1, largest=False)   # (batch_size, num_points, k)
    weight = weight / torch.sum(weight, dim=-1, keepdim=True)
    return weight, idx 

def get_bounding_box(pc): 
    x_min, y_min, z_min = np.min(pc, axis=0)
    x_max, y_max, z_max = np.max(pc, axis=0) 
    x_origin, y_origin, z_origin = int(math.floor(x_min)), int(math.floor(y_min)), int(math.floor(z_min))
    bbox_size = [int(x_max - x_origin) + 1, int(y_max - y_origin) + 1, int(z_max - z_origin) + 1]
    
    max_size = np.max(bbox_size)
    if max_size > 512:
        bbox_size = 1024
    else:
        bbox_size = 512
    
    return  [[x_origin, y_origin, z_origin]], bbox_size 

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def local_pc_normalize(global_pc, local_pc, query, centroid):
    global_pc = global_pc - centroid
    local_pc = local_pc - centroid
    query = query - centroid
    m = np.max(np.sqrt(np.sum(global_pc**2, axis=1)))  
    global_pc = global_pc / m
    local_pc = local_pc /m
    query = query / m
    
    return global_pc, local_pc, query

def to_voxel(flatten):
    voxel = np.zeros((2,2,2,1))
    i = 0
    for z in [0, 1]:
        for y in [0, 1]:
            for x in [0, 1]:
                voxel[x][y][z] = flatten[i]
                i += 1
    return voxel

def flatten_voxel(voxel):
    flatten = np.zeros(8)
    i = 0
    for z in [0, 1]:
        for y in [0, 1]:
            for x in [0, 1]:
                flatten[i] = voxel[x][y][z]
                i += 1
    return flatten

def fastpc2oct(pc, level): 
    # print("the new one")
    bbox_origin, bbox_size = get_bounding_box(pc) 
    pc = pc - bbox_origin 
    bbox_origin = [[0, 0, 0]]
    # print("box:", bbox_size, pc.shape)
    id_formatter = "{x},{y},{z}"
    
    cube_number =  1 << (level-1) 
    cube_size =  bbox_size / cube_number
    cube_ids = pc[:, :3] // cube_size #floor division, like quantization
    # print("cube:", cube_number, cube_size)
    cube_ids = cube_ids.astype(np.uint32)
    cube_ids = np.unique(cube_ids, axis=0)
    # print(cube_ids.shape, cube_ids, np.max(np.max(cube_ids,axis=0)))
    cube_occupancy = {"id":"occ"}
    for id in cube_ids: cube_occupancy[id_formatter.format(x=id[0],y=id[1],z=id[2])] = 1
    
    pc_voxel = np.zeros((cube_number, cube_number, cube_number, 1), dtype=np.int8) 
    for id in cube_ids: pc_voxel[id[0], id[1], id[2], :] = 1

    fa_len = 8
    fa_halflen = 4
    fa_voxel = np.zeros((cube_ids.shape[0], fa_len, fa_len, fa_len, 1), dtype=np.int8) 
    for index, id in enumerate(cube_ids):
        tmp_voxel =  pc_voxel[max(id[0]-fa_halflen, 0):min(id[0]+fa_halflen, cube_number), max(id[1]-fa_halflen, 0):min(id[1]+fa_halflen, cube_number), max(id[2]-fa_halflen, 0):min(id[2]+fa_halflen, cube_number), :]
        fa_voxel[index, :, :, :, :] = np.pad(tmp_voxel, ((max(fa_halflen-id[0], 0), max(id[0]+fa_halflen-cube_number, 0)), (max(fa_halflen-id[1], 0), max(id[1]+fa_halflen-cube_number, 0)), (max(fa_halflen-id[2], 0), max(id[2]+fa_halflen-cube_number, 0)), (0, 0)), 'constant')
    # print("pcvoxel", pc_voxel.shape, fa_voxel.shape)
    
    sub_cube_number = 1 << level
    sub_cube_size = bbox_size / sub_cube_number
    sub_cube_ids = pc[:, :3] // sub_cube_size
    sub_cube_ids = sub_cube_ids.astype(np.uint32)
    sub_cube_ids = np.unique(sub_cube_ids, axis=0)
    sub_cube_occupancy = {"id":"occ"}
    for id in sub_cube_ids:sub_cube_occupancy[id_formatter.format(x=id[0],y=id[1],z=id[2])] = 1
    
    octpc = cube_ids * cube_size + cube_size / 2
    n = octpc.shape[0]
    querybias = np.array([[0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]])
    prebias = np.array([[-1, -1, -1], [0, -1, -1], [-1, 0, -1], [0, 0, -1], [-1, -1, 0], [0, -1, 0], [-1, 0, 0]])#七个前序节点相对当前节点的坐标偏移 
    preid = cube_ids.reshape(-1, 1, 3) + prebias.reshape(1, 7, 3)
    queryid = cube_ids.reshape(-1, 1, 3) * 2 + querybias.reshape(1, 8, 3)#nx8nx3
    query = queryid * sub_cube_size + sub_cube_size / 2
    gtocc = np.array([sub_cube_occupancy.get(id_formatter.format(x=id[0],y=id[1],z=id[2]), 0) for id in queryid.reshape(-1, 3)], dtype=np.int8).reshape(-1, 8)
    preocc =  np.array([sub_cube_occupancy.get(id_formatter.format(x=id[0],y=id[1],z=id[2]), 0) for id in (preid.reshape(-1, 7, 1, 3) * 2 + querybias.reshape(1, 1, 8, 3)).reshape(-1, 3)], dtype=np.int8).reshape(-1, 8)
    # preocc = np.repeat(preocc, 8, axis=1)
     
    mask = np.zeros((8, 8))
    for i in range(7):mask[i+1:,i] = 1 
    maskocc = (gtocc.reshape(-1, 1, 8) * mask.reshape(1, 8, 8)).reshape(-1, 8)
    voxel = np.zeros((n, 8, 4, 4, 4, 1), dtype=np.int8) 
    block_pre = np.array([to_voxel(flatten) for flatten in preocc], dtype=np.int8).reshape(n, 7, 2, 2, 2, 1)
    block_mask = np.array([to_voxel(flatten) for flatten in maskocc], dtype=np.int8).reshape(n, 8, 2, 2, 2, 1) 
    voxel[:,:,:2,:2,:2,:] = block_pre[:,0,:,:,:,:].reshape(n, 1, 2, 2, 2, 1).repeat(8,axis=1)
    voxel[:,:,2:,:2,:2,:] = block_pre[:,1,:,:,:,:].reshape(n, 1, 2, 2, 2, 1).repeat(8,axis=1)
    voxel[:,:,:2,2:,:2,:] = block_pre[:,2,:,:,:,:].reshape(n, 1, 2, 2, 2, 1).repeat(8,axis=1)
    voxel[:,:,2:,2:,:2,:] = block_pre[:,3,:,:,:,:].reshape(n, 1, 2, 2, 2, 1).repeat(8,axis=1)
    voxel[:,:,:2,:2,2:,:] = block_pre[:,4,:,:,:,:].reshape(n, 1, 2, 2, 2, 1).repeat(8,axis=1)
    voxel[:,:,2:,:2,2:,:] = block_pre[:,5,:,:,:,:].reshape(n, 1, 2, 2, 2, 1).repeat(8,axis=1)
    voxel[:,:,:2,2:,2:,:] = block_pre[:,6,:,:,:,:].reshape(n, 1, 2, 2, 2, 1).repeat(8,axis=1)
    voxel[:,:,2:,2:,2:,:] = block_mask[:,:,:,:,:,:] 
    # print(preocc.reshape(-1, 7, 8)[1000][0], flatten_voxel(voxel[1000][0][:2,:2,:2]))
    # print(preocc.reshape(-1, 7, 8)[1000][1], flatten_voxel(voxel[1000][0][2:,:2,:2]))
    # print(preocc.reshape(-1, 7, 8)[1000][2], flatten_voxel(voxel[1000][0][:2,2:,:2]))
    # print(preocc.reshape(-1, 7, 8)[1000][3], flatten_voxel(voxel[1000][0][2:,2:,:2]))
    # print(preocc.reshape(-1, 7, 8)[1000][4], flatten_voxel(voxel[1000][0][:2,:2,2:]))
    # print(preocc.reshape(-1, 7, 8)[1000][5], flatten_voxel(voxel[1000][0][2:,:2,2:]))
    # print(preocc.reshape(-1, 7, 8)[1000][6], flatten_voxel(voxel[1000][0][:2,2:,2:]))
    # print(gtocc[1000])
    # print(flatten_voxel(voxel[1000][0][2:,2:,2:]))
    # print(flatten_voxel(voxel[1000][1][2:,2:,2:]))
    # print(flatten_voxel(voxel[1000][2][2:,2:,2:]))
    # print(flatten_voxel(voxel[1000][3][2:,2:,2:]))
    # print(flatten_voxel(voxel[1000][4][2:,2:,2:]))
    # print(flatten_voxel(voxel[1000][5][2:,2:,2:]))
    # print(flatten_voxel(voxel[1000][6][2:,2:,2:]))
    # print(flatten_voxel(voxel[1000][7][2:,2:,2:]))
    # print("-----------------------------------------")
    # print("------")
    # print(octpc.shape, query.shape, voxel.shape, gtocc.shape)
    # print(octpc, "ieieie", query, "ieieie", voxel, "ieieie", gtocc)
    
    return fa_voxel, octpc, query, voxel, gtocc


    
    
 
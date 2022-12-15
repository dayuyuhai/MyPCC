import queue
import numpy as np
import torch
import open3d as o3d 
import os
import sys
sys.path.append('/data/ImplicitCompress/src') 
from encoder.pointnet import SimplePointnet
from encoder.pointnetpp import PointNetPlusPlus
from utils.fastpc2oct import knn, to_voxel 
from model.onet import DecoderCBatchNormBlk, OnetPlusPlus 
from entropy_codeing.binary_arithmetic_encoder import binary_arithmetic_coding 
device = torch.device('cpu')
 
def pc_normalize(pc): 
    centroid = np.mean(pc, axis=0) 
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1))) 
    pc = pc / m
    return pc

def getboundingbox(pc):
    radius = np.max(np.max(np.abs(pc),axis=0)) + 1e-4
    bbmin = [[-radius, -radius, -radius]]
    pc = pc - bbmin
    bbmax = [2*radius, 2*radius, 2*radius]
    return pc, [0, 0, 0], bbmax
 
def visualboundingbox(bbmin, bbmax): 
    vertex = [bbmin,[bbmin[0],bbmax[1], bbmin[2]],[bbmax[0], bbmax[1], bbmin[2]],[bbmax[0], bbmin[1], bbmin[2]], [bbmin[0], bbmin[1], bbmax[2]], [bbmin[0],bbmax[1], bbmax[2]], bbmax, [bbmax[0], bbmin[1],  bbmax[2]]]
    bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [[0, 1, 0] for _ in range(len(bbox_lines))]  #green
    bbox = o3d.geometry.LineSet()
    bbox.lines  = o3d.utility.Vector2iVector(bbox_lines)
    bbox.colors = o3d.utility.Vector3dVector(colors)
    bbox.points = o3d.utility.Vector3dVector(vertex) 
    return bbox
    
def split_octree(level, model, coder, oct_pc, sub_cube_occupancy, points, block_size, block_id):
    id_formatter = "{x},{y},{z}" 
    binstr = 0b00000000
    new_blocks = [() for x in range(8)]  
    block_pos = np.zeros((2, 2, 2),dtype=np.uint8)
    coding_order = np.zeros((2, 2, 2),dtype=np.uint8)
    pos = 0
    location = 0b10000000 
    for z in [2*block_id[2], 2*block_id[2]+1]:
            for y in [2*block_id[1], 2*block_id[1]+1]:
                for x in [2*block_id[0], 2*block_id[0]+1]:
                    coding_order[x-2*block_id[0]][y-2*block_id[1]][z-2*block_id[2]] = location
                    block_pos[x-2*block_id[0]][y-2*block_id[1]][z-2*block_id[2]] = pos
                    pos += 1
                    location = location >> 1  
    occupancy_points = []
    new_block_size = block_size / 2
    new_block_ids = points[:, :3] // new_block_size #floor division, like quantization
    new_block_ids = new_block_ids.astype(np.uint32)
    new_block_ids, new_block_idxs = np.unique(new_block_ids, return_inverse=True, axis=0) 
    for i, new_block_id in enumerate(new_block_ids):
        occupancy_points.append([new_block_id[0] * new_block_size + new_block_size / 2,  new_block_id[1] * new_block_size + new_block_size / 2, new_block_id[2] *new_block_size + new_block_size / 2])
        binstr = binstr | coding_order[new_block_id[0]-2*block_id[0]][new_block_id[1]-2*block_id[1]][new_block_id[2]-2*block_id[2]]
        new_points = points[new_block_idxs == i]
        new_blocks[block_pos[new_block_id[0]-2*block_id[0]][new_block_id[1]-2*block_id[1]][new_block_id[2]-2*block_id[2]]] = (new_points, new_block_size, new_block_id) 
    coding_blocks = []
    for block in new_blocks:
        if len(block) != 0:
            coding_blocks.append(block)
    if level < 5:
        return binstr, coding_blocks, np.array(occupancy_points)
    
    oct_pc = torch.FloatTensor(oct_pc.reshape(1, oct_pc.shape[0], 3)).to(device)      
    centroid = torch.FloatTensor((block_size * block_id + block_size / 2)).reshape(1, 1, 3).to(device) 
    k = 512 
    dist, k_idx = knn(centroid, oct_pc, k) 
    g_pc = oct_pc[:, k_idx.view(-1), :].view(k, 3) 
    l_pc = g_pc[:64,:]  
    querybias = np.array([[0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]])
    prebias = np.array([[-1, -1, -1], [0, -1, -1], [-1, 0, -1], [0, 0, -1], [-1, -1, 0], [0, -1, 0], [-1, 0, 0]])#父节点七个前序节点相对当前节点的坐标偏移
    preid = block_id.reshape(1, 3) + prebias
    queryid = preid.reshape(-1, 1, 3) * 2 + querybias.reshape(1, 8, 3)#7x8x3
    preocc = np.array([sub_cube_occupancy.get(id_formatter.format(x=id[0],y=id[1],z=id[2]), 0) for id in queryid.reshape(-1, 3)], dtype=np.int8).reshape(7, 8)
    block_pre = np.array([to_voxel(flatten) for flatten in preocc], dtype=np.int8).reshape(7, 2, 2, 2, 1)
    voxel = np.zeros((1, 4, 4, 4, 1), dtype=np.int8) 
    voxel[:,:2,:2,:2,:] = block_pre[0,:,:,:,:]
    voxel[:,2:,:2,:2,:] = block_pre[1,:,:,:,:]
    voxel[:,:2,2:,:2,:] = block_pre[2,:,:,:,:]
    voxel[:,2:,2:,:2,:] = block_pre[3,:,:,:,:]
    voxel[:,:2,:2,2:,:] = block_pre[4,:,:,:,:]
    voxel[:,2:,:2,2:,:] = block_pre[5,:,:,:,:]
    voxel[:,:2,2:,2:,:] = block_pre[6,:,:,:,:]
    voxel = torch.FloatTensor(voxel).to(device)
    location = 0b10000000 
    g_pc = g_pc.cpu().numpy()
    l_pc = l_pc.cpu().numpy()
    centroid = centroid.cpu().numpy().reshape(1, 3)
    g_pc = g_pc - centroid
    l_pc = l_pc - centroid
    m = np.max(np.sqrt(np.sum(g_pc**2, axis=1)))
    g_pc = g_pc / m
    l_pc = l_pc / m
    g_pc = torch.FloatTensor(g_pc.reshape(1, -1, 3)).to(device)
    l_pc = torch.FloatTensor(l_pc.reshape(1, -1, 3)).to(device) 
    for z in [2*block_id[2], 2*block_id[2]+1]:
        for y in [2*block_id[1], 2*block_id[1]+1]:
            for x in [2*block_id[0], 2*block_id[0]+1]:
                query = (new_block_size * np.array([x, y, z]) + new_block_size / 2).reshape(1, 3)
                query = query - centroid
                query = query / m
                query = torch.FloatTensor(query.reshape(1, -1, 3)).to(device)
                 
                #predict
                model.eval()
                with torch.no_grad():
                    probility = torch.sigmoid(model(g_pc, l_pc, query, voxel)).cpu().detach().numpy().reshape(-1) 
                if probility > 1-2e-6:
                    probility = 1-2e-6
                elif probility < 2e-6:
                    probility = 2e-6
                #compress
                sym = (int)((binstr & location) != 0)
                print(sym, probility, coder.bin2dec(coder.high) - coder.bin2dec(coder.low))
                coder.encode(probility, sym) 
                
                voxel[0][2+x-2*block_id[0]][2+y-2*block_id[1]][2+z-2*block_id[2]][0] = sym
                sub_cube_occupancy[id_formatter.format(x=x,y=y,z=z)] = sym
                location = location >> 1
                
                    
    
    return binstr, coding_blocks, np.array(occupancy_points)
    
    
def split_octree_by_occupancy(oct_pc, bitstr, block_size, block_id):
    location = 0b10000000 
    new_block_size = block_size / 2
    occupancy_points = []
    coding_blocks = []
    new_block_ids = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]]
    pos = 0
    while location != 0:
        occupancy = location & bitstr
        if occupancy:#占用
            new_block_id = 2*np.array(block_id) + np.array(new_block_ids[pos])
            occupancy_points.append([new_block_id[0] * new_block_size + new_block_size / 2,  new_block_id[1] * new_block_size + new_block_size / 2, new_block_id[2] *new_block_size + new_block_size / 2])
            coding_blocks.append((new_block_size, new_block_id))
        pos += 1
        location = location >> 1
    return coding_blocks, occupancy_points
     

def partition_octree(points, model, coder, bpp1, bpp2, level):   
    points = pc_normalize(points)
    points, bbmin, bbmax = getboundingbox(points)
    oct_pc = (np.array(bbmax) / 2).reshape(1, -1)
    block_size = bbmax[0] 
    block_ids = points[:, :3] // block_size #floor division, like quantization
    block_ids = block_ids.astype(np.uint32)
    block_ids = np.unique(block_ids, axis=0)
    binstr_list = []
    blocks = []
    blocks.append((points, block_size, block_ids[0])) 
    for l in range(level-1):
        sub_cube_occupancy = {"id":"occ"}
        new_blocks = []
        new_oct_pc = []
        for block in blocks:
            binstr, coding_blocks, occupancy_points = split_octree(l, model, coder, oct_pc, sub_cube_occupancy, *block)
            if l < 5:
                binstr_list = binstr_list + [binstr]
            new_blocks.extend(coding_blocks)
            new_oct_pc.extend(occupancy_points)
        blocks = new_blocks
        oct_pc = np.array(new_oct_pc)
        if l >= 5:
            bpp1[l+1].append((len(coder.bitstream)) / oct_pc.shape[0])
            bpp2[l+1].append((len(binstr_list)*8 + len(coder.bitstream)) / oct_pc.shape[0])
            print('bpp1:', bpp1, 'bpp2:', bpp2) 
            np.savez('/data/ImplicitCompress/src/entropy_codeing/bpp.npz', bpp1=bpp1, bpp2=bpp2)
    return binstr_list, oct_pc, bbmax 
            
            
    
def departition_octree(bitstr_list, bbmax):
    oct_pc = (np.array(bbmax) / 2).reshape(1, -1)
    block_size = bbmax[0] 
    block_id = [0, 0, 0]
    blocks = [(block_size, block_id)]
    i = 0
    while i < len(bitstr_list):
        k = len(blocks)
        block_bitstr = bitstr_list[i:i+k]
        new_oct_pc = []
        new_blocks = []
        for bitstr, block in zip(block_bitstr, blocks):
            coding_blocks, occupancy_points = split_octree_by_occupancy(oct_pc, bitstr, *block)
            new_blocks.extend(coding_blocks)
            new_oct_pc.extend(occupancy_points)
        blocks = new_blocks
        oct_pc = new_oct_pc    
        i += k
    return oct_pc


if __name__ == '__main__':
     
    global_encoder = PointNetPlusPlus()
    local_encoder = SimplePointnet() 
    decoder = DecoderCBatchNormBlk()
    model = OnetPlusPlus(global_encoder, local_encoder, decoder).to(device) 
    state_dict = torch.load('weights/model-4.pth')
    model.load_state_dict(state_dict) 
    files = os.listdir('entropy_codeing/test/')
    bpp1 = {6:[],7:[],8:[],9:[],10:[]}#6bit point cloud --> 7 level octree
    bpp2 = {6:[],7:[],8:[],9:[],10:[]}  
    for i, file in enumerate(files): 
        coder = binary_arithmetic_coding()   
        print('level', ":", 11, file)
        points = np.load('entropy_codeing/test/' + file)['arr_0'] 
        binstr_list, oct_pc, bbmax = partition_octree(points, model, coder, bpp1, bpp2, 11)
        coder.end()  
      
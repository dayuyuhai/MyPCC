
import torch
from  torch import  nn
from model.attentionnet import attentionnet 
import torch.nn.functional as F 
from utils.fastpc2oct import knn 
from encoder.voxelcnn import parent_voxelcnn, gc_voxelcnn, voxelcnn
from model.layers import  CResBatchnormBlock ,CBatchNorm1d, CBatchNorm1d_legacy

class DecoderCBatchNormBlk(nn.Module):
    def __init__(self, dim=512, global_c_dim=1024, local_c_dim = 256,
                 hidden_size=256, leaky=True, legacy=False): 
        super().__init__()
        self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.block0 = CResBatchnormBlock(global_c_dim, local_c_dim, hidden_size, legacy=legacy)
        self.block1 = CResBatchnormBlock(global_c_dim, local_c_dim, hidden_size, legacy=legacy)
        self.block2 = CResBatchnormBlock(global_c_dim, local_c_dim, hidden_size, legacy=legacy)
        self.block3 = CResBatchnormBlock(global_c_dim, local_c_dim, hidden_size, legacy=legacy)
        self.block4 = CResBatchnormBlock(global_c_dim, local_c_dim, hidden_size, legacy=legacy) 

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not legacy:
            self.bn = CBatchNorm1d(global_c_dim + local_c_dim, hidden_size)
        else:
            self.bn = CBatchNorm1d_legacy(global_c_dim + local_c_dim, hidden_size)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, global_c, local_c, query, mask):
        net = torch.cat([query, mask], dim=1)
        net = self.fc_p(net)
        net = self.block0(net, global_c, local_c)
        net = self.block1(net, global_c, local_c)
        net = self.block2(net, global_c, local_c)
        net = self.block3(net, global_c, local_c)
        net = self.block4(net, global_c, local_c)

        net =  net + query + mask

        c = torch.cat([global_c, local_c], dim=1) 
        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out
    

class OnetPlusPlus(nn.Module):
    def __init__(self, global_encoder, local_edcoder,  decoder, leaky=True) -> None:
        super().__init__()
        
        self.global_encoder = global_encoder
        self.local_encoder = local_edcoder
        self.decoder = decoder
       
        self.fc_voxel = voxelcnn()
        self.gc_voxel = gc_voxelcnn()
        self.position_embding = nn.Sequential(
            nn.Conv1d(3, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.fc_query = attentionnet(128, 128, 128)

    def forward(self, pc_voxel, global_pc, local_pc, query, voxel): 
        b,_,_ = global_pc.shape   
        global_c, point_feature = self.global_encoder(global_pc)
        # print("p be", point_feature.shape)
        local_c = self.local_encoder(local_pc)
        # print("pv be", pc_voxel.shape)
        # print("g be", global_c.shape)
        # print("l be", local_c.shape)
        # print("q be", query.shape)
        # print("v be", voxel.shape)
 
        _, idx = knn(query, global_pc, 16)
        idx = idx.view(-1, 16)
        knn_pos = None
        knn_feature = None
        for id, q, p, p_f in zip(idx, query, global_pc, point_feature):
            if knn_pos == None:
                knn_pos = (q - p[id]).view(1, 1, 16, 3)
            else:
                knn_pos = torch.cat([knn_pos, (q - p[id]).view(1, 1, 16, 3)], dim=0)
            if knn_feature == None:
                knn_feature = p_f[id].view(1, 1, 16, 128)
            else:
                knn_feature = torch.cat([knn_feature, p_f[id].view(1, 1, 16, 128)], dim=0)  
        query = query.permute(0, 2, 1)
        pc_voxel = pc_voxel.permute(0, 4, 1, 2, 3)
        pc_voxel_feature = self.gc_voxel(pc_voxel)
        # print("pvf ", pc_voxel_feature.shape)
        pc_voxel_feature = pc_voxel_feature.reshape(pc_voxel.shape[0], -1, 1) 
        # print("pvf ", pc_voxel_feature.shape)
        
        voxel = voxel.permute(0, 4, 1, 2, 3)
        voxel_feature = self.fc_voxel(voxel)
        # print("vf ", voxel_feature.shape)
        voxel_feature = voxel_feature.reshape(voxel.shape[0], -1, 1) 
        # print("vf ", voxel_feature.shape)
        local_c = local_c.permute(0, 2, 1)  
        global_c = global_c.permute(0, 2, 1)   
        # print("g af", global_c.shape)
        # print("l af", local_c.shape)
        # print("q af", query.shape)
        # print("v af", voxel.shape)
        # print("knn p be", knn_pos.shape)
        # print("knn f be", knn_feature.shape)
        knn_pos = knn_pos.permute(0, 3, 1, 2)
        knn_feature = knn_feature.permute(0, 3, 1, 2)
        # print("knn p af", knn_pos.shape)
        # print("knn f af", knn_feature.shape)
  
        position_feature = self.position_embding(query)
        # print("pos f", position_feature.shape)
        query_feature = self.fc_query(knn_feature, knn_pos).permute(0, 2, 1)
        # print("q f", query_feature.shape)
        query_feature = torch.cat([query_feature, position_feature], dim=1)
        # print("q f", query_feature.shape)
        out = self.decoder(pc_voxel_feature, local_c, query_feature, voxel_feature)
        # print("out", out.shape)

        return out

class OnetPlusPlus2(nn.Module):
    def __init__(self, global_encoder, local_edcoder,  decoder, leaky=True) -> None:
        super().__init__()
        
        self.global_encoder = global_encoder
        self.local_encoder = local_edcoder
        self.decoder = decoder
       
        self.fc_voxel = parent_voxelcnn()
        self.position_embding = nn.Sequential(
            nn.Conv1d(3, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.fc_query = attentionnet(128, 128, 128)

    def forward(self, global_pc, local_pc, query, voxel): 
        b,_,_ = global_pc.shape   
        global_c, point_feature = self.global_encoder(global_pc)
        # print("p be", point_feature.shape)
        local_c = self.local_encoder(local_pc)
        # print("pv be", pc_voxel.shape)
        # print("g be", global_c.shape)
        # print("l be", local_c.shape)
        # print("q be", query.shape)
        # print("v be", voxel.shape)
 
        _, idx = knn(query, global_pc, 16)
        idx = idx.view(-1, 16)
        knn_pos = None
        knn_feature = None
        for id, q, p, p_f in zip(idx, query, global_pc, point_feature):
            if knn_pos == None:
                knn_pos = (q - p[id]).view(1, 1, 16, 3)
            else:
                knn_pos = torch.cat([knn_pos, (q - p[id]).view(1, 1, 16, 3)], dim=0)
            if knn_feature == None:
                knn_feature = p_f[id].view(1, 1, 16, 128)
            else:
                knn_feature = torch.cat([knn_feature, p_f[id].view(1, 1, 16, 128)], dim=0)  
        query = query.permute(0, 2, 1)

        voxel = voxel.permute(0, 4, 1, 2, 3)
        voxel_feature = self.fc_voxel(voxel)
        # print("vf ", voxel_feature.shape)
        voxel_feature = voxel_feature.reshape(voxel.shape[0], -1, 1) 
        # print("vf ", voxel_feature.shape)
        local_c = local_c.permute(0, 2, 1)  
        global_c = global_c.permute(0, 2, 1)   
        # print("g af", global_c.shape)
        # print("l af", local_c.shape)
        # print("q af", query.shape)
        # print("v af", voxel.shape)
        # print("knn p be", knn_pos.shape)
        # print("knn f be", knn_feature.shape)
        knn_pos = knn_pos.permute(0, 3, 1, 2)
        knn_feature = knn_feature.permute(0, 3, 1, 2)
        # print("knn p af", knn_pos.shape)
        # print("knn f af", knn_feature.shape)
  
        position_feature = self.position_embding(query)
        # print("pos f", position_feature.shape)
        query_feature = self.fc_query(knn_feature, knn_pos).permute(0, 2, 1)
        # print("q f", query_feature.shape)
        query_feature = torch.cat([query_feature, position_feature], dim=1)
        # print("q f", query_feature.shape)
        out = self.decoder(global_c, local_c, query_feature, voxel_feature)
        # print("out", out.shape)

        return out





 
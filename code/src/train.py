import os 
import argparse
import numpy as np
import torch  
from torch.utils.tensorboard import SummaryWriter 
import sys 
from tqdm import tqdm  
from utils.distributed_utils import reduce_value, is_main_process
from utils.distributed_utils import init_distributed_mode,  cleanup
from math import cos, pi 
from utils.fastpc2oct import knn, local_pc_normalize 
from torch.utils.data import Dataset 
from encoder.pointnet import SimplePointnet
from encoder.pointnetpp import PointNetPlusPlus
from model.onet import DecoderCBatchNormBlk, OnetPlusPlus 
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score 
import warnings
warnings.filterwarnings("ignore")    

class train_dataset(Dataset):
    def __init__(self, dir): 
        self.len = 0
        self.dir = dir
        self.files = os.listdir(self.dir)
        self.pc_nums = []
        for file in self.files:
            self.pc_nums.append(len(os.listdir(self.dir+file+"/")))
        for num in self.pc_nums:
            self.len += num 
        
    def __getitem__(self, index): 
        # print(index)
        i = 0 
        while index >= self.pc_nums[i]:
            index -= self.pc_nums[i]
            i += 1 
        data = np.load(self.dir+self.files[i]+"/" + str(index) + ".npz") 
        g_pc = data["pc"]
        l_pc = g_pc[:64,:] 
        query = data["query"]
        centroid = g_pc[0:1,:]
        g_pc, l_pc, query = local_pc_normalize(g_pc, l_pc, query, centroid)
        g_pc = torch.FloatTensor(g_pc)
        l_pc = torch.FloatTensor(l_pc)
        query = torch.FloatTensor(query) 
        occ = torch.FloatTensor(data["gtocc"]) 
        voxel = torch.FloatTensor(data["voxel"]) 
        pc_voxel = torch.FloatTensor(data["pc_voxel"]) 
        return pc_voxel, g_pc, l_pc, query, voxel, occ  
 
    def __len__(self): 
        return self.len
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        pc_voxel, g_pc, l_pc, query, voxel, occ = tuple(zip(*batch)) 
        g_pc = torch.stack(g_pc, dim=0)
        l_pc = torch.stack(l_pc, dim=0)
        query = torch.stack(query, dim=0)
        voxel = torch.stack(voxel, dim=0).contiguous()
        pc_voxel = torch.stack(pc_voxel, dim=0).contiguous()
        occ = torch.stack(occ, dim=0)
        return pc_voxel, g_pc, l_pc, query, voxel, occ
     
class test_dataset(Dataset):
    def __init__(self, dir, level): 
        self.dir = dir
        self.len = 0
        files = os.listdir(self.dir)
        self.files =  []
        self.pc_nums = []
        for file in files:
            if file.split('_')[-1] == level:
                self.files.append(file)
                self.pc_nums.append(len(os.listdir(self.dir+file+"/")))
        for num in self.pc_nums:
            self.len += num 
        
    def __getitem__(self, index):    
        i = 0 
        while index >= self.pc_nums[i]:
            index -= self.pc_nums[i]
            i += 1 
        data = np.load(self.dir+self.files[i]+"/" + str(index) + ".npz")
        g_pc = data["pc"]
        l_pc = g_pc[:64,:] 
        query = data["query"]
        centroid = g_pc[0:1,:]
        g_pc, l_pc, query = local_pc_normalize(g_pc, l_pc, query, centroid)
        g_pc = torch.FloatTensor(g_pc)
        l_pc = torch.FloatTensor(l_pc)
        query = torch.FloatTensor(query) 
        occ = torch.FloatTensor(data["gtocc"]) 
        pc_voxel = torch.FloatTensor(data["pc_voxel"]) 
        voxel = torch.FloatTensor(data["voxel"]) 
        return pc_voxel, g_pc, l_pc, query, voxel, occ 
 
    def __len__(self): 
        return self.len
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        g_pc, l_pc, query, voxel, occ = tuple(zip(*batch)) 
        g_pc = torch.stack(g_pc, dim=0)
        l_pc = torch.stack(l_pc, dim=0)
        query = torch.stack(query, dim=0)
        voxel = torch.stack(voxel, dim=0).contiguous()
        pc_voxel = torch.stack(pc_voxel, dim=0).contiguous()
        occ = torch.stack(occ, dim=0)
        return pc_voxel, g_pc, l_pc, query, voxel, occ
 
def Precision(pred, gt):
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.view(-1)
    gt = gt.view(-1)
    return torch.FloatTensor([precision_score(gt.cpu().numpy(), pred.cpu().detach().numpy())])

def Recall(pred, gt):
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.view(-1)
    gt = gt.view(-1)
    return torch.FloatTensor([recall_score(gt.cpu().numpy(), pred.cpu().detach().numpy())])

def F1score(pred, gt):
    pred = torch.sigmoid(pred) > 0.5
    pred = pred.view(-1)
    gt = gt.view(-1)
    return torch.FloatTensor([f1_score(gt.cpu().numpy(), pred.cpu().detach().numpy())])

def Auc(pred, gt): 
    pred = torch.sigmoid(pred).view(-1)
    gt = gt.view(-1)
    return torch.FloatTensor([roc_auc_score(gt.cpu().numpy(), pred.cpu().detach().numpy())])

def Pred_value_0(pred, gt):
    pred = torch.sigmoid(pred).cpu().detach().numpy().reshape(-1)
    gt = gt.cpu().numpy().reshape(-1)
    index_0 = gt == 0 
    pred_value_0 = np.sum(pred[index_0]) / pred[index_0].shape[0] 
    return torch.FloatTensor([pred_value_0])

def Pred_value_1(pred, gt):
    pred = torch.sigmoid(pred).cpu().detach().numpy().reshape(-1)
    gt = gt.cpu().numpy().reshape(-1) 
    index_1 = gt == 1 
    pred_value_1 = np.sum(pred[index_1]) / pred[index_1].shape[0]
    return torch.FloatTensor([pred_value_1])

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.BCEWithLogitsLoss()  
    mean_loss = torch.zeros(1).to(device)
    mean_precision = torch.zeros(1).to(device)
    mean_recall = torch.zeros(1).to(device)
    mean_auc = torch.zeros(1).to(device)
    mean_f1score = torch.zeros(1).to(device)
    mean_value_1 = torch.zeros(1).to(device)
    mean_value_0 = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    # 这里相当于每次处理一个batch，每个batch的数据会分在不同卡上并行算？
    for step, data in enumerate(data_loader):
        pc_voxel, global_pc, local_pc, query, voxel, occ_gt = data
        pc_voxel = pc_voxel.to(device, non_blocking=True)
        global_pc = global_pc.to(device, non_blocking=True)
        local_pc = local_pc.to(device, non_blocking=True)
        query = query.to(device, non_blocking=True)
        voxel = voxel.to(device, non_blocking=True)
        occ_gt = occ_gt.to(device, non_blocking=True)
        pred = model(pc_voxel, global_pc, local_pc, query, voxel)
        # print("train one epoch:", pc_voxel.shape, global_pc.shape, local_pc.shape, query.shape, voxel.shape, occ_gt.shape, pred.shape)
        loss = loss_function(pred, occ_gt)
        loss.backward()
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        
        precision = Precision(pred, occ_gt).to(device)
        precision = reduce_value(precision, average=True)
        mean_precision = (mean_precision * step + precision ) / (step + 1)
        
        recall = Recall(pred, occ_gt).to(device)
        recall = reduce_value(recall, average=True)
        mean_recall = (mean_recall * step + recall) / (step + 1)
        
        auc = Auc(pred, occ_gt).to(device)
        auc = reduce_value(auc, average=True)
        mean_auc = (mean_auc * step + auc) / (step + 1)
        
        f1score = F1score(pred, occ_gt).to(device)
        f1score = reduce_value(f1score, average=True)
        mean_f1score = (mean_f1score * step + f1score) / (step + 1)
        
        pred_value_1 = Pred_value_1(pred, occ_gt).to(device)
        pred_value_1 = reduce_value(pred_value_1, average=True)
        mean_value_1 = (mean_value_1 * step + pred_value_1) / (step + 1)
        
        pred_value_0 = Pred_value_0(pred, occ_gt).to(device)
        pred_value_0 = reduce_value(pred_value_0, average=True)
        mean_value_0 = (mean_value_0 * step + pred_value_0) / (step + 1)
        

        # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[epoch {}] train_loss {} precison {} recall {} auc {} f1 {}".format(epoch, round(mean_loss.item(), 5),round(mean_precision.item(), 5), round(mean_recall.item(), 5), round(mean_auc.item(), 5), round(mean_f1score.item(), 5))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item(), mean_precision, mean_recall, mean_auc, mean_f1score, mean_value_0, mean_value_1


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, level):
    model.eval()
    loss_function = torch.nn.BCEWithLogitsLoss()  
    mean_loss = torch.zeros(1).to(device) 
    mean_precision = torch.zeros(1).to(device)
    mean_recall = torch.zeros(1).to(device)
    mean_auc = torch.zeros(1).to(device)
    mean_f1score = torch.zeros(1).to(device)
    mean_value_1 = torch.zeros(1).to(device)
    mean_value_0 = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        global_pc, local_pc, query, voxel, occ_gt = data
        global_pc = global_pc.to(device)
        local_pc = local_pc.to(device)
        query = query.to(device)
        voxel = voxel.to(device)
        occ_gt = occ_gt.to(device)
        pred = model(global_pc, local_pc, query, voxel.contiguous())

        loss = loss_function(pred, occ_gt) 
        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses
        
        precision = Precision(pred, occ_gt).to(device)
        precision = reduce_value(precision, average=True)
        mean_precision = (mean_precision * step + precision ) / (step + 1)
        
        recall = Recall(pred, occ_gt).to(device)
        recall = reduce_value(recall, average=True)
        mean_recall = (mean_recall * step + recall) / (step + 1)
        
        auc = Auc(pred, occ_gt).to(device)
        auc = reduce_value(auc, average=True)
        mean_auc = (mean_auc * step + auc) / (step + 1)
        
        f1score = F1score(pred, occ_gt).to(device)
        f1score = reduce_value(f1score, average=True)
        mean_f1score = (mean_f1score * step + f1score) / (step + 1)
        
        pred_value_1 = Pred_value_1(pred, occ_gt).to(device)
        pred_value_1 = reduce_value(pred_value_1, average=True)
        mean_value_1 = (mean_value_1 * step + pred_value_1) / (step + 1)
        
        pred_value_0 = Pred_value_0(pred, occ_gt).to(device)
        pred_value_0 = reduce_value(pred_value_0, average=True)
        mean_value_0 = (mean_value_0 * step + pred_value_0) / (step + 1)

    # 在进程0中打印平均loss
        if is_main_process():
            data_loader.desc = "[level {} epoch {}] test_loss {} precison {} recall {} auc {} f1 {}".format(level, epoch, round(mean_loss.item(), 5),round(mean_precision.item(), 5), round(mean_recall.item(), 5), round(mean_auc.item(), 5), round(mean_f1score.item(), 5))
        # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)
    return mean_loss.item(), mean_precision, mean_recall, mean_auc, mean_f1score, mean_value_0, mean_value_1

def adjust_learning_rate(optimizer, current_epoch,max_epoch=10,lr_min=1e-6,lr_max=1e-3,warmup=True):
    warmup_epoch = 2 if warmup else 0
    if current_epoch <= warmup_epoch:
        lr = lr_max * (current_epoch*50 + 1) / (warmup_epoch*50 + 1)
    else:
        lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  


def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")

    # 初始化各进程环境
    init_distributed_mode(args=args)

    rank = args.rank
    device = torch.device(args.device)
    batch_size = args.batch_size  
    checkpoint_path = ""

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        tb_writer = SummaryWriter(flush_secs=120)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/') 
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")
         

    
    if rank == 0:
        print('Using {} dataloader workers every process'.format(10))
    # 实例化训练数据集
    train_data_set = train_dataset('dataset/dense/train/octree/')
    # 给每个rank对应的进程分配训练的样本索引 
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)
    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=10,
                                               collate_fn=train_data_set.collate_fn)

    # 实例化验证数据集
    val_data_set_6 = test_dataset('dataset/dense/test/octree/', "6") 
    val_sampler_6 = torch.utils.data.distributed.DistributedSampler(val_data_set_6)  
    val_loader_6 = torch.utils.data.DataLoader(val_data_set_6,
                                             batch_size=batch_size,
                                             sampler=val_sampler_6,
                                             pin_memory=True,
                                             num_workers=32,
                                             collate_fn=val_data_set_6.collate_fn)
    
    val_data_set_7 = test_dataset('dataset/dense/test/octree/', "7") 
    val_sampler_7 = torch.utils.data.distributed.DistributedSampler(val_data_set_7)  
    val_loader_7 = torch.utils.data.DataLoader(val_data_set_7,
                                             batch_size=batch_size,
                                             sampler=val_sampler_7,
                                             pin_memory=True,
                                             num_workers=32,
                                             collate_fn=val_data_set_7.collate_fn)
    
    val_data_set_8 = test_dataset('dataset/dense/test/octree/', "8") 
    val_sampler_8 = torch.utils.data.distributed.DistributedSampler(val_data_set_8)  
    val_loader_8 = torch.utils.data.DataLoader(val_data_set_8,
                                             batch_size=batch_size,
                                             sampler=val_sampler_8,
                                             pin_memory=True,
                                             num_workers=32,
                                             collate_fn=val_data_set_8.collate_fn)
    
    val_data_set_9 = test_dataset('dataset/dense/test/octree/', "9") 
    val_sampler_9 = torch.utils.data.distributed.DistributedSampler(val_data_set_9)  
    val_loader_9 = torch.utils.data.DataLoader(val_data_set_9,
                                             batch_size=batch_size,
                                             sampler=val_sampler_9,
                                             pin_memory=True,
                                             num_workers=32,
                                             collate_fn=val_data_set_9.collate_fn)
    
    val_data_set_10 = test_dataset('dataset/dense/test/octree/', "10") 
    val_sampler_10 = torch.utils.data.distributed.DistributedSampler(val_data_set_10)  
    val_loader_10 = torch.utils.data.DataLoader(val_data_set_10,
                                             batch_size=batch_size,
                                             sampler=val_sampler_10,
                                             pin_memory=True,
                                             num_workers=32,
                                             collate_fn=val_data_set_10.collate_fn)
    # 实例化模型
    global_encoder = PointNetPlusPlus()
    local_encoder = SimplePointnet() 
    decoder = DecoderCBatchNormBlk()
    model = OnetPlusPlus(global_encoder, local_encoder, decoder).to(device) 
    # state_dict = torch.load('weights/model-4.pth')
    # model.load_state_dict(state_dict) 

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # optimizer 
    optimizer = torch.optim.AdamW(params=model.parameters(),lr=1e-3,betas=(0.9,0.999),eps=1e-08,weight_decay=0.01,amsgrad=False)  
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        
        adjust_learning_rate(optimizer, current_epoch=epoch,max_epoch=args.epochs,lr_min=1e-6,lr_max=1e-3,warmup=True)

        train_loss,train_precision, train_recall, train_auc, train_f1score, train_value_0, train_value_1 = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)  
        test_loss_6,test_precision_6, test_recall_6, test_auc_6, test_f1score_6, test_value_0_6, test_value_1_6 = evaluate(model=model,
                           data_loader=val_loader_6,
                           device=device,
                           epoch=epoch,
                           level=6) 
        test_loss_7,test_precision_7, test_recall_7, test_auc_7, test_f1score_7, test_value_0_7, test_value_1_7 = evaluate(model=model,
                           data_loader=val_loader_7,
                           device=device,
                           epoch=epoch,
                           level=7) 
        test_loss_8,test_precision_8, test_recall_8, test_auc_8, test_f1score_8, test_value_0_8, test_value_1_8 = evaluate(model=model,
                           data_loader=val_loader_8,
                           device=device,
                           epoch=epoch,
                           level=8) 
        test_loss_9,test_precision_9, test_recall_9, test_auc_9, test_f1score_9, test_value_0_9, test_value_1_9 = evaluate(model=model,
                           data_loader=val_loader_9,
                           device=device,
                           epoch=epoch,
                           level=9) 
        test_loss_10,test_precision_10, test_recall_10, test_auc_10, test_f1score_10, test_value_0_10, test_value_1_10 = evaluate(model=model,
                           data_loader=val_loader_10,
                           device=device,
                           epoch=epoch,
                           level=10) 

        if rank == 0: 
            tb_writer.add_scalars(main_tag='loss',
                                tag_scalar_dict={'train/loss': train_loss, 'test/loss_6': test_loss_6, 'test/loss_7': test_loss_7, 'test/loss_8': test_loss_8, 'test/loss_9': test_loss_9, 'test/loss_10': test_loss_10},
                                global_step=epoch)
            tb_writer.add_scalars(main_tag='precision',
                                tag_scalar_dict={'train/precision': train_precision, 'test/precision_6': test_precision_6, 'test/precision_7': test_precision_7, 'test/precision_8': test_precision_8, 'test/precision_9': test_precision_9, 'test/precision_10': test_precision_10},
                                global_step=epoch)
            tb_writer.add_scalars(main_tag='recall',
                                tag_scalar_dict={'train/recall': train_recall, 'test/recall_6': test_recall_6, 'test/recall_7': test_recall_7, 'test/recall_8': test_recall_8, 'test/recall_9': test_recall_9, 'test/recall_10': test_recall_10},
                                global_step=epoch)
            tb_writer.add_scalars(main_tag='f1score',
                                tag_scalar_dict={'train/f1score': train_f1score, 'test/f1score_6': test_f1score_6, 'test/f1score_7': test_f1score_7, 'test/f1score_8': test_f1score_8, 'test/f1score_9': test_f1score_9, 'test/f1score_10': test_f1score_10},
                                global_step=epoch)
            tb_writer.add_scalars(main_tag='auc',
                                tag_scalar_dict={'train/auc': train_auc, 'test/auc_6': test_auc_6, 'test/auc_7': test_auc_7, 'test/auc_8': test_auc_8, 'test/auc_9': test_auc_9, 'test/auc_10': test_auc_10},
                                global_step=epoch)
            tb_writer.add_scalars(main_tag='pred_value',
                                tag_scalar_dict={'train/value_0': train_value_0, 'train/value_1': train_value_1, 
                                                 'test/value_0_6': test_value_0_6, 'test/value_1_6': test_value_1_6,
                                                 'test/value_0_7': test_value_0_7, 'test/value_1_7': test_value_1_7,
                                                 'test/value_0_8': test_value_0_8, 'test/value_1_8': test_value_1_8,
                                                 'test/value_0_9': test_value_0_9, 'test/value_1_9': test_value_1_9,
                                                 'test/value_0_10': test_value_0_10, 'test/value_1_10': test_value_1_10},
                                global_step=epoch)  
            tb_writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)

            torch.save(model.module.state_dict(), "./weights/model-{}.pth".format(epoch))

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)

    cleanup()
 
if __name__ == '__main__': 
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.1) 
   
    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world-size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    main(opt) 
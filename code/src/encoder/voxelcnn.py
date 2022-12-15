import torch
from torch import nn  
        
class voxelcnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=2),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(64, 128, kernel_size=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(128, 256, kernel_size=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(negative_slope=0.2)
        ) 
         
    def forward(self, x): 
        out = self.feature(x.contiguous()) 
        return out
     
class gc_voxelcnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(64, 128, kernel_size=3),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(128, 256, kernel_size=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(256, 512, kernel_size=2),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(512, 1024, kernel_size=2),
            nn.BatchNorm3d(1024),
            nn.LeakyReLU(negative_slope=0.2),
        ) 
    def forward(self, x): 
        out = self.feature(x.contiguous())
        # print("out", out.shape) 
        return out

class parent_voxelcnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=7),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(64, 128, kernel_size=7),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv3d(128, 256, kernel_size=4),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(negative_slope=0.2),
        ) 

    def forward(self, x): 
        out = self.feature(x.contiguous())
        # print("out", out.shape) 
        return out

if __name__ == '__main__':
    model = voxelcnn() 
    x = torch.rand((10, 1, 4, 4, 4))
    y = model(x)
    # print(y.shape)
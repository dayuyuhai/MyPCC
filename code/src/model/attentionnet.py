import torch
from torch import nn 

class attentionnet(nn.Module):
    def __init__(self, size_in, hidden_size, latent_size) -> None:
        super().__init__()

        self.fc1 = torch.nn.Conv2d(size_in+3,hidden_size, 1)
        self.fc2 = torch.nn.Conv2d(hidden_size, hidden_size, 1)
        self.fc3 = torch.nn.Conv2d(hidden_size, latent_size, 1) 
        self.bn1 = nn.BatchNorm2d(hidden_size) 
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.bn3 = nn.BatchNorm2d(latent_size)
        self.activation = torch.nn.LeakyReLU(negative_slope=0.2)

        self.fc_weight = torch.nn.Conv2d(latent_size, 1, 1)
        self.fc_feature = torch.nn.Conv2d(latent_size, latent_size,1)
        self.short_cut = nn.Conv2d(size_in, latent_size, kernel_size=1)

    def forward(self, knn_features, knn_pos):

        x = torch.cat([knn_features, knn_pos], dim=1)

        x = self.activation(self.bn1(self.fc1(x.contiguous())))
        x = self.activation(self.bn2(self.fc2(x.contiguous())))
        x = self.activation(self.bn3(self.fc3(x.contiguous())))

        weight = self.fc_weight(x.contiguous()).squeeze(dim=1)
        weight = torch.nn.functional.softmax(weight, dim=-1)

        feature = self.fc_feature(x.contiguous()) + self.short_cut(knn_features.contiguous())

        feature = torch.matmul(weight.unsqueeze(-2), feature.permute(0,2,3,1)).squeeze(-2)

        return feature
    
  
if __name__ == '__main__':
    model = attentionnet(512, 256, 128) 






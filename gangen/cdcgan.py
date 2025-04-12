import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm
# Conditional Deep Convolutional GAN (CDCGAN) model architecture


class Discriminator(nn.Module):
    def __init__(self, n_classes=43, embedding_dim = 32):
        super().__init__()
        
        self.image_features = nn.Sequential(
                    spectral_norm(
                        nn.Conv2d(in_channels=3,
                                out_channels=16,
                                kernel_size=4,
                                stride=2, # for aggressive downsampling - and a better alternative to max pooling here     
                                padding=1 # for getting edge/border info - also keeps downsampling math simple halving here
                        ), # 32 -> 16
                    )
                    # nn.BatchNorm2d(num_features=16),
                    nn.LeakyReLU(negative_slope=0.2,
                                    inplace=True),
                    
                    spectral_norm(
                        nn.Conv2d(in_channels=16,
                                out_channels=32,
                                kernel_size=4,
                                stride=2,
                                padding=1
                        ), # 16 -> 8 == 8x8x32
                    )
                    # nn.BatchNorm2d(num_features=32),
                    nn.LeakyReLU(negative_slope=0.2,
                                    inplace=True),
                    
                    spectral_norm(
                        nn.Conv2d(in_channels=32,
                                out_channels=32,
                                kernel_size=4,
                                stride=4,
                                padding=0
                        ), # 8 -> 2 == 2x2x32
                    )
                    # nn.BatchNorm2d(num_features=32),
                    nn.LeakyReLU(negative_slope=0.2,
                                    inplace=True),
                    
                    spectral_norm(
                        nn.Conv2d(in_channels=32,
                                out_channels=32,
                                kernel_size=2,
                                stride=1,
                                padding=0
                        ), # 2 -> 1 == 1 x 32
                    )
                    # nn.BatchNorm2d(num_features=32),
                    nn.LeakyReLU(negative_slope=0.2,
                                    inplace=True),
                    nn.Flatten()
        )
        
        self.class_emb = spectral_norm(
                            nn.Embedding(
                                num_embeddings = n_classes, 
                                embedding_dim = embedding_dim
                            )
                        )
        
        self.unconditioned = spectral_norm(nn.Linear(embedding_dim, 1))
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, y):
        
        features_img = self.image_features(x)
        emb_class = self.class_emb(y)
        
        inner_prod = torch.sum(features_img * emb_class, dim=1, keepdim=True)
        
        unconditioned_term = self.unconditioned(features_img)
        
        final_logit = unconditioned_term + inner_prod
        
        return self.sigmoid(final_logit)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return
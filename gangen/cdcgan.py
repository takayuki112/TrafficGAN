import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm
# Conditional Deep Convolutional GAN (CDCGAN) model architecture


class Discriminator(nn.Module):
    def __init__(self, n_classes=43, embedding_dim = 64):
        super().__init__()
        
        self.image_features = nn.Sequential(
                    spectral_norm(
                        nn.Conv2d(in_channels=3,
                                out_channels=16,
                                kernel_size=4,
                                stride=2, # for aggressive downsampling - and a better alternative to max pooling here     
                                padding=1, # for getting edge/border info - also keeps downsampling math simple halving here
                                bias=False  #
                        ), # 32 -> 16
                    ),
                    # nn.BatchNorm2d(num_features=16),
                    nn.LeakyReLU(negative_slope=0.2,
                                    inplace=True),
                    
                    spectral_norm(
                        nn.Conv2d(in_channels=16,
                                out_channels=32,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False
                        ), # 16 -> 8 == 8x8x32
                    ),
                    # nn.BatchNorm2d(num_features=32),
                    nn.LeakyReLU(negative_slope=0.2,
                                    inplace=True),
                    
                    spectral_norm(
                        nn.Conv2d(in_channels=32,
                                out_channels=64,
                                kernel_size=4,
                                stride=2,
                                padding=1,
                                bias=False
                        ), # 8 -> 2 == 4x4x64
                    ),
                    # nn.BatchNorm2d(num_features=32),
                    nn.LeakyReLU(negative_slope=0.2,
                                    inplace=True),
                    
                    spectral_norm(
                        nn.Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=2,
                                stride=2,
                                padding=1
                        ), # 2 -> 1 == 2 x 2 x 128
                    ),
                    # nn.BatchNorm2d(num_features=32),
                    nn.LeakyReLU(negative_slope=0.2,
                                    inplace=True),
                    nn.Flatten(),
                    
                    spectral_norm(nn.Linear(512, embedding_dim))
        )
        
        self.class_emb = spectral_norm(
                            nn.Embedding(
                                num_embeddings = n_classes, 
                                embedding_dim = embedding_dim
                            )
                        )
        
        self.unconditioned = spectral_norm(nn.Linear(embedding_dim, 1))
        
        # self.sigmoid = nn.Sigmoid()  # NOTE: this or linear instead of this sigmoid with BCEWithLogitsLoss as it is more numerically stable
    
    def forward(self, x, y):
        
        features_img = self.image_features(x)
        emb_class = self.class_emb(y)
        
        inner_prod = torch.sum(features_img * emb_class, dim=1, keepdim=True)
        
        unconditioned_term = self.unconditioned(features_img)
        
        final_logit = unconditioned_term + inner_prod
        
        # return self.sigmoid(final_logit)
        
        return final_logit.view(-1, 1) # using BCEWithLogitsLoss() for numerical stability

class ConditionalBatchNorm(nn.Module):
    def __init__(self, num_features, n_classes = 43, embedding_dim = 64):
        super().__init__()
        
        self.num_features = num_features
        
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        
        self.emb = nn.Embedding(num_embeddings=n_classes, embedding_dim = embedding_dim)
        
        self.gamma_c = spectral_norm(nn.Linear(embedding_dim, num_features))
        self.beta_c = spectral_norm(nn.Linear(embedding_dim, num_features))
        
        nn.init.zeros_(self.gamma_c.weight)
        nn.init.zeros_(self.beta_c.weight)
    
    def forward(self, x, y):
        '''
        y is the class 
        '''
        
        x_norm = self.bn(x)
        class_emb = self.emb(y)
        
        gamma = 1 + self.gamma_c(class_emb)
        beta = self.beta_c(class_emb)
        
        #reshape
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        
        return gamma * x_norm + beta


class Generator(nn.Module):
    def __init__(self, noise_dim=100, n_classes=43, embedding_dim = 64):
        super().__init__()
        
        self.noise_dim = noise_dim
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        
        self.class_emb = nn.Embedding(num_embeddings=n_classes, embedding_dim=embedding_dim)
        
        initial_dim = 2
        initial_channels = 256 
               
        self.initial = nn.Sequential(
            spectral_norm(nn.Linear(noise_dim + embedding_dim, initial_dim ** 2 * initial_channels)),
            nn.Unflatten(dim=1, unflattened_size=(initial_channels, initial_dim, initial_dim))
        )
        
        # 2 x 2 x 256 -> 4 x 4 x 128
        self.upsample1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), 
            spectral_norm(
                nn.Conv2d(in_channels=initial_channels,
                        out_channels=initial_channels // 2,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False
                ), 
            ), 
            ConditionalBatchNorm(
                num_features=initial_channels // 2,
                n_classes=n_classes,
                embedding_dim=embedding_dim
            ),
            nn.ReLU(inplace=True)
        )
        
        # 4 x 4 x 128 -> 8 x 8 x 64
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), 
            spectral_norm(
                nn.Conv2d(in_channels=initial_channels // 2,
                        out_channels=initial_channels // 4,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False
                ), 
            ), 
            ConditionalBatchNorm(
                num_features=initial_channels // 4,
                n_classes=n_classes,
                embedding_dim=embedding_dim
            ),
            nn.ReLU(inplace=True)
        )
        
        # 8 x 8 x 64 -> 16 x 16 x 32
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), 
            spectral_norm(
                nn.Conv2d(in_channels=initial_channels // 4,
                        out_channels=initial_channels // 8,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False
                ), 
            ), 
            ConditionalBatchNorm(
                num_features=initial_channels // 8,
                n_classes=n_classes,
                embedding_dim=embedding_dim
            ),
            nn.ReLU(inplace=True)
        )
        
        # 16 x 16 x 32 -> 32 x 32 x 16
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'), 
            spectral_norm(
                nn.Conv2d(in_channels=initial_channels // 8,
                        out_channels=initial_channels // 16,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False
                ), 
            ), 
            ConditionalBatchNorm(
                num_features=initial_channels // 16,
                n_classes=n_classes,
                embedding_dim=embedding_dim
            ),
            nn.ReLU(inplace=True)
        )
        # 32 x 32 x 16 -> 32 x 32 x 3
        
        self.final = nn.Sequential(
            nn.Conv2d(in_channels=initial_channels // 16,
                    out_channels=3,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False
            ),
            nn.Tanh()
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if hasattr(self, 'final') and isinstance(self.final, nn.Sequential) and m is self.final[0]:
                    nn.init.normal_(m.weight, 0.0, 0.02)
                else:
                    nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  
        
    def forward(self, z, y):
        y_emb = self.class_emb(y)
        
        z_y = torch.cat([z, y_emb], dim=1)
        
        x = self.initial(z_y)
        
        x = self.upsample1(x, y)
        x = self.upsample2(x, y)
        x = self.upsample3(x, y)
        x = self.upsample4(x, y)
        x = self.final(x)
        
        return x
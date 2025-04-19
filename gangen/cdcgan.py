import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, n_classes=43, embedding_dim = 64):
        super().__init__()

        # Original channels: 3 -> 16 -> 32 -> 64 -> 128 -> embedding_dim
        # NEW Try: Reduce final channel depth
        # Example: 3 -> 16 -> 32 -> 64 -> 64 -> embedding_dim

        ch1, ch2, ch3, ch4 = 16, 32, 64, 128 # Reduced ch4 from 128

        self.image_features = nn.Sequential(
            spectral_norm(
                nn.Conv2d(3, ch1, 4, 2, 1, bias=False), # 32 -> 16
            ),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(ch1, ch2, 4, 2, 1, bias=False), # 16 -> 8
            ),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv2d(ch2, ch3, 4, 2, 1, bias=False), # 8 -> 4
            ),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                # Last Conv layer uses ch4
                nn.Conv2d(ch3, ch4, 4, 1, 0, bias=False), # 4 -> 1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            # Linear layer input dim must now match ch4
            spectral_norm(nn.Linear(ch4, embedding_dim))
        )

        self.class_emb = spectral_norm(
            nn.Embedding(n_classes, embedding_dim)
        )
        # This linear layer input also matches embedding_dim, so it's okay
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

        self.emb = nn.Embedding(num_embeddings=n_classes, embedding_dim=embedding_dim)

        self.gamma_c = spectral_norm(nn.Linear(embedding_dim, num_features))
        self.beta_c = spectral_norm(nn.Linear(embedding_dim, num_features))

        # Initialization: Start close to identity transform
        nn.init.zeros_(self.gamma_c.weight)
        nn.init.ones_(self.gamma_c.bias) # Initialize bias to 1 for gamma
        nn.init.zeros_(self.beta_c.weight)
        nn.init.zeros_(self.beta_c.bias)  # Initialize bias to 0 for beta

    def forward(self, x, y):
        '''
        y is the class
        '''

        x_norm = self.bn(x)
        class_emb = self.emb(y)

        gamma = self.gamma_c(class_emb) # Gamma gain
        beta = self.beta_c(class_emb)   # Beta shift

        #reshape
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)

        # Apply gain and shift: y = gamma * x + beta
        # Note: The original paper adds 1 to gamma (making it a scale around 1).
        # Let's stick to the original paper's formulation for Conditional BN.
        # gamma = 1 + self.gamma_c(class_emb) # Rescale gamma around 1
        # beta = self.beta_c(class_emb)
        # gamma = gamma.view(-1, self.num_features, 1, 1)
        # beta = beta.view(-1, self.num_features, 1, 1)
        # return gamma * x_norm + beta

        # Or simpler: let the linear layers learn the appropriate scale/shift directly
        return gamma * x_norm + beta


class Generator(nn.Module):
    def __init__(self, noise_dim=100, n_classes=43, embedding_dim = 64):
        super().__init__()

        self.noise_dim = noise_dim
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

        self.class_emb = nn.Embedding(num_embeddings=n_classes, embedding_dim=embedding_dim)

        initial_dim = 4
        initial_channels = 256

        ## INITIAL LAYER
        # 4 x 4 x 256
        self.initial_fc = spectral_norm(nn.Linear(noise_dim + embedding_dim, initial_dim * initial_dim * initial_channels))
        self.initial_reshape = nn.Unflatten(dim=1, unflattened_size=(initial_channels, initial_dim, initial_dim))
        self.initial_relu = nn.ReLU(inplace=True)

        ch1_out = initial_channels // 2 # 128
        ch2_out = initial_channels // 4 # 64
        ch3_out = initial_channels // 4 # 64 - Keeping last layer wider

        ## UPSAMPLE 1
        # 4 x 4 x 256 -> 8 x 8 x 128
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = spectral_norm(
            nn.Conv2d(in_channels=initial_channels,
                      out_channels=ch1_out, # 128
                      kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.cbn1 = ConditionalBatchNorm(
            num_features=ch1_out, n_classes=n_classes, embedding_dim=embedding_dim
        )
        self.relu1 = nn.ReLU(inplace=True)

        ## UPSAMPLE 2
        # 8 x 8 x 128 -> 16 x 16 x 64
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = spectral_norm(
            nn.Conv2d(in_channels=ch1_out,
                      out_channels=ch2_out, # 64
                      kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.cbn2 = ConditionalBatchNorm(
            num_features=ch2_out, n_classes=n_classes, embedding_dim=embedding_dim
        )
        self.relu2 = nn.ReLU(inplace=True)

        ## UPSAMPLE 3
        # 16 x 16 x 64 -> 32 x 32 x 64
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = spectral_norm(
            nn.Conv2d(in_channels=ch2_out,  # 64
                      out_channels=ch3_out, # 64 (Correctly set)
                      kernel_size=3, stride=1, padding=1, bias=False)
        )
        # --- FIX HERE ---
        self.cbn3 = ConditionalBatchNorm(
            num_features=ch3_out, # << Use ch3_out (64)
            n_classes=n_classes, embedding_dim=embedding_dim
        )
        self.relu3 = nn.ReLU(inplace=True)

        ## FINAL LAYER
        # 32 x 32 x 64 -> 32 x 32 x 3
        # --- FIX HERE ---
        self.final_conv = nn.Conv2d(
            in_channels=ch3_out, # << Use ch3_out (64)
            out_channels=3,
            kernel_size=3, stride=1, padding=1, bias=False)
        self.final_act = nn.Tanh()


        self._initialize_weights()
        
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, ConditionalBatchNorm)):
                 if hasattr(m, 'weight') and m.weight is not None:
                     nn.init.normal_(m.weight, 1.0, 0.02)
                 if hasattr(m, 'bias') and m.bias is not None:
                     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                 nn.init.normal_(m.weight, 0.0, 0.02) 


    def forward(self, z, y):
        y_emb = self.class_emb(y)

        z_y = torch.cat([z, y_emb], dim=1)

        x = self.initial_fc(z_y)
        x = self.initial_relu(x) 
        x = self.initial_reshape(x)

        # Upsample block 1
        x = self.up1(x)
        x = self.conv1(x)
        x = self.cbn1(x, y) 
        x = self.relu1(x)

        # Upsample block 2
        x = self.up2(x)
        x = self.conv2(x)
        x = self.cbn2(x, y) 
        x = self.relu2(x)

        # Upsample block 3
        x = self.up3(x)
        x = self.conv3(x)
        x = self.cbn3(x, y) 
        x = self.relu3(x)

        x = self.final_conv(x)
        x = self.final_act(x)

        return x

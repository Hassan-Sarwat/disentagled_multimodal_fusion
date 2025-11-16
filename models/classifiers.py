import torch
from torch import nn
import math
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.distributions import Normal, Independent
from utils import initialize_weights, activation_function


class IdentityEncoder(nn.Module):
    """Pass-through encoder that returns input as-is."""
    def forward(self, x):
        return x


class Linear(nn.Module):
    """
    Multi-layer perceptron with a flexible number of layers.

    Args:
        layers (tuple): A tuple defining the sizes of each layer, e.g., (input_dim, hidden1, hidden2, ..., last_hidden_dim).
        output_dims (int, optional): The output dimension. Defaults to 128.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        initialization (str, optional): The initialization method. Defaults to 'xavier'.
    """

    def __init__(self, dropout=0.1, output_dims=128, index=0, layers=(5, 10, 50), initialization='xavier'):
        super(Linear, self).__init__()
        self.dropout = dropout
        self.output_dims = output_dims
        self.layers = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i+1])         
            self.layers.append(linear)
            self.layers.append(nn.ReLU())
            if self.dropout > 0:
                self.layers.append(nn.Dropout(self.dropout))
        
        self.layers.append(nn.Linear(layers[-1], output_dims))
        self.layers = initialize_weights(self.layers, initialization)
    
    def forward(self, x):
        """Forward pass through the MLP."""
        output = x.float() 
        for layer in self.layers:
            output = layer(output)
        return output


# =============================================================================
# LUMA-Style Feature Encoders
# =============================================================================

class ImageEncoder(nn.Module):
    """
    CNN-based image encoder for LUMA dataset.
    
    Based on the LUMA paper architecture (Figure 11):
    - Processes 32x32 RGB images from CIFAR-10/100
    - Uses convolutional layers for feature extraction
    - Outputs fixed-dimensional embeddings
    
    Args:
        output_dim (int): Output embedding dimension. Defaults to 200.
        dropout (float): Dropout probability. Defaults to 0.1.
    """
    
    def __init__(self, output_dim=200, dropout=0.1):
        super(ImageEncoder, self).__init__()
        self.output_dim = output_dim
        
        # Convolutional feature extractor
        # Input: (batch, 3072) -> reshape to (batch, 3, 32, 32)
        self.conv_layers = nn.Sequential(
            # First conv block: 32x32 -> 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Dropout2d(dropout),
            
            # Second conv block: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Dropout2d(dropout),
            
            # Third conv block: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x4
            nn.Dropout2d(dropout),
        )
        
        # Fully connected layers
        # 128 channels * 4 * 4 = 2048
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Flattened image tensor of shape (batch, 3072)
        
        Returns:
            torch.Tensor: Encoded features of shape (batch, output_dim)
        """
        # Reshape from (batch, 3072) to (batch, 3, 32, 32)
        batch_size = x.shape[0]
        x = x.view(batch_size, 3, 32, 32)
        
        # Apply conv layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(batch_size, -1)
        
        # Apply FC layers
        x = self.fc_layers(x)
        
        return x


class AudioEncoder(nn.Module):
    """
    CNN-based audio encoder for LUMA dataset.
    
    Based on the LUMA paper architecture (Figure 12):
    - Processes MFCC features (default: 40 coefficients)
    - Can work with either 1D MFCC vectors or 2D spectrograms
    - Outputs fixed-dimensional embeddings
    
    Args:
        input_dim (int): Input MFCC dimension. Defaults to 40.
        output_dim (int): Output embedding dimension. Defaults to 200.
        dropout (float): Dropout probability. Defaults to 0.1.
        use_2d (bool): If True, treat input as 2D spectrogram. Defaults to False (1D MFCC).
    """
    
    def __init__(self, input_dim=40, output_dim=200, dropout=0.1, use_2d=False):
        super(AudioEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_2d = use_2d
        
        if use_2d:
            # For 2D spectrogram input (e.g., 128x128 mel-spectrogram)
            # This matches the LUMA paper's approach for audio
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout),
                
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(dropout),
                
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1)  # Global average pooling
            )
            
            self.fc_layers = nn.Sequential(
                nn.Linear(128, output_dim)
            )
        else:
            # For 1D MFCC input (simpler, more common approach)
            # Use MLP for MFCC features
            self.fc_layers = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, output_dim)
            )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Audio features
                - If use_2d=False: shape (batch, input_dim) - MFCC vector
                - If use_2d=True: shape (batch, H, W) - spectrogram
        
        Returns:
            torch.Tensor: Encoded features of shape (batch, output_dim)
        """
        if self.use_2d:
            # Add channel dimension if needed
            if x.dim() == 3:
                x = x.unsqueeze(1)  # (batch, 1, H, W)
            
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.fc_layers(x)
        else:
            # Direct MLP processing for MFCC
            x = self.fc_layers(x)
        
        return x


class TextEncoder(nn.Module):
    """
    MLP-based text encoder for LUMA dataset.
    
    Based on the LUMA paper architecture (Figure 13):
    - Processes pre-computed BERT embeddings (averaged token embeddings)
    - Uses feed-forward network for final encoding
    - Outputs fixed-dimensional embeddings
    
    Note: The LUMA dataset provides text as token IDs or pre-computed embeddings.
    This encoder assumes you're working with a fixed-dimensional representation
    (e.g., mean-pooled BERT embeddings of dimension 768 or token ID sequences 
    mapped to embeddings).
    
    Args:
        input_dim (int): Input dimension (e.g., 128 for token sequences, 768 for BERT embeddings).
        output_dim (int): Output embedding dimension. Defaults to 200.
        dropout (float): Dropout probability. Defaults to 0.1.
    """
    
    def __init__(self, input_dim=128, output_dim=200, dropout=0.1):
        super(TextEncoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Feed-forward network for text encoding
        # LUMA paper uses averaged BERT embeddings -> MLP
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Text features of shape (batch, input_dim)
                - Could be token IDs mapped to embeddings
                - Could be pre-computed BERT embeddings
        
        Returns:
            torch.Tensor: Encoded features of shape (batch, output_dim)
        """
        x = self.fc_layers(x)
        return x


# Alias for backward compatibility
MLP = Linear


# =============================================================================
# Probabilistic Encoders (from your original code)
# =============================================================================

class VonMisesFisher(torch.distributions.Distribution):
    """
    Von Mises-Fisher distribution for hyperspherical latent spaces.
    From the paper "An Information Criterion for Disentanglement of Multimodal Data"
    https://arxiv.org/pdf/2410.23996
    """
    arg_constraints = {
        "loc": torch.distributions.constraints.real,
        "scale": torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def stddev(self):
        return self.scale

    def __init__(self, loc, scale, validate_args=None, k=1):
        self.dtype = loc.dtype
        self.loc = loc
        self.scale = scale
        self.device = loc.device
        self.__m = loc.shape[-1]
        self.__e1 = (torch.Tensor([1.0] + [0] * (loc.shape[-1] - 1))).to(self.device)
        self.k = k

        super().__init__(self.loc.size(), validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, shape=torch.Size()):
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])

        w = (
            self.__sample_w3(shape=shape)
            if self.__m == 3
            else self.__sample_w_rej(shape=shape)
        )

        v = (
            torch.distributions.Normal(0, 1)
            .sample(shape + torch.Size(self.loc.shape))
            .to(self.device)
            .transpose(0, -1)[1:]
        ).transpose(0, -1)
        v = v / v.norm(dim=-1, keepdim=True)

        w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))
        x = torch.cat((w, w_ * v), -1)
        z = self.__householder_rotation(x)

        return z.type(self.dtype)

    def __sample_w3(self, shape):
        shape = shape + torch.Size(self.scale.shape)
        u = torch.distributions.Uniform(0, 1).sample(shape).to(self.device)
        self.__w = (
            1
            + torch.stack(
                [torch.log(u), torch.log(1 - u) - 2 * self.scale], dim=0
            ).logsumexp(0)
            / self.scale
        )
        return self.__w

    def __sample_w_rej(self, shape):
        c = torch.sqrt((4 * (self.scale ** 2)) + (self.__m - 1) ** 2)
        b_true = (-2 * self.scale + c) / (self.__m - 1)

        # using Taylor approximation with a smooth swift from 10 < scale < 11
        # to avoid numerical errors for large scale
        b_app = (self.__m - 1) / (4 * self.scale)
        s = torch.min(
            torch.max(
                torch.tensor([0.0], dtype=self.dtype, device=self.device),
                self.scale - 10,
            ),
            torch.tensor([1.0], dtype=self.dtype, device=self.device),
        )
        b = b_app * s + b_true * (1 - s)

        a = (self.__m - 1 + 2 * self.scale + c) / 4
        d = (4 * a * b) / (1 + b) - (self.__m - 1) * math.log(self.__m - 1)

        self.__b, (self.__e, self.__w) = b, self.__while_loop(b, a, d, shape, k=self.k)
        return self.__w

    @staticmethod
    def first_nonzero(x, dim, invalid_val=-1):
        mask = x > 0
        idx = torch.where(
            mask.any(dim=dim),
            mask.float().argmax(dim=1).squeeze(),
            torch.tensor(invalid_val, device=x.device),
        )
        return idx

    def __while_loop(self, b, a, d, shape, k=20, eps=1e-20):
        #  matrix while loop: samples a matrix of [A, k] samples, to avoid looping all together
        b, a, d = [
            e.repeat(*shape, *([1] * len(self.scale.shape))).reshape(-1, 1)
            for e in (b, a, d)
        ]
        w, e, bool_mask = (
            torch.zeros_like(b).to(self.device),
            torch.zeros_like(b).to(self.device),
            (torch.ones_like(b) == 1).to(self.device),
        )

        sample_shape = torch.Size([b.shape[0], k])
        shape = shape + torch.Size(self.scale.shape)

        while bool_mask.sum() != 0:
            con1 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            con2 = torch.tensor((self.__m - 1) / 2, dtype=torch.float64)
            e_ = (
                torch.distributions.Beta(con1, con2)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            u = (
                torch.distributions.Uniform(0 + eps, 1 - eps)
                .sample(sample_shape)
                .to(self.device)
                .type(self.dtype)
            )

            w_ = (1 - (1 + b) * e_) / (1 - (1 - b) * e_)
            t = (2 * a * b) / (1 - (1 - b) * e_)

            accept = ((self.__m - 1.0) * t.log() - t + d) > torch.log(u)
            accept_idx = self.first_nonzero(accept, dim=-1, invalid_val=-1).unsqueeze(1)
            accept_idx_clamped = accept_idx.clamp(0)
            # we use .abs(), in order to not get -1 index issues, the -1 is still used afterwards
            w_ = w_.gather(1, accept_idx_clamped.view(-1, 1))
            e_ = e_.gather(1, accept_idx_clamped.view(-1, 1))

            reject = accept_idx < 0
            accept = ~reject if torch.__version__ >= "1.2.0" else 1 - reject

            w[bool_mask * accept] = w_[bool_mask * accept]
            e[bool_mask * accept] = e_[bool_mask * accept]

            bool_mask[bool_mask * accept] = reject[bool_mask * accept]

        return e.reshape(shape), w.reshape(shape)

    def __householder_rotation(self, x):
        u = self.__e1 - self.loc
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z

    def _log_unnormalized_prob(self, x):
        output = self.scale * (self.loc * x).sum(-1, keepdim=True)
        return output.view(*(output.shape[:-1]))


class ProbabilisticEncoder(nn.Module):
    """
    Probabilistic encoder wrapping a neural network with a distribution.
    From the paper "An Information Criterion for Disentanglement of Multimodal Data"
    https://arxiv.org/pdf/2410.23996
    """
    def __init__(self, net, distribution='normal', vmfkappa=1):
        super().__init__()
        self.net = net
        self.distribution = distribution
        self.vmfkappa = vmfkappa

    def forward(self, x):
        params = self.net(x)
        
        if self.distribution == 'normal':  # Standard normal gaussians
            mu = params
            sigma = torch.ones(params.shape[-1]).unsqueeze(0).expand(mu.shape[0], -1).cuda()
            return Independent(Normal(mu, sigma), 1), mu  # Return a factorized Normal distribution
        elif self.distribution == 'vmf':
            loc = params / params.norm(dim=-1, keepdim=True)
            scale = self.vmfkappa * torch.ones(params.shape[0], 1).cuda()
            return VonMisesFisher(loc, scale), params


class EvidentialNN(nn.Module):
    """
    Evidential neural network for uncertainty quantification.
    Outputs parameters for Dirichlet distribution.
    
    Args:
        layers (tuple): Hidden layer dimensions.
        output_dims (int): Number of output classes/parameters.
        dropout (float): Dropout probability.
        initialization (str): Weight initialization method.
    """

    def __init__(self, dropout=0.1, output_dims=10, layers=(100, 100), initialization='xavier'):
        super(EvidentialNN, self).__init__()
        self.dropout = dropout
        self.output_dims = output_dims
        self.layers = nn.ModuleList()
        
        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i+1])
            self.layers.append(linear)
            self.layers.append(nn.ReLU())
            if self.dropout > 0:
                self.layers.append(nn.Dropout(self.dropout))
        
        self.layers.append(nn.Linear(layers[-1], output_dims))
        self.layers = initialize_weights(self.layers, initialization)
    
    def forward(self, x):
        """Forward pass through the MLP with exponential activation."""
        output = x.float() 
        for layer in self.layers:
            output = layer(output)
        return activation_function(output, 'exp')
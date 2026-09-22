import torch
import torch.nn as nn
import math


def build_denoising_network(config, env):
    """Build neural network for spectral denoising

    Args:
        config: Configuration object
        env: Environment dictionary

    Returns:
        net: DenoisingNetwork instance
        network_name: Name of the network architecture
    """
    num_features = env['spectral']['num_energy_points'] * env['spectral']['num_angle_points']
    num_hidden_units = config.network.num_hidden_units
    layer_type = config.network.layer_type
    encoder_output_dim = config.network.encoder_output_dim

    net = DenoisingNetwork(
        num_features=num_features,
        num_hidden_units=num_hidden_units,
        layer_type=layer_type,
        encoder_output_dim=encoder_output_dim
    )

    return net, layer_type


class ResidualBlock(nn.Module):
    """Residual block with skip connection for FCNN"""

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class ResidualBlock1D(nn.Module):
    """Residual block with skip connection for 1D-CNN"""

    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer

    Adds position information to input embeddings using sine and cosine
    functions of different frequencies.
    """

    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PreNormTransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm Transformer encoder layer

    Pre-LN places LayerNorm before attention and FFN, which improves
    training stability compared to Post-LN (standard Transformer).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Pre-LN: LayerNorm before attention
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        # Pre-LN: LayerNorm before FFN
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)

        return src


class SpectralTransformer(nn.Module):
    """Improved Transformer for spectral denoising

    Key improvements over vanilla Transformer:
    1. Proper positional encoding for spectral data (optional)
    2. Pre-LayerNorm for training stability
    3. Learnable input projection to appropriate d_model
    4. GELU activation in FFN
    5. Skip connection from input to output (like U-Net)

    Args:
        use_positional_encoding: If False, no positional encoding is added.
            This makes the model position-invariant (better for random peak positions).
    """

    def __init__(self, num_features, d_model=128, nhead=8, num_layers=4,
                 dim_feedforward=512, dropout=0.1, use_positional_encoding=True):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding

        # Input projection: (batch, num_features) -> (batch, seq_len, d_model)
        # Treat spectrum as sequence of patches
        self.patch_size = 8  # Each patch covers 8 energy points

        # Checked here rather than left to the first forward pass. The reshape
        # into patches cannot divide a spectrum that is not a whole number of
        # them, and the failure it produces -- "shape '[2, 31, 8]' is invalid
        # for input of size 500" -- names neither the constraint nor the
        # architecture. Raising at construction means a user learns before
        # training starts rather than after it does.
        if num_features % self.patch_size != 0:
            lower = (num_features // self.patch_size) * self.patch_size
            upper = lower + self.patch_size
            raise ValueError(
                f"the Transformer reads the spectrum as patches of "
                f"{self.patch_size} points, so num_features must be a multiple "
                f"of {self.patch_size}; got {num_features}. The nearest usable "
                f"lengths are {lower} and {upper} — resample the spectra, or "
                f"choose an architecture without this constraint (every other "
                f"one in this package accepts any length)."
            )

        self.seq_len = num_features // self.patch_size
        self.input_proj = nn.Linear(self.patch_size, d_model)

        # Positional encoding (optional)
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, max_len=self.seq_len + 1, dropout=dropout)
        else:
            self.pos_encoder = None

        # CLS token for global representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Transformer encoder with Pre-LN
        self.encoder_layers = nn.ModuleList([
            PreNormTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection: (batch, seq_len, d_model) -> (batch, num_features)
        self.output_proj = nn.Linear(d_model, self.patch_size)

        # Skip connection weight (learnable)
        self.skip_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        Args:
            x: (batch, num_features)
        Returns:
            denoised: (batch, num_features)
        """
        batch_size = x.size(0)
        residual = x  # Save for skip connection

        # Reshape to patches: (batch, num_features) -> (batch, seq_len, patch_size)
        x = x.view(batch_size, self.seq_len, self.patch_size)

        # Project to d_model: (batch, seq_len, patch_size) -> (batch, seq_len, d_model)
        x = self.input_proj(x)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, seq_len+1, d_model)

        # Add positional encoding (if enabled)
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.final_norm(x)

        # Remove CLS token and project back
        x = x[:, 1:, :]  # (batch, seq_len, d_model)
        x = self.output_proj(x)  # (batch, seq_len, patch_size)

        # Reshape back: (batch, seq_len, patch_size) -> (batch, num_features)
        x = x.view(batch_size, self.num_features)

        # Skip connection: blend with input
        x = x + self.skip_weight * residual

        return x


class TrueSequentialBiLSTM(nn.Module):
    """True sequential bi-LSTM for position-invariant spectral denoising.

    Unlike the original bi-LSTM implementation that treats the spectrum as
    a single 256-dim feature vector (seq_len=1), this processes each energy
    point as a separate time step (seq_len=256).

    Consequences of that choice:
    1. The recurrence runs along the energy axis, so what is learned is a local
       pattern rather than a mapping tied to absolute bin indices.
    2. Smaller: 133,762 parameters at num_features=256, against 579,657 for the
       seq_len=1 bi-LSTM at the panel default configuration (~4.3x fewer).
       Reproduce with sum(p.numel() for p in module.parameters()).

    No denoising-quality comparison is asserted here. This module is
    experimental and is not exposed by the command-line interface; see the
    project documentation for the supported architecture set.

    The bidirectional nature is crucial: it allows the model to see context
    from both sides of each energy point, similar to how a human would
    analyze a spectrum by looking at neighboring points.
    """

    def __init__(self, num_features, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size

        # True sequential LSTM: input_size=1 (one value per time step)
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection: hidden*2 (bidirectional) -> 1
        self.output_proj = nn.Linear(hidden_size * 2, 1)

        # Skip connection weight
        self.skip_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        """
        Args:
            x: (batch, num_features)
        Returns:
            denoised: (batch, num_features)
        """
        residual = x

        # Reshape: (batch, 256) -> (batch, 256, 1)
        x = x.unsqueeze(-1)

        # LSTM: (batch, 256, 1) -> (batch, 256, hidden*2)
        output, _ = self.lstm(x)

        # Project: (batch, 256, hidden*2) -> (batch, 256, 1)
        output = self.output_proj(output)

        # Reshape: (batch, 256, 1) -> (batch, 256)
        output = output.squeeze(-1)

        # Skip connection
        output = output + self.skip_weight * residual

        return output


class DenoisingNetwork(nn.Module):
    """Neural network for spectral denoising with multi-output

    Supports multiple architectures:
      - FCNN: Fully Connected Neural Network
      - ResNet-FCNN: FCNN with skip connections (residual blocks)
      - 1D-CNN: 1D Convolutional Neural Network
      - ResNet-1DCNN: 1D-CNN with skip connections
      - GRU: Gated Recurrent Unit
      - LSTM: Long Short-Term Memory
      - bi-LSTM: Bidirectional LSTM (legacy, fake sequential)
      - bi-LSTM-seq: True sequential bi-LSTM (experimental)
      - Transformer: Self-attention based (with positional encoding)
      - Transformer-noPE: Transformer without PE (deprecated: PE version is equivalent or better.
            Position robustness comes from jitter training, not from removing PE.)
    """

    def __init__(self, num_features, num_hidden_units=100, layer_type='FCNN',
                 encoder_output_dim=64):
        super().__init__()

        self.num_features = num_features
        self.num_hidden_units = num_hidden_units
        self.layer_type = layer_type
        self.encoder_output_dim = encoder_output_dim

        # Build encoder based on architecture type
        if layer_type == 'FCNN':
            self.encoder = self._build_fcnn()
        elif layer_type == 'ResNet-FCNN':
            self.encoder = self._build_resnet_fcnn()
        elif layer_type == '1D-CNN':
            self.encoder = self._build_1dcnn()
        elif layer_type == 'ResNet-1DCNN':
            self.encoder = self._build_resnet_1dcnn()
        elif layer_type == 'GRU':
            self.encoder = self._build_gru()
        elif layer_type == 'LSTM':
            self.encoder = self._build_lstm()
        elif layer_type == 'bi-LSTM':
            self.encoder = self._build_bilstm()
        elif layer_type == 'bi-LSTM-seq':
            self.encoder = self._build_bilstm_seq()
        elif layer_type == 'Transformer':
            self.encoder = self._build_transformer(use_pe=True)
        elif layer_type == 'Transformer-noPE':
            self.encoder = self._build_transformer(use_pe=False)
        else:
            raise ValueError(f'Unknown network type: {layer_type}')

        # Multi-output heads
        output_dim = self._get_encoder_output_dim()
        self.fc_denoise = nn.Linear(output_dim, num_features)
        self.fc_top = nn.Linear(output_dim, 1)

    def _get_encoder_output_dim(self):
        """Get output dimension of encoder based on architecture"""
        if self.layer_type in ['GRU', 'LSTM']:
            return self.num_hidden_units
        elif self.layer_type == 'bi-LSTM':
            return self.num_hidden_units * 2
        elif self.layer_type == 'bi-LSTM-seq':
            # True sequential bi-LSTM outputs num_features directly
            return self.num_features
        elif self.layer_type == 'ResNet-FCNN':
            # ResNet-FCNN outputs num_features (same as input for skip connection)
            return self.num_features
        elif self.layer_type == 'ResNet-1DCNN':
            # ResNet-1DCNN outputs num_features (same as input)
            return self.num_features
        elif self.layer_type in ['Transformer', 'Transformer-noPE']:
            # Transformer outputs num_features directly
            return self.num_features
        else:
            # FCNN, CNN use encoder_output_dim
            return self.encoder_output_dim

    def _build_fcnn(self):
        """Build Fully Connected Neural Network"""
        return nn.Sequential(
            nn.Linear(self.num_features, self.encoder_output_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_output_dim, self.encoder_output_dim // 2),
            nn.ReLU(),
            nn.Linear(self.encoder_output_dim // 2, self.encoder_output_dim),
            nn.ReLU()
        )

    def _build_resnet_fcnn(self):
        """Build FCNN with skip connections (ResNet-style)

        Architecture:
          Input (num_features)
            ↓
          Linear → num_features (projection)
            ↓
          ResidualBlock × 4
            ↓
          Output (num_features)

        The skip connections help preserve input information
        and enable better gradient flow for deeper networks.
        """
        return nn.Sequential(
            # Initial projection (keep same dimension for skip connections)
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(),
            # Residual blocks
            ResidualBlock(self.num_features, dropout=0.1),
            ResidualBlock(self.num_features, dropout=0.1),
            ResidualBlock(self.num_features, dropout=0.1),
            ResidualBlock(self.num_features, dropout=0.1),
        )

    def _build_1dcnn(self):
        """Build 1D Convolutional Neural Network"""
        return nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * self.num_features, self.encoder_output_dim),
            nn.ReLU()
        )

    def _build_resnet_1dcnn(self):
        """Build 1D-CNN with skip connections (ResNet-style)

        Architecture:
          Input (1, num_features)
            ↓
          Conv1d → 32 channels
            ↓
          ResidualBlock1D × 4
            ↓
          Conv1d → 1 channel
            ↓
          Output (num_features)

        The skip connections preserve local patterns and
        enable better gradient flow through the network.
        """
        # Store components for ResNet-1DCNN
        self.resnet_cnn = nn.ModuleDict({
            'input_conv': nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
            ),
            'res_blocks': nn.Sequential(
                ResidualBlock1D(32, kernel_size=3, dropout=0.1),
                ResidualBlock1D(32, kernel_size=3, dropout=0.1),
                ResidualBlock1D(32, kernel_size=3, dropout=0.1),
                ResidualBlock1D(32, kernel_size=3, dropout=0.1),
            ),
            'output_conv': nn.Sequential(
                nn.Conv1d(32, 1, kernel_size=3, padding=1),
            ),
        })
        # Return a dummy - actual forward is handled specially
        return None

    def _build_gru(self):
        """Build GRU network"""
        return nn.GRU(
            self.num_features, self.num_hidden_units,
            num_layers=2, batch_first=True
        )

    def _build_lstm(self):
        """Build LSTM network"""
        return nn.LSTM(
            self.num_features, self.num_hidden_units,
            num_layers=2, batch_first=True
        )

    def _build_bilstm(self):
        """Build Bidirectional LSTM network (legacy, fake sequential)

        Note: This implementation treats the spectrum as a single 256-dim
        feature vector with seq_len=1. For true sequential processing,
        use 'bi-LSTM-seq' instead.
        """
        return nn.LSTM(
            self.num_features, self.num_hidden_units,
            num_layers=2, batch_first=True, bidirectional=True
        )

    def _build_bilstm_seq(self):
        """Build True Sequential Bidirectional LSTM network

        This processes each energy point as a separate time step,
        achieving position-invariant denoising.

        Returns None - forward is handled specially via TrueSequentialBiLSTM.
        """
        self.true_seq_bilstm = TrueSequentialBiLSTM(
            num_features=self.num_features,
            hidden_size=64,  # Smaller since we have many time steps
            num_layers=2,
            dropout=0.1
        )
        return None

    def _build_transformer(self, use_pe=True):
        """Build improved Transformer-based network

        Uses SpectralTransformer with:
        - Positional encoding (optional, controlled by use_pe)
        - Pre-LayerNorm for stability
        - Patch-based input processing
        - Skip connection from input to output

        Args:
            use_pe: If True, use positional encoding (better for fixed positions).
                   If False, no PE (better for position invariance).
        """
        # Create improved Transformer
        self.spectral_transformer = SpectralTransformer(
            num_features=self.num_features,
            d_model=128,         # Compact but effective
            nhead=8,             # 8 attention heads
            num_layers=4,        # 4 layers
            dim_feedforward=512, # FFN dimension
            dropout=0.1,
            use_positional_encoding=use_pe
        )
        # Return None - forward is handled specially
        return None

    def forward(self, x, global_skip=False):
        """Forward pass

        Args:
            x: Input tensor (batch_size, sequence_length, num_features) or
               (batch_size, num_features) for non-sequential models
            global_skip: If True, add global residual: output = fc(encoder(x)) + x
                        This makes the network learn the correction (noise) rather
                        than the full clean signal. Beneficial for architectures
                        without strong internal skip connections.

        Returns:
            denoised: Denoised spectrum (batch_size, num_features)
            peak_top: Peak top prediction (batch_size, 1)
        """
        # Save input for global skip connection
        x_input = x.squeeze(1) if len(x.shape) == 3 else x

        encoded = self._encode(x)

        # Multi-output heads
        denoised = self.fc_denoise(encoded)
        peak_top = self.fc_top(encoded)

        # Global residual: network learns correction, not full signal
        if global_skip:
            denoised = denoised + x_input

        return denoised, peak_top

    def _encode(self, x):
        """Encode input through the encoder

        Args:
            x: Input tensor

        Returns:
            Encoded tensor
        """
        if self.layer_type in ['GRU', 'LSTM', 'bi-LSTM']:
            return self._encode_rnn(x)
        elif self.layer_type == 'bi-LSTM-seq':
            return self._encode_bilstm_seq(x)
        elif self.layer_type == '1D-CNN':
            return self._encode_cnn(x)
        elif self.layer_type == 'ResNet-1DCNN':
            return self._encode_resnet_cnn(x)
        elif self.layer_type in ['Transformer', 'Transformer-noPE']:
            return self._encode_transformer(x)
        else:
            return self._encode_default(x)

    def _encode_rnn(self, x):
        """Encode using RNN-based encoder (legacy fake sequential)"""
        # RNN expects (batch, seq, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        output, _ = self.encoder(x)
        # Use last time step
        return output[:, -1, :]

    def _encode_bilstm_seq(self, x):
        """Encode using true sequential bi-LSTM"""
        # TrueSequentialBiLSTM expects (batch, features)
        if len(x.shape) == 3:
            x = x.squeeze(1)  # Remove sequence dimension if present
        return self.true_seq_bilstm(x)

    def _encode_cnn(self, x):
        """Encode using CNN encoder"""
        # CNN expects (batch, channels, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
        return self.encoder(x)

    def _encode_resnet_cnn(self, x):
        """Encode using ResNet-1DCNN encoder"""
        # CNN expects (batch, channels, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension

        # Forward through ResNet-1DCNN
        h = self.resnet_cnn['input_conv'](x)
        h = self.resnet_cnn['res_blocks'](h)
        h = self.resnet_cnn['output_conv'](h)

        # Flatten: (batch, 1, features) -> (batch, features)
        return h.squeeze(1)

    def _encode_transformer(self, x):
        """Encode using improved SpectralTransformer"""
        # Transformer expects (batch, features)
        if len(x.shape) == 3:
            x = x.squeeze(1)  # Remove sequence dimension if present
        return self.spectral_transformer(x)

    def _encode_default(self, x):
        """Encode using default encoder (FCNN)"""
        # FCNN expects (batch, features)
        if len(x.shape) == 3:
            x = x.squeeze(1)  # Remove sequence dimension if present
        return self.encoder(x)

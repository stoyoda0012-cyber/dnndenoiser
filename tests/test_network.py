"""Unit tests for neural network architectures."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dnndenoiser.models.network import DenoisingNetwork, ResidualBlock, ResidualBlock1D

# All supported architectures
ALL_ARCHITECTURES = [
    'FCNN', 'ResNet-FCNN', '1D-CNN', 'ResNet-1DCNN',
    'LSTM', 'bi-LSTM', 'GRU', 'Transformer'
]


# =============================================================================
# Network Forward Pass
# =============================================================================

class TestDenoisingNetwork:
    """Test DenoisingNetwork forward pass and output format."""

    @pytest.fixture
    def sample_input(self):
        """Sample input tensor: (batch=8, features=256)."""
        return torch.randn(8, 256)

    @pytest.mark.parametrize("arch", ALL_ARCHITECTURES)
    def test_forward_shape(self, sample_input, arch):
        """All architectures should produce (denoised, peak_top) tuple."""
        model = DenoisingNetwork(
            num_features=256, num_hidden_units=64,
            layer_type=arch, encoder_output_dim=32
        )
        output = model(sample_input)

        assert isinstance(output, tuple) and len(output) == 2
        denoised, peak_top = output
        assert denoised.shape == sample_input.shape
        assert peak_top.shape == (8, 1)
        assert not torch.isnan(denoised).any()

    def test_resnet_has_residual_blocks(self):
        """ResNet-FCNN should contain ResidualBlock modules."""
        model = DenoisingNetwork(256, 256, 'ResNet-FCNN')
        has_residual = any(isinstance(m, ResidualBlock) for m in model.modules())
        assert has_residual


# =============================================================================
# Residual Blocks
# =============================================================================

class TestResidualBlocks:
    """Test residual block components."""

    @pytest.mark.parametrize("block_cls,input_shape", [
        (lambda: ResidualBlock(dim=64), (4, 64)),
        (lambda: ResidualBlock1D(channels=32, kernel_size=3), (4, 32, 64)),
    ])
    def test_shape_preservation(self, block_cls, input_shape):
        """Residual blocks should preserve input shape."""
        block = block_cls()
        x = torch.randn(*input_shape)
        y = block(x)
        assert y.shape == x.shape


# =============================================================================
# Model Save/Load
# =============================================================================

class TestModelPersistence:
    """Test model serialization."""

    def test_save_load_consistency(self, tmp_path):
        """Saved and loaded model should produce identical outputs."""
        model = DenoisingNetwork(256, 64, 'FCNN')

        # Save
        path = tmp_path / 'model.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': 'FCNN',
            'num_features': 256, 'num_hidden_units': 64
        }, path)

        # Load
        state = torch.load(path, weights_only=False)
        loaded = DenoisingNetwork(
            state['num_features'], state['num_hidden_units'], state['architecture']
        )
        loaded.load_state_dict(state['model_state_dict'])

        # Compare
        x = torch.randn(4, 256)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            y1, _ = model(x)
            y2, _ = loaded(x)
        torch.testing.assert_close(y1, y2)


# =============================================================================
# Device Compatibility
# =============================================================================

class TestDeviceCompatibility:
    """Test CPU/GPU compatibility."""

    def test_cpu_inference(self):
        """Model should work on CPU."""
        model = DenoisingNetwork(256, 64, 'FCNN').cpu()
        x = torch.randn(4, 256).cpu()
        denoised, _ = model(x)
        assert denoised.device.type == 'cpu'

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
    def test_mps_inference(self):
        """Model should work on MPS (Apple Silicon)."""
        device = torch.device('mps')
        model = DenoisingNetwork(256, 64, 'FCNN').to(device)
        x = torch.randn(4, 256).to(device)
        denoised, _ = model(x)
        assert denoised.device.type == 'mps'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    @pytest.mark.parametrize("arch", ALL_ARCHITECTURES)
    def test_cuda_inference(self, arch):
        """Model should work on CUDA for every architecture."""
        device = torch.device('cuda')
        model = DenoisingNetwork(256, 64, arch, encoder_output_dim=32).to(device)
        x = torch.randn(4, 256).to(device)
        denoised, peak_top = model(x)
        assert denoised.device.type == 'cuda'
        assert denoised.shape == x.shape
        assert peak_top.shape == (4, 1)
        assert not torch.isnan(denoised).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    @pytest.mark.parametrize("arch", ALL_ARCHITECTURES)
    def test_cuda_cpu_parity(self, arch):
        """CPU and CUDA outputs should agree in fp32.

        cuDNN runs RNN kernels in TF32 by default (allow_tf32=True), which
        widens the CPU/CUDA gap to ~3e-4 relative. Disable it here so parity
        checks true fp32.
        """
        prev = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            torch.manual_seed(42)
            model = DenoisingNetwork(256, 64, arch, encoder_output_dim=32).eval()
            x = torch.randn(16, 256)
            with torch.no_grad():
                y_cpu, _ = model(x)
                y_gpu, _ = model.cuda()(x.cuda())
            torch.testing.assert_close(y_cpu, y_gpu.cpu(), rtol=1e-4, atol=1e-5)
        finally:
            torch.backends.cudnn.allow_tf32 = prev


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

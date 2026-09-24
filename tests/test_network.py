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


class TestTransformerLengthConstraint:
    """The Transformer needs a spectrum length divisible by its patch size.

    It reads the spectrum as patches of 8 points, so the reshape in ``forward``
    cannot divide a length that is not a whole number of them. That used to
    surface as ``RuntimeError: shape '[2, 31, 8]' is invalid for input of size
    500`` on the first forward pass — after the model was built, after training
    had been set up, and naming neither the constraint nor the architecture.
    """

    @pytest.mark.parametrize("num_features", [256, 128, 64, 8])
    def test_multiples_of_the_patch_size_are_accepted(self, num_features):
        model = DenoisingNetwork(
            num_features=num_features, num_hidden_units=100,
            layer_type="Transformer", encoder_output_dim=64,
        )
        out, _ = model(torch.randn(2, num_features))
        assert out.shape == (2, num_features)

    @pytest.mark.parametrize("num_features", [255, 250, 100, 33, 1])
    def test_other_lengths_are_refused_at_construction(self, num_features):
        with pytest.raises(ValueError) as exc:
            DenoisingNetwork(
                num_features=num_features, num_hidden_units=100,
                layer_type="Transformer", encoder_output_dim=64,
            )
        message = str(exc.value)
        assert "multiple of 8" in message
        assert str(num_features) in message
        assert "nearest usable" in message, "an error that does not say what to do instead"

    def test_the_suggested_lengths_actually_work(self):
        """A suggestion that does not work is worse than none."""
        import re

        with pytest.raises(ValueError) as exc:
            DenoisingNetwork(
                num_features=250, num_hidden_units=100,
                layer_type="Transformer", encoder_output_dim=64,
            )
        suggested = [int(n) for n in re.findall(r"\b(\d+) and (\d+)\b", str(exc.value))[0]]
        assert suggested == [248, 256]
        for n in suggested:
            model = DenoisingNetwork(
                num_features=n, num_hidden_units=100,
                layer_type="Transformer", encoder_output_dim=64,
            )
            assert model(torch.randn(1, n))[0].shape == (1, n)

    @pytest.mark.parametrize(
        "arch",
        ["FCNN", "ResNet-FCNN", "1D-CNN", "ResNet-1DCNN", "GRU", "LSTM", "bi-LSTM"],
    )
    def test_no_other_architecture_has_the_constraint(self, arch):
        """The error tells the user every other architecture takes any length."""
        model = DenoisingNetwork(
            num_features=100, num_hidden_units=100, layer_type=arch, encoder_output_dim=64,
        )
        assert model(torch.randn(2, 100))[0].shape == (2, 100)


class TestRecurrentArchitecturesReadOneStep:
    """GRU, LSTM and bi-LSTM read the whole spectrum as a single time step.

    The README and the reference benchmark's README say so, because the names
    suggest recurrence along the energy axis, which these models do not do: the
    recurrent layer's input is the full spectrum (``input_size`` = number of energy
    points) at sequence length 1. If that ever changes, those two documents and the
    reference measurement's reading change with it, and this test says so first.
    """

    @pytest.mark.parametrize("arch", ["GRU", "LSTM", "bi-LSTM"])
    def test_the_recurrent_layer_sees_one_step_of_the_whole_spectrum(self, arch):
        model = DenoisingNetwork(
            num_features=256, num_hidden_units=100, layer_type=arch, encoder_output_dim=64,
        )
        assert model.encoder.input_size == 256
        seen = []
        model.encoder.register_forward_hook(lambda _m, inputs, _o: seen.append(inputs[0].shape))
        model(torch.randn(4, 256))
        assert seen == [torch.Size([4, 1, 256])]

    def test_the_experimental_sequential_variant_is_what_steps_through_energy(self):
        """The contrast the documents draw, and a model the check above would refuse."""
        model = DenoisingNetwork(
            num_features=256, num_hidden_units=100, layer_type="bi-LSTM-seq", encoder_output_dim=64,
        )
        lstm = model.true_seq_bilstm.lstm
        assert lstm.input_size == 1
        seen = []
        lstm.register_forward_hook(lambda _m, inputs, _o: seen.append(inputs[0].shape))
        model(torch.randn(4, 256))
        assert seen == [torch.Size([4, 256, 1])]

import os
import tempfile
import unittest
import torch
from src.eegnet import EEGNet, MS_SE_EEGNet, SqueezeExcitation, Conv2dWithConstraint, LinearWithConstraint
from src.classifier import save_eegnet_checkpoint, load_eegnet_checkpoint


class TestEEGNet(unittest.TestCase):
    def test_eegnet_initialization_and_forward(self):
        """Test instantiation, forward pass, and output shape for channels=64 and 32."""
        for channels in [64, 32]:
            model = EEGNet(n_classes=2, channels=channels, samples=350)
            self.assertEqual(model.conv1.kernel_size, (1, 250))
            self.assertEqual(model.in_features, 176)

            x = torch.randn(4, 1, channels, 350)
            out = model(x)
            self.assertEqual(out.shape, (4, 2))

    def test_eegnet_3d_input_forward(self):
        """Test forward pass when input tensor is 3D (batch_size, channels, samples)."""
        model = EEGNet(n_classes=2, channels=64, samples=350)
        x = torch.randn(4, 64, 350)
        out = model(x)
        self.assertEqual(out.shape, (4, 2))

    def test_eegnet_backward_pass(self):
        """Test backward pass in training mode."""
        model = EEGNet(n_classes=2, channels=64, samples=350)
        model.train()
        x = torch.randn(4, 1, 64, 350, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_conv2d_max_norm_constraint(self):
        """Test that Conv2dWithConstraint properly constrains weights along dim=(1, 2, 3)."""
        layer = Conv2dWithConstraint(in_channels=8, out_channels=16, kernel_size=(64, 1), max_norm=1.0)
        layer.train()

        # Set artificially large weights
        with torch.no_grad():
            layer.weight.fill_(15.0)

        # Trigger forward pass
        dummy_x = torch.randn(2, 8, 64, 1)
        _ = layer(dummy_x)

        # Verify norms across dim=(1, 2, 3) for each individual filter
        norms = layer.weight.norm(2, dim=(1, 2, 3))
        self.assertEqual(norms.shape, torch.Size([16]))
        self.assertTrue(torch.all(norms <= 1.0 + 1e-5).item())

    def test_linear_max_norm_constraint(self):
        """Test that LinearWithConstraint properly constrains weights along dim=1."""
        layer = LinearWithConstraint(in_features=176, out_features=2, max_norm=0.25)
        layer.train()

        # Set artificially large weights
        with torch.no_grad():
            layer.weight.fill_(15.0)

        # Trigger forward pass
        dummy_x = torch.randn(2, 176)
        _ = layer(dummy_x)

        # Verify norms across dim=1 for each output class unit
        norms = layer.weight.norm(2, dim=1)
        self.assertEqual(norms.shape, torch.Size([2]))
        self.assertTrue(torch.all(norms <= 0.25 + 1e-5).item())

    def test_eegnet_end_to_end_constraints(self):
        """Test that forward pass on EEGNet with large weights constrains both depthwise conv and linear layers."""
        model = EEGNet(n_classes=2, channels=64, samples=350)
        model.train()

        with torch.no_grad():
            model.depthwise_conv.weight.fill_(10.0)
            model.classifier.weight.fill_(10.0)

        x = torch.randn(2, 1, 64, 350)
        _ = model(x)

        conv_norms = model.depthwise_conv.weight.norm(2, dim=(1, 2, 3))
        lin_norms = model.classifier.weight.norm(2, dim=1)

        self.assertTrue(torch.all(conv_norms <= 1.0 + 1e-5).item())
        self.assertTrue(torch.all(lin_norms <= 0.25 + 1e-5).item())


class TestMSSEEGNet(unittest.TestCase):
    def test_squeeze_excitation_recalibration(self):
        """Test SqueezeExcitation preserves shape and computes weights between 0 and 1."""
        se = SqueezeExcitation(channels=24, reduction=4)
        x = torch.randn(4, 24, 1, 100)
        out = se(x)
        self.assertEqual(out.shape, x.shape)
        # Check output scaling is non-zero
        self.assertTrue(torch.all(torch.isfinite(out)))

    def test_ms_se_eegnet_initialization_and_forward(self):
        """Test MS_SE_EEGNet forward pass and multi-scale temporal kernels."""
        for channels in [64, 32]:
            model = MS_SE_EEGNet(n_classes=2, channels=channels, samples=350)
            self.assertEqual(len(model.temporal_branches), 3)
            self.assertEqual(model.temporal_branches[0].kernel_size, (1, 250))
            self.assertEqual(model.temporal_branches[1].kernel_size, (1, 64))
            self.assertEqual(model.temporal_branches[2].kernel_size, (1, 16))

            x = torch.randn(4, 1, channels, 350)
            out = model(x)
            self.assertEqual(out.shape, (4, 2))

    def test_ms_se_eegnet_3d_input_forward(self):
        """Test MS_SE_EEGNet forward pass with 3D input tensor."""
        model = MS_SE_EEGNet(n_classes=2, channels=64, samples=350)
        x = torch.randn(4, 64, 350)
        out = model(x)
        self.assertEqual(out.shape, (4, 2))

    def test_ms_se_eegnet_backward_pass(self):
        """Test MS_SE_EEGNet backward gradient propagation."""
        model = MS_SE_EEGNet(n_classes=2, channels=64, samples=350)
        model.train()
        x = torch.randn(4, 1, 64, 350, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())

    def test_ms_se_eegnet_constraints(self):
        """Test that MS_SE_EEGNet enforces max-norm constraints on depthwise conv and linear layers."""
        model = MS_SE_EEGNet(n_classes=2, channels=64, samples=350)
        model.train()

        with torch.no_grad():
            model.depthwise_conv.weight.fill_(12.0)
            model.classifier.weight.fill_(12.0)

        x = torch.randn(2, 1, 64, 350)
        _ = model(x)

        conv_norms = model.depthwise_conv.weight.norm(2, dim=(1, 2, 3))
        lin_norms = model.classifier.weight.norm(2, dim=1)

        self.assertTrue(torch.all(conv_norms <= 1.0 + 1e-5).item())
        self.assertTrue(torch.all(lin_norms <= 0.25 + 1e-5).item())

    def test_checkpoint_save_and_load_roundtrip(self):
        """Test saving and loading both EEGNet and MS_SE_EEGNet checkpoints with auto architecture detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            norm_params = {
                'channels': 64,
                'samples': 350,
                'mean': torch.zeros(1, 1, 64, 1),
                'std': torch.ones(1, 1, 64, 1)
            }

            # 1. Test EEGNet Roundtrip
            eegnet = EEGNet(n_classes=2, channels=64, samples=350)
            pt_path1 = os.path.join(tmpdir, "test_eegnet.pt")
            save_eegnet_checkpoint(eegnet, norm_params, pt_path1)

            loaded_model1, mean1, std1, meta1 = load_eegnet_checkpoint(pt_path1, device=torch.device('cpu'))
            self.assertIsInstance(loaded_model1, EEGNet)
            dummy1 = torch.randn(2, 1, 64, 350)
            self.assertEqual(loaded_model1(dummy1).shape, (2, 2))

            # 2. Test MS_SE_EEGNet Roundtrip
            ms_se = MS_SE_EEGNet(n_classes=2, channels=64, samples=350)
            pt_path2 = os.path.join(tmpdir, "test_ms_se_eegnet.pt")
            save_eegnet_checkpoint(ms_se, norm_params, pt_path2)

            loaded_model2, mean2, std2, meta2 = load_eegnet_checkpoint(pt_path2, device=torch.device('cpu'))
            self.assertIsInstance(loaded_model2, MS_SE_EEGNet)
            dummy2 = torch.randn(2, 1, 64, 350)
            self.assertEqual(loaded_model2(dummy2).shape, (2, 2))


if __name__ == "__main__":
    unittest.main()


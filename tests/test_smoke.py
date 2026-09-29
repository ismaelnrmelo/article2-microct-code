"""Bounded deterministic CPU checks; no training or neural inference."""
import unittest
from types import SimpleNamespace
import numpy as np
from microct.noise import flatten_quadratic, ResidualNoise, mixed_noise
from microct.inference import normalize, assign_labels
from microct.__main__ import require_cuda

class SmokeTests(unittest.TestCase):
    def test_quadratic_surface(self):
        y,x=np.mgrid[-1:1:32j,-1:1:40j]
        surface=(17+2*x-3*y+4*x*x-5*x*y+6*y*y).astype(np.float64)
        residual=flatten_quadratic(np.stack([surface,surface+23]),stride=8)
        self.assertLess(float(np.abs(residual).max()),1e-10)
    def test_normalization(self):
        anchor=0.30599761684099036
        values=np.array([45820.0,45820.0-2887.0])
        np.testing.assert_array_equal(normalize(values),np.array([-2*anchor,2*(1-anchor)]))
    def test_label_precedence(self):
        c=np.array([0,1,1,1],bool);n=np.array([0,0,1,1],bool);p=np.array([0,0,0,1],bool)
        np.testing.assert_array_equal(assign_labels(c,n,p),np.array([2,3,4,1],np.uint8))
    def test_repeatable_residual_and_mixed_noise(self):
        psd=np.ones((16,16,16),np.float64);psd[0,0,0]=0
        residual=ResidualNoise(psd)
        a=mixed_noise((4,16,20),np.random.default_rng(123),residual)
        b=mixed_noise((4,16,20),np.random.default_rng(123),residual)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(4,16,20));self.assertTrue(np.isfinite(a).all())
    def test_cuda_guard(self):
        cpu=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        with self.assertRaisesRegex(RuntimeError,"CUDA is required"):require_cuda(cpu)
        no_bf16=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,is_bf16_supported=lambda:False))
        with self.assertRaisesRegex(RuntimeError,"bfloat16"):require_cuda(no_bf16)

if __name__ == "__main__":unittest.main()

"""LIF 示例的单元测试。

运行：
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

from lif_neuron import (
    LIFParams,
    firing_rate_hz,
    simulate_lif,
    theoretical_fi,
)


class TestLIFParams(unittest.TestCase):
    def test_rheobase(self):
        p = LIFParams()
        self.assertAlmostEqual(p.rheobase_na, 1.5)

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            LIFParams(tau_m=0)
        with self.assertRaises(ValueError):
            LIFParams(r_m=-1)
        with self.assertRaises(ValueError):
            LIFParams(v_reset=-50)  # reset 不能高于阈值
        with self.assertRaises(ValueError):
            LIFParams(e_leak=-50)   # 静息电位不能高于阈值
        with self.assertRaises(ValueError):
            LIFParams(t_refractory=-1)


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.p = LIFParams()

    def test_result_shapes(self):
        r = simulate_lif(self.p, 2.0, duration_ms=100.0, dt_ms=0.05)
        self.assertEqual(r.t_ms.shape, r.v_mv.shape)
        self.assertEqual(r.t_ms.size, 2000)
        self.assertEqual(r.spikes_ms.ndim, 1)
        self.assertEqual(r.spike_indices.ndim, 1)
        self.assertTrue(np.all(np.diff(r.spikes_ms) > 0))
        for t_spike, idx in zip(r.spikes_ms, r.spike_indices):
            self.assertAlmostEqual(r.t_ms[idx], t_spike)

    def test_invalid_duration(self):
        with self.assertRaises(ValueError):
            simulate_lif(self.p, 2.0, duration_ms=0.01, dt_ms=0.05)
        with self.assertRaises(ValueError):
            simulate_lif(self.p, 2.0, duration_ms=100.0, dt_ms=0)

    def test_subthreshold_never_spikes(self):
        # 1.5 nA 正好等于基强度：V 渐近逼近阈值但不应发放
        for current in (0.0, 1.2, 1.5):
            r = simulate_lif(self.p, current, duration_ms=2000.0)
            self.assertEqual(r.spikes_ms.size, 0, msg=f"I={current} nA")
            self.assertLessEqual(r.v_mv.max(),
                                 self.p.v_threshold + 1e-9)

    def test_supra_threshold_spikes(self):
        r = simulate_lif(self.p, 2.0, duration_ms=500.0)
        self.assertGreater(r.spikes_ms.size, 5)
        self.assertLessEqual(r.spikes_ms[0], 100.0)

    def test_reset_after_spike(self):
        r = simulate_lif(self.p, 2.5, duration_ms=100.0)
        self.assertGreater(r.spikes_ms.size, 0)
        idx = int(r.spike_indices[0])
        self.assertAlmostEqual(r.v_mv[idx], self.p.v_threshold)
        self.assertAlmostEqual(r.v_mv[idx + 1], self.p.v_reset)

    def test_refractory_is_clamped(self):
        r = simulate_lif(self.p, 2.5, duration_ms=100.0)
        for t_spike in r.spikes_ms:
            mask = (
                (r.t_ms > t_spike + 1e-9)
                & (r.t_ms < t_spike + self.p.t_refractory - 1e-9)
            )
            np.testing.assert_allclose(
                r.v_mv[mask], self.p.v_reset,
                err_msg=f"spike at {t_spike} ms",
            )

    def test_custom_reset_and_refractory(self):
        p = LIFParams(v_reset=-65.0, t_refractory=5.0)
        r = simulate_lif(p, 3.0, duration_ms=300.0)
        self.assertGreater(r.spikes_ms.size, 3)
        for t_spike in r.spikes_ms:
            mask = (
                (r.t_ms > t_spike + 1e-9)
                & (r.t_ms < t_spike + p.t_refractory - 1e-9)
            )
            np.testing.assert_allclose(r.v_mv[mask], p.v_reset)


class TestFiringRate(unittest.TestCase):
    def test_matches_analytic_solution(self):
        cases = [
            (10.0, -55.0, 2.0),
            (20.0, -55.0, 2.0),
            (40.0, -55.0, 2.0),
            (20.0, -50.0, 3.0),
            (20.0, -45.0, 4.0),
            (20.0, -55.0, 5.0),
        ]
        for tau, vth, current in cases:
            p = LIFParams(tau_m=tau, v_threshold=vth)
            r = simulate_lif(p, current, duration_ms=1200.0)
            f_sim = firing_rate_hz(r, discard_ms=200.0)
            f_theory = float(theoretical_fi(current, p)[0])
            rel_error = abs(f_sim - f_theory) / f_theory
            self.assertLess(
                rel_error, 0.01,
                msg=f"tau={tau}, Vth={vth}, I={current}: "
                    f"sim={f_sim:.3f}, theory={f_theory:.3f}",
            )

    def test_rate_increases_with_current(self):
        currents = np.array([1.55, 1.8, 2.5, 4.0, 5.0])
        rates = theoretical_fi(currents, LIFParams())
        self.assertTrue(np.all(np.diff(rates) > 0))

    def test_zero_rate_below_rheobase(self):
        currents = np.array([0.0, 0.5, 1.0, 1.5])
        np.testing.assert_allclose(
            theoretical_fi(currents, LIFParams()), 0.0)

    def test_dt_convergence(self):
        p = LIFParams()
        f_coarse = firing_rate_hz(
            simulate_lif(p, 3.0, duration_ms=1000.0, dt_ms=0.1))
        f_fine = firing_rate_hz(
            simulate_lif(p, 3.0, duration_ms=1000.0, dt_ms=0.025))
        self.assertLess(abs(f_coarse - f_fine) / f_fine, 0.01)


if __name__ == "__main__":
    unittest.main()

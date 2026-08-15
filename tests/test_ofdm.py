"""
Unit Tests for Modular OFDM Package
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python_src.modulation import Modulator
from python_src.framing import OFDMFramer
from python_src.transceiver import OFDMTransceiver

class TestModularOFDM(unittest.TestCase):

    def test_bpsk_mapping(self):
        mod = Modulator(scheme='BPSK')
        bits = np.array([0, 1, 0, 1])
        syms = mod.map_bits_to_symbols(bits)
        np.testing.assert_array_almost_equal(syms, [1.0+0j, -1.0+0j, 1.0+0j, -1.0+0j])
        demod_bits = mod.demodulate_symbols_to_bits(syms)
        np.testing.assert_array_equal(bits, demod_bits)

    def test_qpsk_mapping(self):
        mod = Modulator(scheme='QPSK')
        bits = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        syms = mod.map_bits_to_symbols(bits)
        demod_bits = mod.demodulate_symbols_to_bits(syms)
        np.testing.assert_array_equal(bits, demod_bits)

    def test_transceiver_noiseless(self):
        tx = OFDMTransceiver(n_fft=64, cp_length=16, modulation='QPSK', channel_type='AWGN')
        res = tx.run_pipeline(n_ofdm_symbols=20, snr_db=100.0, est_method='LS', eq_method='ZF')
        self.assertEqual(res['bit_errors'], 0)
        self.assertEqual(res['ber'], 0.0)

    def test_rayleigh_equalization(self):
        tx = OFDMTransceiver(n_fft=64, cp_length=16, modulation='QPSK', channel_type='RAYLEIGH')
        res = tx.run_pipeline(n_ofdm_symbols=100, snr_db=25.0, est_method='LS', eq_method='ZF')
        self.assertLess(res['ber'], 0.05)

if __name__ == '__main__':
    unittest.main()

"""
Unit Tests for OFDM Engine
Tests modulation, subcarrier allocation, IFFT/FFT noiseless loopback, channel estimation, and BER calculations.
"""

import unittest
import numpy as np
from ofdm_engine import OFDMTransceiver

class TestOFDMTransceiver(unittest.TestCase):

    def test_bpsk_noiseless_loopback(self):
        """Test BPSK OFDM transmission and reception in noiseless AWGN channel."""
        transceiver = OFDMTransceiver(n_fft=64, cp_length=16, modulation='BPSK', pilot_spacing=4)
        n_symbols = 10
        
        # 1. Bits generation
        tx_bits = transceiver.generate_bits(n_symbols)
        
        # 2. Map to symbols
        tx_symbols = transceiver.map_bits_to_symbols(tx_bits).reshape(n_symbols, transceiver.n_data)
        
        # 3. Transmit (IFFT + CP)
        tx_signal, _ = transceiver.transmit(tx_symbols)
        
        # 4. Noiseless AWGN channel (SNR = 100 dB)
        taps = np.array([1.0])
        rx_signal, actual_h, _ = transceiver.pass_channel(tx_signal, taps, snr_db=100.0, channel_type='AWGN')
        
        # 5. Receive (CP remove + FFT)
        rx_freq_frames = transceiver.receive(rx_signal, n_symbols)
        
        # 6. Channel Estimation & Equalization
        h_est = transceiver.estimate_channel(rx_freq_frames, method='LS')
        eq_data_symbols, _ = transceiver.equalize(rx_freq_frames, h_est, method='ZF', snr_db=100.0)
        
        # 7. Demodulate
        rx_bits = transceiver.demodulate_symbols_to_bits(eq_data_symbols.ravel())
        
        # 8. Check 0 bit errors
        ber, bit_errors = transceiver.compute_ber(tx_bits, rx_bits)
        self.assertEqual(bit_errors, 0, f"Expected 0 bit errors in noiseless loopback, got {bit_errors}")
        self.assertEqual(ber, 0.0)

    def test_qpsk_noiseless_loopback(self):
        """Test QPSK OFDM transmission and reception in noiseless AWGN channel."""
        transceiver = OFDMTransceiver(n_fft=64, cp_length=16, modulation='QPSK', pilot_spacing=4)
        n_symbols = 10
        
        tx_bits = transceiver.generate_bits(n_symbols)
        tx_symbols = transceiver.map_bits_to_symbols(tx_bits).reshape(n_symbols, transceiver.n_data)
        tx_signal, _ = transceiver.transmit(tx_symbols)
        
        taps = np.array([1.0])
        rx_signal, _, _ = transceiver.pass_channel(tx_signal, taps, snr_db=100.0, channel_type='AWGN')
        rx_freq_frames = transceiver.receive(rx_signal, n_symbols)
        
        h_est = transceiver.estimate_channel(rx_freq_frames, method='LS')
        eq_data_symbols, _ = transceiver.equalize(rx_freq_frames, h_est, method='ZF', snr_db=100.0)
        
        rx_bits = transceiver.demodulate_symbols_to_bits(eq_data_symbols.ravel())
        ber, bit_errors = transceiver.compute_ber(tx_bits, rx_bits)
        self.assertEqual(bit_errors, 0, f"Expected 0 bit errors for QPSK, got {bit_errors}")

    def test_rayleigh_fading_equalization(self):
        """Test that LS estimation + ZF equalization reduces BER in Rayleigh fading channel."""
        transceiver = OFDMTransceiver(n_fft=64, cp_length=16, modulation='QPSK', pilot_spacing=4)
        n_symbols = 100
        
        tx_bits = transceiver.generate_bits(n_symbols)
        tx_symbols = transceiver.map_bits_to_symbols(tx_bits).reshape(n_symbols, transceiver.n_data)
        tx_signal, _ = transceiver.transmit(tx_symbols)
        
        taps = transceiver.generate_rayleigh_channel()
        rx_signal, _, noise_pwr = transceiver.pass_channel(tx_signal, taps, snr_db=25.0, channel_type='RAYLEIGH')
        rx_freq_frames = transceiver.receive(rx_signal, n_symbols)
        
        # LS Estimation + ZF Equalization
        h_est = transceiver.estimate_channel(rx_freq_frames, method='LS')
        eq_data_symbols, _ = transceiver.equalize(rx_freq_frames, h_est, method='ZF', snr_db=25.0)
        
        rx_bits = transceiver.demodulate_symbols_to_bits(eq_data_symbols.ravel())
        ber, _ = transceiver.compute_ber(tx_bits, rx_bits)
        
        # With high SNR (25dB), BER after equalization should be < 0.05
        self.assertLess(ber, 0.05, f"BER after equalization too high: {ber}")

if __name__ == '__main__':
    unittest.main()

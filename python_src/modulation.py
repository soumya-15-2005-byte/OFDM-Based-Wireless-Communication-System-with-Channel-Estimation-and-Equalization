"""
Modulation Module - BPSK & QPSK Mapping & Demodulation
"""

import numpy as np

class Modulator:
    def __init__(self, scheme='QPSK'):
        self.scheme = scheme.upper()
        if self.scheme not in ['BPSK', 'QPSK']:
            raise ValueError(f"Unsupported modulation scheme: {scheme}")
        self.bits_per_symbol = 1 if self.scheme == 'BPSK' else 2

    def generate_bits(self, num_bits):
        """Generate random binary bits array (0 or 1)."""
        return np.random.randint(0, 2, num_bits)

    def map_bits_to_symbols(self, bits):
        """
        Map binary bits to complex constellation symbols.
        - BPSK: 0 -> +1, 1 -> -1
        - QPSK: Gray coding: 00 -> (1+j)/sqrt(2), 01 -> (-1+j)/sqrt(2), 11 -> (-1-j)/sqrt(2), 10 -> (1-j)/sqrt(2)
        """
        if self.scheme == 'BPSK':
            symbols = 1.0 - 2.0 * bits.astype(float)
            return symbols.astype(complex)
        else: # QPSK
            bits_reshaped = bits.reshape(-1, 2)
            i_val = 1.0 - 2.0 * bits_reshaped[:, 0]
            q_val = 1.0 - 2.0 * bits_reshaped[:, 1]
            symbols = (i_val + 1j * q_val) / np.sqrt(2.0)
            return symbols

    def demodulate_symbols_to_bits(self, symbols):
        """Hard-decision slicer demodulating complex symbols back to bits."""
        if self.scheme == 'BPSK':
            return (np.real(symbols) < 0.0).astype(int)
        else: # QPSK
            bits_i = (np.real(symbols) < 0.0).astype(int)
            bits_q = (np.imag(symbols) < 0.0).astype(int)
            return np.column_stack((bits_i, bits_q)).ravel()

    @staticmethod
    def compute_ber(tx_bits, rx_bits):
        """Compute Bit Error Rate (BER) and error count."""
        min_len = min(len(tx_bits), len(rx_bits))
        bit_errors = np.sum(tx_bits[:min_len] != rx_bits[:min_len])
        ber = bit_errors / min_len
        return ber, bit_errors

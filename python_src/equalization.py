"""
Channel Equalization Module - Zero-Forcing (ZF) & MMSE Equalizers
"""

import numpy as np

class ChannelEqualizer:
    @staticmethod
    def equalize(rx_freq_frames, H_est, data_indices, n_fft, method='ZF', snr_db=20):
        """
        Channel Equalization.
        Methods:
        --------
        - 'ZF': Zero Forcing equalizer X_hat = Y / H_est
        - 'MMSE': MMSE equalizer X_hat = Y * H_est* / (|H_est|^2 + 1/SNR)
        """
        snr_linear = 10.0 ** (snr_db / 10.0)
        n_symbols = len(rx_freq_frames)

        if method.upper() == 'ZF':
            H_safe = np.where(np.abs(H_est) < 1e-12, 1e-12, H_est)
            equalized_freq = rx_freq_frames / H_safe
        elif method.upper() == 'MMSE':
            denom = np.abs(H_est) ** 2 + (1.0 / snr_linear)
            equalized_freq = rx_freq_frames * np.conj(H_est) / denom
        else:
            raise ValueError(f"Unknown equalization method: {method}")

        # Extract data subcarriers
        equalized_data_symbols = equalized_freq[:, data_indices]
        return equalized_data_symbols, equalized_freq

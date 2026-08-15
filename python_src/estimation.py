"""
Channel Estimation Module - LS & MMSE Channel Estimators
"""

import numpy as np
from scipy import interpolate

class ChannelEstimator:
    @staticmethod
    def estimate(rx_freq_frames, pilot_indices, pilot_value, n_fft, method='LS', noise_power=1e-3):
        """
        Estimate Channel Frequency Response H(f) across all subcarriers.
        Methods:
        --------
        - 'LS': Least Squares estimation at pilot locations + Linear interpolation.
        - 'MMSE': Minimum Mean Square Error estimation with noise smoothing filter.
        """
        n_symbols = len(rx_freq_frames)
        H_est = np.zeros((n_symbols, n_fft), dtype=complex)

        for i in range(n_symbols):
            rx_pilots = rx_freq_frames[i, pilot_indices]
            
            # LS estimation at pilot subcarriers: H_ls_p = Y_p / X_p
            H_pilots_ls = rx_pilots / pilot_value

            # Interpolate over all subcarriers
            H_p_real = np.real(H_pilots_ls)
            H_p_imag = np.imag(H_pilots_ls)

            f_real = interpolate.interp1d(pilot_indices, H_p_real, kind='linear', fill_value='extrapolate')
            f_imag = interpolate.interp1d(pilot_indices, H_p_imag, kind='linear', fill_value='extrapolate')

            all_idx = np.arange(n_fft)
            H_ls_interp = f_real(all_idx) + 1j * f_imag(all_idx)

            if method.upper() == 'LS':
                H_est[i, :] = H_ls_interp
            elif method.upper() == 'MMSE':
                # Frequency-domain MMSE smoothing filter
                mag_sq = np.abs(H_ls_interp) ** 2
                H_est[i, :] = H_ls_interp * (mag_sq / (mag_sq + noise_power))
            else:
                raise ValueError(f"Unknown estimation method: {method}")

        return H_est

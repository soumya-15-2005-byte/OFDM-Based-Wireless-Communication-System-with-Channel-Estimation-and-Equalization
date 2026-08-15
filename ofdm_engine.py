"""
OFDM Transceiver DSP Engine - Sig2Sig Implementation
Provides complete end-to-end signal processing for OFDM communications:
- Bit Generation & Modulation (BPSK, QPSK with Gray coding)
- Comb-type Pilot Insertion & Subcarrier Allocation
- IFFT & Cyclic Prefix (CP) addition/removal
- Multipath Rayleigh Fading Channel & AWGN Noise Injection
- Least-Squares (LS) & MMSE Channel Estimation
- Zero-Forcing (ZF) & MMSE Equalization
- Hard Decision Demodulation & BER Computation
"""

import numpy as np
from scipy import interpolate

class OFDMTransceiver:
    def __init__(self, n_fft=64, cp_length=16, modulation='QPSK', pilot_spacing=4, channel_taps=None):
        """
        Parameters:
        -----------
        n_fft : int
            Number of FFT subcarriers (default 64)
        cp_length : int
            Cyclic prefix length in samples (default 16)
        modulation : str
            'BPSK' or 'QPSK'
        pilot_spacing : int
            Spacing between pilot subcarriers (comb-type pilots)
        channel_taps : array-like or None
            Channel impulse response taps. If None, default 4-tap Rayleigh profile used.
        """
        self.n_fft = n_fft
        self.cp_length = cp_length
        self.modulation = modulation.upper()
        self.pilot_spacing = pilot_spacing
        
        # Subcarrier allocation (DC at 0, Guard bands at edges, Pilots every pilot_spacing)
        # We reserve subcarrier 0 as DC null. Edge subcarriers can be active.
        all_indices = np.arange(self.n_fft)
        
        # Pilot indices: 0 is DC, so pilots at 1, 1+spacing, ...
        self.pilot_indices = all_indices[1::self.pilot_spacing]
        self.data_indices = np.array([i for i in all_indices if i not in self.pilot_indices and i != 0])
        
        self.n_pilots = len(self.pilot_indices)
        self.n_data = len(self.data_indices)
        
        # Fixed pilot values (known to Tx and Rx)
        self.pilot_value = 1.0 + 0.0j if self.modulation == 'BPSK' else (1.0 + 1.0j) / np.sqrt(2)
        
        # Bits per symbol
        self.bits_per_symbol = 1 if self.modulation == 'BPSK' else 2
        
        # Default channel impulse response if not provided
        if channel_taps is None:
            # 4-tap Rayleigh fading channel with exponential power delay profile
            pdp = np.exp(-np.arange(4) / 1.5)
            pdp /= np.sum(pdp)  # Normalize total channel power to 1
            self.pdp = pdp
        else:
            self.pdp = np.array(channel_taps) / np.sqrt(np.sum(np.abs(channel_taps)**2))
            
    def generate_bits(self, n_ofdm_symbols):
        """Generate random binary sequence for given number of OFDM symbols."""
        total_bits = n_ofdm_symbols * self.n_data * self.bits_per_symbol
        return np.random.randint(0, 2, total_bits)
        
    def map_bits_to_symbols(self, bits):
        """Map bits to BPSK or QPSK complex constellation symbols with Gray coding."""
        if self.modulation == 'BPSK':
            # 0 -> +1, 1 -> -1
            symbols = 1.0 - 2.0 * bits.astype(float)
            return symbols.astype(complex)
        elif self.modulation == 'QPSK':
            # Group into 2-bit pairs (Gray mapping)
            bits_reshaped = bits.reshape(-1, 2)
            # 00 -> (1+j)/sqrt(2), 01 -> (-1+j)/sqrt(2), 11 -> (-1-j)/sqrt(2), 10 -> (1-j)/sqrt(2)
            i_val = 1.0 - 2.0 * bits_reshaped[:, 0]
            q_val = 1.0 - 2.0 * bits_reshaped[:, 1]
            symbols = (i_val + 1j * q_val) / np.sqrt(2.0)
            return symbols
        else:
            raise ValueError(f"Unsupported modulation: {self.modulation}")
            
    def demodulate_symbols_to_bits(self, symbols):
        """Hard decision demodulation from complex symbols back to bits."""
        if self.modulation == 'BPSK':
            # Re(symbol) > 0 -> bit 0, else bit 1
            return (np.real(symbols) < 0.0).astype(int)
        elif self.modulation == 'QPSK':
            bits_i = (np.real(symbols) < 0.0).astype(int)
            bits_q = (np.imag(symbols) < 0.0).astype(int)
            bits = np.column_stack((bits_i, bits_q)).ravel()
            return bits

    def transmit(self, data_symbols):
        """
        Build OFDM frame in frequency domain, apply IFFT, add CP.
        data_symbols shape: (n_ofdm_symbols, n_data)
        Returns:
        --------
        tx_signal : 1D complex array of time domain OFDM signal
        tx_freq_frames : 2D complex array of frequency domain subcarriers (n_symbols, n_fft)
        """
        n_symbols = len(data_symbols)
        tx_freq_frames = np.zeros((n_symbols, self.n_fft), dtype=complex)
        
        # Populate subcarriers
        tx_freq_frames[:, self.data_indices] = data_symbols
        tx_freq_frames[:, self.pilot_indices] = self.pilot_value
        tx_freq_frames[:, 0] = 0.0  # DC null
        
        # IFFT per symbol (scaling by sqrt(n_fft) for energy preservation)
        tx_time_frames = np.fft.ifft(tx_freq_frames, axis=1) * np.sqrt(self.n_fft)
        
        # Cyclic Prefix addition
        cp = tx_time_frames[:, -self.cp_length:]
        tx_time_with_cp = np.hstack((cp, tx_time_frames))
        
        # Serialize for transmission
        tx_signal = tx_time_with_cp.ravel()
        return tx_signal, tx_freq_frames

    def generate_rayleigh_channel(self, num_taps=None):
        """Generate random Rayleigh fading taps sampled from complex Gaussian distribution."""
        if num_taps is None:
            num_taps = len(self.pdp)
        sigma = np.sqrt(self.pdp / 2.0)
        taps = (np.random.normal(0, sigma) + 1j * np.random.normal(0, sigma))
        return taps

    def pass_channel(self, tx_signal, channel_taps, snr_db, channel_type='RAYLEIGH'):
        """
        Pass signal through wireless channel (Multipath Rayleigh Fading or AWGN) and add noise.
        """
        if channel_type.upper() == 'AWGN':
            rx_signal = tx_signal.copy()
            actual_channel = np.array([1.0])
        else:  # RAYLEIGH
            rx_signal = np.convolve(tx_signal, channel_taps, mode='full')[:len(tx_signal)]
            actual_channel = channel_taps

        # Calculate noise variance for given SNR dB
        # SNR_linear = P_signal / P_noise
        signal_power = np.mean(np.abs(rx_signal)**2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = signal_power / snr_linear
        
        # Complex AWGN noise generator
        noise = np.sqrt(noise_power / 2.0) * (np.random.randn(*rx_signal.shape) + 1j * np.random.randn(*rx_signal.shape))
        rx_signal_noisy = rx_signal + noise
        
        return rx_signal_noisy, actual_channel, noise_power

    def receive(self, rx_signal, n_symbols):
        """
        De-serialize, remove CP, and perform FFT.
        Returns:
        --------
        rx_freq_frames : 2D complex array (n_symbols, n_fft)
        """
        frame_len = self.n_fft + self.cp_length
        rx_time_with_cp = rx_signal[:n_symbols * frame_len].reshape(n_symbols, frame_len)
        
        # CP removal
        rx_time_frames = rx_time_with_cp[:, self.cp_length:]
        
        # FFT per symbol
        rx_freq_frames = np.fft.fft(rx_time_frames, axis=1) / np.sqrt(self.n_fft)
        return rx_freq_frames

    def estimate_channel(self, rx_freq_frames, method='LS', noise_power=1e-3):
        """
        Perform Channel Estimation using pilots.
        Methods:
        --------
        - 'LS': Least Squares estimation at pilot locations + Linear interpolation.
        - 'MMSE': Minimum Mean Square Error estimation.
        """
        n_symbols = len(rx_freq_frames)
        H_est = np.zeros((n_symbols, self.n_fft), dtype=complex)
        
        for i in range(n_symbols):
            rx_pilots = rx_freq_frames[i, self.pilot_indices]
            
            # LS estimation at pilot subcarriers
            H_pilots_ls = rx_pilots / self.pilot_value
            
            if method.upper() == 'LS':
                # Interpolate over all subcarriers
                # Handle boundaries by duplicating edge pilots
                pilots_idx = self.pilot_indices
                H_p_real = np.real(H_pilots_ls)
                H_p_imag = np.imag(H_pilots_ls)
                
                f_real = interpolate.interp1d(pilots_idx, H_p_real, kind='linear', fill_value='extrapolate')
                f_imag = interpolate.interp1d(pilots_idx, H_p_imag, kind='linear', fill_value='extrapolate')
                
                all_idx = np.arange(self.n_fft)
                H_est[i, :] = f_real(all_idx) + 1j * f_imag(all_idx)

            elif method.upper() == 'MMSE':
                # MMSE estimate derived from LS pilots with noise variance smoothing
                pilots_idx = self.pilot_indices
                H_p_real = np.real(H_pilots_ls)
                H_p_imag = np.imag(H_pilots_ls)
                
                f_real = interpolate.interp1d(pilots_idx, H_p_real, kind='linear', fill_value='extrapolate')
                f_imag = interpolate.interp1d(pilots_idx, H_p_imag, kind='linear', fill_value='extrapolate')
                
                all_idx = np.arange(self.n_fft)
                H_ls_interp = f_real(all_idx) + 1j * f_imag(all_idx)
                
                # Frequency-domain MMSE smoothing filter
                snr_inv = noise_power
                H_est[i, :] = H_ls_interp * (np.abs(H_ls_interp)**2 / (np.abs(H_ls_interp)**2 + snr_inv))
            else:
                raise ValueError(f"Unknown estimation method: {method}")
                
        return H_est

    def equalize(self, rx_freq_frames, H_est, method='ZF', snr_db=20):
        """
        Channel Equalization.
        Methods:
        --------
        - 'ZF': Zero Forcing (X_hat = Y / H_est)
        - 'MMSE': Minimum Mean Square Error (X_hat = Y * H_est* / (|H_est|^2 + 1/SNR))
        """
        snr_linear = 10.0 ** (snr_db / 10.0)
        
        if method.upper() == 'ZF':
            # Avoid division by zero with small epsilon
            H_safe = np.where(np.abs(H_est) < 1e-12, 1e-12, H_est)
            equalized_freq = rx_freq_frames / H_safe
        elif method.upper() == 'MMSE':
            # MMSE equalizer weight: W = H* / (|H|^2 + 1/SNR)
            denom = np.abs(H_est)**2 + (1.0 / snr_linear)
            equalized_freq = rx_freq_frames * np.conj(H_est) / denom
        else:
            raise ValueError(f"Unknown equalization method: {method}")
            
        # Extract data subcarriers
        equalized_data_symbols = equalized_freq[:, self.data_indices]
        return equalized_data_symbols, equalized_freq

    def compute_ber(self, tx_bits, rx_bits):
        """Compute Bit Error Rate (BER)."""
        bit_errors = np.sum(tx_bits != rx_bits)
        ber = bit_errors / len(tx_bits)
        return ber, bit_errors

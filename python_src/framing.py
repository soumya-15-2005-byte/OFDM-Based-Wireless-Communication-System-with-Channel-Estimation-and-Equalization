"""
OFDM Framing Module - IFFT/FFT Processing & Cyclic Prefix Guard
"""

import numpy as np

class OFDMFramer:
    def __init__(self, n_fft=64, cp_length=16, pilot_spacing=4, modulation_scheme='QPSK'):
        self.n_fft = n_fft
        self.cp_length = cp_length
        self.pilot_spacing = pilot_spacing
        
        all_indices = np.arange(self.n_fft)
        self.pilot_indices = all_indices[1::self.pilot_spacing]
        self.data_indices = np.array([i for i in all_indices if i not in self.pilot_indices and i != 0])
        
        self.n_pilots = len(self.pilot_indices)
        self.n_data = len(self.data_indices)
        
        # Pilot symbol value
        self.pilot_value = 1.0 + 0.0j if modulation_scheme.upper() == 'BPSK' else (1.0 + 1.0j) / np.sqrt(2)

    def assemble_and_ifft(self, data_symbols):
        """
        Map data and pilot symbols to subcarriers, execute IFFT, add CP.
        Returns:
        --------
        tx_signal : 1D time-domain array
        tx_freq_frames : 2D frequency-domain array
        """
        n_symbols = len(data_symbols) // self.n_data
        tx_freq_frames = np.zeros((n_symbols, self.n_fft), dtype=complex)

        for s in range(n_symbols):
            sym_slice = data_symbols[s * self.n_data : (s + 1) * self.n_data]
            tx_freq_frames[s, self.data_indices] = sym_slice
            tx_freq_frames[s, self.pilot_indices] = self.pilot_value
            tx_freq_frames[s, 0] = 0.0 # DC null

        # IFFT per symbol (energy-preserving scaling)
        tx_time_frames = np.fft.ifft(tx_freq_frames, axis=1) * np.sqrt(self.n_fft)

        # Add Cyclic Prefix (CP)
        cp = tx_time_frames[:, -self.cp_length:]
        tx_time_with_cp = np.hstack((cp, tx_time_frames))

        tx_signal = tx_time_with_cp.ravel()
        return tx_signal, tx_freq_frames

    def remove_cp_and_fft(self, rx_signal, n_symbols):
        """
        Remove CP and perform FFT to recover frequency domain subcarrier frames.
        """
        frame_len = self.n_fft + self.cp_length
        rx_time_with_cp = rx_signal[: n_symbols * frame_len].reshape(n_symbols, frame_len)

        # CP removal
        rx_time_frames = rx_time_with_cp[:, self.cp_length :]

        # FFT per symbol
        rx_freq_frames = np.fft.fft(rx_time_frames, axis=1) / np.sqrt(self.n_fft)
        return rx_freq_frames

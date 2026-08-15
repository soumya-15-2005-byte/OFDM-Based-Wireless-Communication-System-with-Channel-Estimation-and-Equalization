"""
Wireless Channel Module - AWGN Noise & Multipath Rayleigh Fading
"""

import numpy as np

class WirelessChannel:
    def __init__(self, channel_type='RAYLEIGH', pdp_taps=4):
        self.channel_type = channel_type.upper()
        # Default 4-tap Rayleigh fading exponential Power Delay Profile (PDP)
        raw_pdp = np.exp(-np.arange(pdp_taps) / 1.5)
        self.pdp = raw_pdp / np.sum(raw_pdp)

    def generate_rayleigh_taps(self):
        """Generate Rayleigh fading complex impulse response taps."""
        sigma = np.sqrt(self.pdp / 2.0)
        return np.random.normal(0, sigma) + 1j * np.random.normal(0, sigma)

    def pass_channel(self, tx_signal, channel_taps, snr_db):
        """
        Convolve signal with channel impulse response and add complex AWGN noise.
        """
        if self.channel_type == 'AWGN':
            rx_conv = tx_signal.copy()
            actual_taps = np.array([1.0])
        else: # RAYLEIGH
            rx_conv = np.convolve(tx_signal, channel_taps, mode='full')[: len(tx_signal)]
            actual_taps = channel_taps

        # Noise scaling based on Eb/N0 or SNR dB
        sig_power = np.mean(np.abs(rx_conv) ** 2)
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = sig_power / max(1e-12, snr_linear)

        sigma_n = np.sqrt(noise_power / 2.0)
        noise = sigma_n * (np.random.randn(*rx_conv.shape) + 1j * np.random.randn(*rx_conv.shape))
        rx_noisy = rx_conv + noise

        return rx_noisy, actual_taps, noise_power

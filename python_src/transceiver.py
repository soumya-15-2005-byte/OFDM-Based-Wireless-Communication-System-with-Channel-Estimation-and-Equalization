"""
OFDM Transceiver Orchestrator
Project Title: "OFDM-Based Wireless Communication System with Channel Estimation and Equalization"
"""

from .modulation import Modulator
from .framing import OFDMFramer
from .channel import WirelessChannel
from .estimation import ChannelEstimator
from .equalization import ChannelEqualizer

class OFDMTransceiver:
    def __init__(self, n_fft=64, cp_length=16, modulation='QPSK', pilot_spacing=4, channel_type='RAYLEIGH'):
        self.n_fft = n_fft
        self.cp_length = cp_length
        self.channel_type = channel_type

        self.modulator = Modulator(scheme=modulation)
        self.framer = OFDMFramer(n_fft=n_fft, cp_length=cp_length, pilot_spacing=pilot_spacing, modulation_scheme=modulation)
        self.channel = WirelessChannel(channel_type=channel_type)

        self.n_data = self.framer.n_data
        self.n_pilots = self.framer.n_pilots

    def run_pipeline(self, n_ofdm_symbols, snr_db, est_method='LS', eq_method='ZF'):
        """
        Run end-to-end Sig2Sig simulation pipeline.
        Returns dictionary of all intermediate signals and metrics.
        """
        # 1. Generate bits
        tx_bits = self.modulator.generate_bits(n_ofdm_symbols * self.n_data * self.modulator.bits_per_symbol)

        # 2. Map to symbols
        tx_data_symbols = self.modulator.map_bits_to_symbols(tx_bits)

        # 3. Framing & IFFT
        tx_signal, tx_freq_frames = self.framer.assemble_and_ifft(tx_data_symbols)

        # 4. Wireless Channel
        taps = self.channel.generate_rayleigh_taps()
        rx_signal, actual_taps, noise_power = self.channel.pass_channel(tx_signal, taps, snr_db)

        # 5. Receive FFT
        rx_freq_frames = self.framer.remove_cp_and_fft(rx_signal, n_ofdm_symbols)

        # 6. Channel Estimation
        H_est = ChannelEstimator.estimate(
            rx_freq_frames=rx_freq_frames,
            pilot_indices=self.framer.pilot_indices,
            pilot_value=self.framer.pilot_value,
            n_fft=self.n_fft,
            method=est_method,
            noise_power=noise_power
        )

        # 7. Equalization
        eq_data_symbols, eq_freq_frames = ChannelEqualizer.equalize(
            rx_freq_frames=rx_freq_frames,
            H_est=H_est,
            data_indices=self.framer.data_indices,
            n_fft=self.n_fft,
            method=eq_method,
            snr_db=snr_db
        )

        # 8. Demodulate & BER
        rx_bits = self.modulator.demodulate_symbols_to_bits(eq_data_symbols.ravel())
        ber, bit_errors = Modulator.compute_ber(tx_bits, rx_bits)

        return {
            'tx_bits': tx_bits,
            'tx_data_symbols': tx_data_symbols,
            'tx_signal': tx_signal,
            'tx_freq_frames': tx_freq_frames,
            'actual_taps': actual_taps,
            'rx_signal': rx_signal,
            'rx_freq_frames': rx_freq_frames,
            'H_est': H_est,
            'eq_data_symbols': eq_data_symbols,
            'rx_bits': rx_bits,
            'ber': ber,
            'bit_errors': bit_errors,
            'total_bits': len(tx_bits)
        }

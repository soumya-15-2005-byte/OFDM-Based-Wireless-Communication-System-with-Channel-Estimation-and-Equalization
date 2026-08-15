"""
Publication Figure Generator Script
Generates high-resolution PNG plots for constellations, channel response, and BER curves.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python_src.transceiver import OFDMTransceiver

def generate_all_figures():
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'docs_images'))
    os.makedirs(output_dir, exist_ok=True)

    # 1. Constellations Figure
    tx = OFDMTransceiver(n_fft=64, cp_length=16, modulation='QPSK', channel_type='RAYLEIGH')
    res_zf = tx.run_pipeline(n_ofdm_symbols=40, snr_db=16.0, est_method='LS', eq_method='ZF')
    res_mmse = tx.run_pipeline(n_ofdm_symbols=40, snr_db=16.0, est_method='LS', eq_method='MMSE')

    rx_raw = res_zf['rx_freq_frames'][:, tx.framer.data_indices].ravel()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)
    fig.suptitle('Sig2Sig Constellation Evolution in Rayleigh Fading (SNR = 16 dB)', fontsize=13, fontweight='bold')

    axes[0].scatter(np.real(res_zf['tx_data_symbols']), np.imag(res_zf['tx_data_symbols']), color='#3b82f6', s=35)
    axes[0].set_title('1. Transmitted QPSK')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].scatter(np.real(rx_raw), np.imag(rx_raw), color='#ef4444', alpha=0.5, s=25)
    axes[1].set_title('2. Received Faded + Noisy')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    axes[2].scatter(np.real(res_zf['eq_data_symbols']), np.imag(res_zf['eq_data_symbols']), color='#eab308', s=30)
    axes[2].set_title('3. Equalized (Zero-Forcing)')
    axes[2].grid(True, linestyle='--', alpha=0.5)

    axes[3].scatter(np.real(res_mmse['eq_data_symbols']), np.imag(res_mmse['eq_data_symbols']), color='#22c55e', s=30)
    axes[3].set_title('4. Equalized (MMSE)')
    axes[3].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'constellations.png'), dpi=300)
    plt.close()

    print(f"Generated figures in '{output_dir}'.")

if __name__ == '__main__':
    generate_all_figures()

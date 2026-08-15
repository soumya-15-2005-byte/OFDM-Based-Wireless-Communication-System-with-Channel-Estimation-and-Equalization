"""
Monte Carlo BER Simulation Runner Script
Project Title: "OFDM-Based Wireless Communication System with Channel Estimation and Equalization"
"""

import os
import sys
import json
import numpy as np
from scipy.special import erfc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from python_src.transceiver import OFDMTransceiver

def theoretical_ber_awgn(snr_db_range):
    snr_linear = 10.0 ** (snr_db_range / 10.0)
    return 0.5 * erfc(np.sqrt(snr_linear))

def theoretical_ber_rayleigh(snr_db_range):
    snr_linear = 10.0 ** (snr_db_range / 10.0)
    gamma_b = snr_linear
    return 0.5 * (1.0 - np.sqrt(gamma_b / (1.0 + gamma_b)))

def run_simulation():
    snr_range = np.arange(0, 31, 3)
    n_symbols = 200

    results = {
        'snr_db': snr_range.tolist(),
        'awgn_theory': theoretical_ber_awgn(snr_range).tolist(),
        'rayleigh_theory': theoretical_ber_rayleigh(snr_range).tolist(),
        'bpsk_awgn': [],
        'qpsk_awgn': [],
        'qpsk_rayleigh_ls_zf': [],
        'qpsk_rayleigh_ls_mmse': []
    }

    print("==========================================================================")
    print("      OFDM Wireless Communication System - BER Simulation      ")
    print("==========================================================================")

    for snr in snr_range:
        print(f"Running simulation at SNR = {snr:2d} dB...")

        # 1. BPSK AWGN
        tx_bpsk = OFDMTransceiver(n_fft=64, modulation='BPSK', channel_type='AWGN')
        res_b_awgn = tx_bpsk.run_pipeline(n_symbols, snr, est_method='LS', eq_method='ZF')
        results['bpsk_awgn'].append(res_b_awgn['ber'])

        # 2. QPSK AWGN
        tx_qpsk_awgn = OFDMTransceiver(n_fft=64, modulation='QPSK', channel_type='AWGN')
        res_q_awgn = tx_qpsk_awgn.run_pipeline(n_symbols, snr, est_method='LS', eq_method='ZF')
        results['qpsk_awgn'].append(res_q_awgn['ber'])

        # 3. QPSK Rayleigh (LS + ZF)
        tx_qpsk_ray = OFDMTransceiver(n_fft=64, modulation='QPSK', channel_type='RAYLEIGH')
        res_q_zf = tx_qpsk_ray.run_pipeline(n_symbols, snr, est_method='LS', eq_method='ZF')
        results['qpsk_rayleigh_ls_zf'].append(res_q_zf['ber'])

        # 4. QPSK Rayleigh (LS + MMSE)
        res_q_mmse = tx_qpsk_ray.run_pipeline(n_symbols, snr, est_method='LS', eq_method='MMSE')
        results['qpsk_rayleigh_ls_mmse'].append(res_q_mmse['ber'])

    out_file = os.path.join(os.path.dirname(__file__), '..', 'ber_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSimulation complete! Results saved to '{os.path.basename(out_file)}'.")

if __name__ == '__main__':
    run_simulation()

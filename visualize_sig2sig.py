"""
Signal & System Visualizer for OFDM Project
Project Title: "OFDM-Based Wireless Communication System with Channel Estimation and Equalization"

Generates static publication-ready matplotlib figures:
1. Constellation Diagrams (Tx, Noisy/Faded Rx, Equalized Rx)
2. OFDM Time Waveform & Cyclic Prefix (CP) demarcation
3. Channel Frequency Response |H(f)| & LS Channel Estimate |H_est(f)|
4. BER vs SNR Performance Curves (AWGN & Rayleigh Fading)
"""

import numpy as np
import matplotlib.pyplot as plt
from ofdm_engine import OFDMTransceiver
from ber_simulation import run_ber_simulation

def plot_constellations():
    """Plot constellation evolution across Sig2Sig stages."""
    transceiver = OFDMTransceiver(n_fft=64, cp_length=16, modulation='QPSK', pilot_spacing=4)
    n_symbols = 30
    
    bits = transceiver.generate_bits(n_symbols)
    tx_symbols = transceiver.map_bits_to_symbols(bits).reshape(n_symbols, transceiver.n_data)
    tx_signal, _ = transceiver.transmit(tx_symbols)
    
    # Rayleigh fading at 15 dB
    taps = transceiver.generate_rayleigh_channel()
    rx_signal, actual_taps, noise_pwr = transceiver.pass_channel(tx_signal, taps, snr_db=15.0, channel_type='RAYLEIGH')
    rx_freq = transceiver.receive(rx_signal, n_symbols)
    
    rx_data_raw = rx_freq[:, transceiver.data_indices]
    
    # Estimation & Equalization
    h_est = transceiver.estimate_channel(rx_freq, method='LS', noise_power=noise_pwr)
    eq_data_zf, _ = transceiver.equalize(rx_freq, h_est, method='ZF', snr_db=15.0)
    eq_data_mmse, _ = transceiver.equalize(rx_freq, h_est, method='MMSE', snr_db=15.0)
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), dpi=150)
    fig.suptitle('Sig2Sig Constellation Evolution in Rayleigh Fading Channel (SNR = 15 dB)', fontsize=14, fontweight='bold')
    
    # 1. Transmitted Constellation
    axes[0].scatter(np.real(tx_symbols), np.imag(tx_symbols), color='#3b82f6', alpha=0.7, edgecolors='k', s=40)
    axes[0].set_title('1. Transmitted QPSK')
    axes[0].set_xlabel('In-Phase (I)')
    axes[0].set_ylabel('Quadrature (Q)')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_xlim([-1.5, 1.5])
    axes[0].set_ylim([-1.5, 1.5])
    
    # 2. Faded + Noisy Received
    axes[1].scatter(np.real(rx_data_raw), np.imag(rx_data_raw), color='#ef4444', alpha=0.6, s=30)
    axes[1].set_title('2. Received (Faded + Noise)')
    axes[1].set_xlabel('In-Phase (I)')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    # 3. Equalized (Zero-Forcing)
    axes[2].scatter(np.real(eq_data_zf), np.imag(eq_data_zf), color='#eab308', alpha=0.7, edgecolors='k', s=35)
    axes[2].set_title('3. Equalized (Zero-Forcing)')
    axes[2].set_xlabel('In-Phase (I)')
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].set_xlim([-2.5, 2.5])
    axes[2].set_ylim([-2.5, 2.5])
    
    # 4. Equalized (MMSE)
    axes[3].scatter(np.real(eq_data_mmse), np.imag(eq_data_mmse), color='#22c55e', alpha=0.7, edgecolors='k', s=35)
    axes[3].set_title('4. Equalized (MMSE)')
    axes[3].set_xlabel('In-Phase (I)')
    axes[3].grid(True, linestyle='--', alpha=0.5)
    axes[3].set_xlim([-1.8, 1.8])
    axes[3].set_ylim([-1.8, 1.8])
    
    plt.tight_layout()
    plt.savefig('ofdm_constellations.png', dpi=300)
    print("Saved 'ofdm_constellations.png'.")
    plt.close()

def plot_channel_frequency_response():
    """Plot Actual Channel Magnitude Response vs LS Estimated Channel Response."""
    transceiver = OFDMTransceiver(n_fft=64, cp_length=16, modulation='QPSK', pilot_spacing=4)
    n_symbols = 5
    
    bits = transceiver.generate_bits(n_symbols)
    tx_symbols = transceiver.map_bits_to_symbols(bits).reshape(n_symbols, transceiver.n_data)
    tx_signal, _ = transceiver.transmit(tx_symbols)
    
    taps = transceiver.generate_rayleigh_channel()
    rx_signal, actual_taps, noise_pwr = transceiver.pass_channel(tx_signal, taps, snr_db=18.0, channel_type='RAYLEIGH')
    rx_freq = transceiver.receive(rx_signal, n_symbols)
    
    H_true = np.abs(np.fft.fft(actual_taps, transceiver.n_fft))
    H_est = np.abs(transceiver.estimate_channel(rx_freq, method='LS', noise_power=noise_pwr)[0])
    
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(np.arange(transceiver.n_fft), 20*np.log10(H_true + 1e-12), 'b-', linewidth=2.5, label='Actual Channel |H(f)|')
    plt.plot(np.arange(transceiver.n_fft), 20*np.log10(H_est + 1e-12), 'r--', linewidth=2, label='LS Estimated |H_est(f)|')
    plt.scatter(transceiver.pilot_indices, 20*np.log10(H_est[transceiver.pilot_indices] + 1e-12), color='darkred', s=60, zorder=5, label='Pilot Locations')
    
    plt.title('Channel Frequency Response & LS Channel Estimation (Comb Pilots)', fontsize=12, fontweight='bold')
    plt.xlabel('Subcarrier Index (k)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('ofdm_channel_estimation.png', dpi=300)
    print("Saved 'ofdm_channel_estimation.png'.")
    plt.close()

def plot_ber_curves():
    """Run BER simulation and plot BER vs SNR comparison curves."""
    results = run_ber_simulation(snr_db_range=np.arange(0, 28, 2), n_symbols_per_snr=150)
    
    snr = results['snr_db']
    
    plt.figure(figsize=(10, 6.5), dpi=150)
    
    # AWGN Theoretical & Simulated
    plt.semilogy(snr, results['awgn_bpsk_theory'], 'k--', linewidth=2, label='AWGN Theoretical (BPSK/QPSK)')
    plt.semilogy(snr, results['bpsk_awgn_sim'], 'bo-', linewidth=1.8, markersize=6, label='BPSK AWGN (Simulated)')
    plt.semilogy(snr, results['qpsk_awgn_sim'], 'gs-', linewidth=1.8, markersize=6, label='QPSK AWGN (Simulated)')
    
    # Rayleigh Theoretical & Simulated
    plt.semilogy(snr, results['awgn_rayleigh_theory'], 'm--', linewidth=2, label='Rayleigh Fading Theoretical (Ideal CSI)')
    plt.semilogy(snr, results['qpsk_rayleigh_ideal_zf'], 'c^-', linewidth=1.8, markersize=6, label='QPSK Rayleigh Ideal CSI + ZF')
    plt.semilogy(snr, results['qpsk_rayleigh_ls_zf'], 'rx-', linewidth=1.8, markersize=7, label='QPSK Rayleigh LS Est + ZF')
    plt.semilogy(snr, results['qpsk_rayleigh_ls_mmse'], 'd-', color='#16a34a', linewidth=1.8, markersize=6, label='QPSK Rayleigh LS Est + MMSE')
    
    plt.title('BER vs SNR Performance Analysis - OFDM Wireless Communication System', fontsize=12, fontweight='bold')
    plt.xlabel('Average $E_b/N_0$ or SNR (dB)', fontsize=11)
    plt.ylabel('Bit Error Rate (BER)', fontsize=11)
    plt.ylim([1e-4, 1.0])
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=9, loc='lower left')
    plt.tight_layout()
    plt.savefig('ber_vs_snr_performance.png', dpi=300)
    print("Saved 'ber_vs_snr_performance.png'.")
    plt.close()

if __name__ == '__main__':
    plot_constellations()
    plot_channel_frequency_response()
    plot_ber_curves()

"""
BER vs SNR Simulation Suite for OFDM System
Project Title: "OFDM-Based Wireless Communication System with Channel Estimation and Equalization"

Performs Monte Carlo BER simulations across SNR range (0 to 30 dB) for:
1. BPSK AWGN (Simulated vs Theoretical)
2. QPSK AWGN (Simulated vs Theoretical)
3. BPSK Rayleigh Fading (Ideal CSI vs LS Channel Estimation + ZF/MMSE)
4. QPSK Rayleigh Fading (Ideal CSI vs LS Channel Estimation + ZF/MMSE)
"""

import numpy as np
from scipy.special import erfc
import json
from ofdm_engine import OFDMTransceiver

def theoretical_ber_awgn(snr_db_range):
    """Theoretical BER for BPSK/QPSK over AWGN channel."""
    snr_linear = 10.0 ** (snr_db_range / 10.0)
    # Eb/N0 = SNR_linear for BPSK/QPSK
    return 0.5 * erfc(np.sqrt(snr_linear))

def theoretical_ber_rayleigh(snr_db_range):
    """Theoretical BER for BPSK/QPSK over flat/uncorrelated Rayleigh fading channel."""
    snr_linear = 10.0 ** (snr_db_range / 10.0)
    gamma_b = snr_linear
    return 0.5 * (1.0 - np.sqrt(gamma_b / (1.0 + gamma_b)))

def run_ber_simulation(snr_db_range=np.arange(0, 31, 3), n_symbols_per_snr=200):
    """
    Run Monte Carlo BER simulation across SNR range for multiple scenarios.
    """
    results = {
        'snr_db': snr_db_range.tolist(),
        'awgn_bpsk_theory': theoretical_ber_awgn(snr_db_range).tolist(),
        'awgn_rayleigh_theory': theoretical_ber_rayleigh(snr_db_range).tolist(),
        'bpsk_awgn_sim': [],
        'qpsk_awgn_sim': [],
        'qpsk_rayleigh_ideal_zf': [],
        'qpsk_rayleigh_ls_zf': [],
        'qpsk_rayleigh_ls_mmse': []
    }

    print("==========================================================================")
    print("      OFDM-Based Wireless Communication System Simulation      ")
    print("==========================================================================")

    for snr_db in snr_db_range:
        print(f"Simulating SNR = {snr_db:2d} dB...")

        # ---------------------------------------------------------
        # Scenario 1: BPSK over AWGN
        # ---------------------------------------------------------
        tx_bpsk = OFDMTransceiver(n_fft=64, cp_length=16, modulation='BPSK', pilot_spacing=4)
        bits = tx_bpsk.generate_bits(n_symbols_per_snr)
        syms = tx_bpsk.map_bits_to_symbols(bits).reshape(n_symbols_per_snr, tx_bpsk.n_data)
        sig, _ = tx_bpsk.transmit(syms)
        rx_sig, _, _ = tx_bpsk.pass_channel(sig, [1.0], snr_db, channel_type='AWGN')
        rx_freq = tx_bpsk.receive(rx_sig, n_symbols_per_snr)
        h_est = tx_bpsk.estimate_channel(rx_freq, method='LS')
        eq_syms, _ = tx_bpsk.equalize(rx_freq, h_est, method='ZF', snr_db=snr_db)
        rx_bits = tx_bpsk.demodulate_symbols_to_bits(eq_syms.ravel())
        ber_bpsk_awgn, _ = tx_bpsk.compute_ber(bits, rx_bits)
        results['bpsk_awgn_sim'].append(ber_bpsk_awgn)

        # ---------------------------------------------------------
        # Scenario 2: QPSK over AWGN
        # ---------------------------------------------------------
        tx_qpsk = OFDMTransceiver(n_fft=64, cp_length=16, modulation='QPSK', pilot_spacing=4)
        bits = tx_qpsk.generate_bits(n_symbols_per_snr)
        syms = tx_qpsk.map_bits_to_symbols(bits).reshape(n_symbols_per_snr, tx_qpsk.n_data)
        sig, _ = tx_qpsk.transmit(syms)
        rx_sig, _, _ = tx_qpsk.pass_channel(sig, [1.0], snr_db, channel_type='AWGN')
        rx_freq = tx_qpsk.receive(rx_sig, n_symbols_per_snr)
        h_est = tx_qpsk.estimate_channel(rx_freq, method='LS')
        eq_syms, _ = tx_qpsk.equalize(rx_freq, h_est, method='ZF', snr_db=snr_db)
        rx_bits = tx_qpsk.demodulate_symbols_to_bits(eq_syms.ravel())
        ber_qpsk_awgn, _ = tx_qpsk.compute_ber(bits, rx_bits)
        results['qpsk_awgn_sim'].append(ber_qpsk_awgn)

        # ---------------------------------------------------------
        # Scenario 3: QPSK over Rayleigh (Ideal CSI + ZF)
        # ---------------------------------------------------------
        bits = tx_qpsk.generate_bits(n_symbols_per_snr)
        syms = tx_qpsk.map_bits_to_symbols(bits).reshape(n_symbols_per_snr, tx_qpsk.n_data)
        sig, _ = tx_qpsk.transmit(syms)
        taps = tx_qpsk.generate_rayleigh_channel()
        rx_sig, actual_taps, noise_pwr = tx_qpsk.pass_channel(sig, taps, snr_db, channel_type='RAYLEIGH')
        rx_freq = tx_qpsk.receive(rx_sig, n_symbols_per_snr)
        
        # True channel frequency response H_ideal
        H_true = np.fft.fft(actual_taps, tx_qpsk.n_fft)
        H_ideal = np.tile(H_true, (n_symbols_per_snr, 1))
        
        # Ideal ZF
        eq_ideal, _ = tx_qpsk.equalize(rx_freq, H_ideal, method='ZF', snr_db=snr_db)
        rx_bits_ideal = tx_qpsk.demodulate_symbols_to_bits(eq_ideal.ravel())
        ber_ideal, _ = tx_qpsk.compute_ber(bits, rx_bits_ideal)
        results['qpsk_rayleigh_ideal_zf'].append(ber_ideal)

        # ---------------------------------------------------------
        # Scenario 4: QPSK over Rayleigh (LS Channel Estimation + ZF)
        # ---------------------------------------------------------
        H_ls = tx_qpsk.estimate_channel(rx_freq, method='LS', noise_power=noise_pwr)
        eq_ls_zf, _ = tx_qpsk.equalize(rx_freq, H_ls, method='ZF', snr_db=snr_db)
        rx_bits_ls_zf = tx_qpsk.demodulate_symbols_to_bits(eq_ls_zf.ravel())
        ber_ls_zf, _ = tx_qpsk.compute_ber(bits, rx_bits_ls_zf)
        results['qpsk_rayleigh_ls_zf'].append(ber_ls_zf)

        # ---------------------------------------------------------
        # Scenario 5: QPSK over Rayleigh (LS Channel Estimation + MMSE)
        # ---------------------------------------------------------
        eq_ls_mmse, _ = tx_qpsk.equalize(rx_freq, H_ls, method='MMSE', snr_db=snr_db)
        rx_bits_ls_mmse = tx_qpsk.demodulate_symbols_to_bits(eq_ls_mmse.ravel())
        ber_ls_mmse, _ = tx_qpsk.compute_ber(bits, rx_bits_ls_mmse)
        results['qpsk_rayleigh_ls_mmse'].append(ber_ls_mmse)

    # Save results to JSON file
    with open('ber_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nSimulation Complete! Saved results to 'ber_results.json'.")
    return results

if __name__ == '__main__':
    run_ber_simulation()

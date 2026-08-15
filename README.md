# OFDM-Based Wireless Communication System with Channel Estimation and Equalization

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![DSP: OFDM](https://img.shields.io/badge/DSP-OFDM%20%7C%20QPSK%20%7C%20Rayleigh-purple.svg)]()

> A complete end-to-end **Sig2Sig (Signal-to-Signal)** Orthogonal Frequency Division Multiplexing (OFDM) transceiver implementation featuring **BPSK & QPSK modulation**, **AWGN & Multipath Rayleigh Fading channels**, **Comb-type Pilot Insertion**, **LS & MMSE Channel Estimation**, **Zero-Forcing (ZF) & MMSE Equalization**, and **Bit Error Rate (BER) vs SNR Monte Carlo performance analysis**.

---

## 🌟 Key Features

- **Sig2Sig Pipeline Architecture**: End-to-end discrete digital signal processing from information bitstream generation to received signal reconstruction.
- **Flexible Modulation Schemes**: BPSK (1 bit/symbol) and QPSK (2 bits/symbol) with Gray coding and optimal decision slicing.
- **OFDM Framing & Guard Interval**: $N_{fft}$ subcarrier mapping, DC subcarrier nulling, Comb-type pilot insertion, IFFT/FFT transformations, and Cyclic Prefix (CP) addition/removal to prevent Inter-Symbol Interference (ISI).
- **Realistic Wireless Channels**:
  - **AWGN Channel**: Additive White Gaussian Noise with configurable $E_b/N_0$ scaling.
  - **Multipath Rayleigh Fading**: 4-tap frequency-selective fading model with exponential Power Delay Profile (PDP).
- **Advanced Channel Estimation**:
  - **Least Squares (LS)**: Fast pilot subcarrier estimation with linear/spline subcarrier interpolation.
  - **MMSE Estimator**: Minimum Mean Square Error estimation with frequency-domain noise variance smoothing.
- **Channel Equalization**:
  - **Zero-Forcing (ZF)**: Inverse channel transfer matrix equalizer ($\hat{X}_{ZF} = Y / \hat{H}$).
  - **MMSE Equalizer**: Noise-aware minimum mean square error equalizer ($\hat{X}_{MMSE} = Y \cdot \frac{\hat{H}^*}{|\hat{H}|^2 + 1/SNR}$).
- **Performance Benchmarking**: Automated Monte Carlo BER vs. SNR evaluation comparing empirical simulation curves against theoretical AWGN ($Q$-function) and Rayleigh fading error bounds.
- **Interactive Modern Web Simulator**: Built-in glassmorphic web interface featuring real-time constellation diagrams, Cyclic Prefix oscilloscope view, channel frequency response $|H(f)|$ plot, and an **Interactive Image Transmission Demo**.

---

## 📐 Mathematical Formulation

### 1. OFDM Modulation & Cyclic Prefix
Given input symbol vector $X[k]$ of length $N_{fft}$, the discrete time-domain OFDM symbol $x[n]$ is obtained via Inverse Fast Fourier Transform (IFFT):
$$x[n] = \frac{1}{\sqrt{N_{fft}}} \sum_{k=0}^{N_{fft}-1} X[k] e^{j \frac{2\pi}{N_{fft}} k n}, \quad 0 \le n < N_{fft}$$

A Cyclic Prefix (CP) of length $N_{cp}$ repeats the last $N_{cp}$ samples of $x[n]$ at the beginning of the frame to convert linear channel convolution into circular convolution:
$$x_{cp}[n] = [x[N_{fft}-N_{cp}], \dots, x[N_{fft}-1], x[0], \dots, x[N_{fft}-1]]$$

### 2. Wireless Channel Model
The signal passes through a time-domain multipath channel $h[l]$ corrupting the signal with complex AWGN noise $w[n] \sim \mathcal{CN}(0, \sigma_n^2)$:
$$y_{cp}[n] = \sum_{l=0}^{L-1} h[l] x_{cp}[n-l] + w[n]$$

After removing the cyclic prefix and executing Fast Fourier Transform (FFT), the frequency-domain received signal at subcarrier $k$ is given by:
$$Y[k] = H[k] X[k] + W[k]$$
where $H[k] = \text{FFT}\{h[l]\}(k)$ is the Channel Frequency Response.

### 3. Channel Estimation & Equalization
At pilot subcarrier indices $k_p$, the **Least Squares (LS)** channel estimate is:
$$\hat{H}_{LS}[k_p] = \frac{Y[k_p]}{X[k_p]}$$

Linear interpolation yields $\hat{H}[k]$ for all data subcarriers $k$.

- **Zero-Forcing (ZF) Equalizer**:
  $$\hat{X}_{ZF}[k] = \frac{Y[k]}{\hat{H}[k]}$$

- **MMSE Equalizer**:
  $$\hat{X}_{MMSE}[k] = Y[k] \cdot \frac{\hat{H}^*[k]}{|\hat{H}[k]|^2 + \frac{1}{\text{SNR}_{linear}}}$$

### 4. Theoretical BER Bounds
- **BPSK / QPSK over AWGN**:
  $$P_{e, \text{AWGN}} = Q\left(\sqrt{2 \frac{E_b}{N_0}}\right) = \frac{1}{2} \text{erfc}\left(\sqrt{\frac{E_b}{N_0}}\right)$$

- **BPSK / QPSK over Rayleigh Fading (Ideal CSI)**:
  $$P_{e, \text{Rayleigh}} = \frac{1}{2} \left(1 - \sqrt{\frac{\bar{\gamma}_b}{1 + \bar{\gamma}_b}}\right), \quad \text{where } \bar{\gamma}_b = \frac{E_b}{N_0}$$

---

## 📁 Repository Structure

```
OFDM-Based-Wireless-Communication-System-with-Channel-Estimation-and-Equalization/
│
├── README.md                           # Project Documentation
├── LICENSE                             # MIT License
├── requirements.txt                    # Python Dependencies
├── .gitignore                          # Git Ignore Patterns
│
├── python_src/                         # Modular Core Python Package
│   ├── __init__.py                     # Package Initialization
│   ├── modulation.py                   # BPSK & QPSK Constellation Mappers & Demodulator
│   ├── framing.py                      # Subcarrier Mapping, Comb Pilots, IFFT/FFT & CP
│   ├── channel.py                      # AWGN & Rayleigh Multipath Fading Models
│   ├── estimation.py                   # LS & MMSE Channel Estimators
│   ├── equalization.py                 # Zero-Forcing (ZF) & MMSE Equalizers
│   └── transceiver.py                  # End-to-End OFDM Transceiver Pipeline
│
├── tests/                              # Unit Testing Suite
│   └── test_ofdm.py                    # Unit Tests verifying noiseless & fading BER
│
├── scripts/                            # Simulation & Figure Scripts
│   ├── run_ber_simulation.py           # Monte Carlo BER vs SNR simulation script
│   └── generate_figures.py             # Publication Figure Generator
│
├── index.html                          # Web UI Application Entry Point
├── styles.css                          # Glassmorphic Dark UI Stylesheet
├── ofdmEngine.js                       # Client-side JavaScript OFDM Transceiver Engine
└── app.js                              # Web UI Controller & Canvas Renderers
```

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.9+
- Python Packages: `numpy`, `scipy`, `matplotlib`

### 1. Setup Environment
Clone the repository and install required packages:
```bash
git clone https://github.com/soumya-15-2005-byte/OFDM-Based-Wireless-Communication-System-with-Channel-Estimation-and-Equalization.git
cd OFDM-Based-Wireless-Communication-System-with-Channel-Estimation-and-Equalization
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
Verify mathematical correctness of constellation mapping, CP preservation, and channel estimation:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### 3. Run BER Monte Carlo Simulation
Execute the BER vs. SNR simulation runner:
```bash
python scripts/run_ber_simulation.py
```

### 4. Launch Interactive Web App
Start a local HTTP server to launch the interactive real-time visual simulator in your browser:
```bash
python -m http.server 8080
```
Open **`http://localhost:8080`** in your browser.

---

## 🌐 Interactive Web Simulator Features

1. **Live Control Panel**: Tweak Modulation (BPSK/QPSK), Channel Type (AWGN/Rayleigh), SNR ($0 \text{ to } 30 \text{ dB}$), Estimator (LS/MMSE), and Equalizer (ZF/MMSE).
2. **Sig2Sig Stage Flow**: Stage-by-stage block diagram tracing discrete signal transformations.
3. **Real-time Constellations Canvas**: Live scatter rendering of Transmitted reference points, Faded Noisy received symbols, and Equalized restored symbols.
4. **OFDM Waveform & CP Oscilloscope**: Visual demarcation of the Cyclic Prefix guard interval ($N_{cp} = 16$).
5. **Channel Response $|H(f)|$ Plot**: Displays True Channel Magnitude Response alongside Pilot LS Estimates and Interpolated Channel curves.
6. **Sig2Sig Image Transmission Demo**: Real-time image transmission showing pixel corruption under fading and restoration after equalization.

---

## 🤝 Pushing Changes to GitHub

To push your latest project files to your GitHub repository:

```bash
git init
git remote add origin https://github.com/soumya-15-2005-byte/OFDM-Based-Wireless-Communication-System-with-Channel-Estimation-and-Equalization.git
git branch -M main
git add .
git commit -m "Initial commit: Modular OFDM system with channel estimation, equalization, BER simulation, and Web UI"
git push -u origin main
```

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

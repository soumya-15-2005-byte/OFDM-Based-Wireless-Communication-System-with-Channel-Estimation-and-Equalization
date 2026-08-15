/**
 * OFDM Transceiver DSP Engine (Client-Side JS)
 * Project: OFDM-Based Wireless Communication System with Channel Estimation and Equalization
 */

class OFDMTransceiverJS {
  constructor(options = {}) {
    this.nFFT = options.nFFT || 64;
    this.cpLength = options.cpLength || 16;
    this.modulation = (options.modulation || 'QPSK').toUpperCase();
    this.pilotSpacing = options.pilotSpacing || 4;
    this.bitsPerSymbol = this.modulation === 'BPSK' ? 1 : 2;

    const all = Array.from({ length: this.nFFT }, (_, i) => i);
    this.pilotIndices = all.filter(i => i > 0 && i % this.pilotSpacing === 0);
    this.dataIndices = all.filter(i => i > 0 && !this.pilotIndices.includes(i));
    
    this.nPilots = this.pilotIndices.length;
    this.nData = this.dataIndices.length;
    
    this.pilotValue = this.modulation === 'BPSK' 
      ? { re: 1.0, im: 0.0 } 
      : { re: 1.0 / Math.SQRT2, im: 1.0 / Math.SQRT2 };

    const rawPdp = [1.0, Math.exp(-1.0 / 1.5), Math.exp(-2.0 / 1.5), Math.exp(-3.0 / 1.5)];
    const sumPdp = rawPdp.reduce((a, b) => a + b, 0);
    this.pdp = rawPdp.map(v => v / sumPdp);
  }

  generateBits(nSymbols) {
    const totalBits = nSymbols * this.nData * this.bitsPerSymbol;
    const bits = new Uint8Array(totalBits);
    for (let i = 0; i < totalBits; i++) {
      bits[i] = Math.random() < 0.5 ? 0 : 1;
    }
    return bits;
  }

  mapBitsToSymbols(bits) {
    const symbols = [];
    if (this.modulation === 'BPSK') {
      for (let i = 0; i < bits.length; i++) {
        symbols.push({ re: bits[i] === 0 ? 1.0 : -1.0, im: 0.0 });
      }
    } else {
      for (let i = 0; i < bits.length; i += 2) {
        const b0 = bits[i];
        const b1 = bits[i + 1];
        const re = (b0 === 0 ? 1.0 : -1.0) / Math.SQRT2;
        const im = (b1 === 0 ? 1.0 : -1.0) / Math.SQRT2;
        symbols.push({ re, im });
      }
    }
    return symbols;
  }

  demodulateSymbolsToBits(symbols) {
    const bits = [];
    if (this.modulation === 'BPSK') {
      for (let i = 0; i < symbols.length; i++) {
        bits.push(symbols[i].re < 0 ? 1 : 0);
      }
    } else {
      for (let i = 0; i < symbols.length; i++) {
        bits.push(symbols[i].re < 0 ? 1 : 0);
        bits.push(symbols[i].im < 0 ? 1 : 0);
      }
    }
    return new Uint8Array(bits);
  }

  ifft(freqFrame) {
    const N = freqFrame.length;
    const conjInput = freqFrame.map(c => ({ re: c.re, im: -c.im }));
    const fftOutput = this.fft(conjInput);
    return fftOutput.map(c => ({
      re: (c.re / N) * Math.sqrt(N),
      im: (-c.im / N) * Math.sqrt(N)
    }));
  }

  fft(x) {
    const N = x.length;
    if (N <= 1) return x;
    if ((N & (N - 1)) !== 0) {
      return this.dft(x);
    }
    const even = this.fft(x.filter((_, i) => i % 2 === 0));
    const odd = this.fft(x.filter((_, i) => i % 2 === 1));

    const X = new Array(N);
    for (let k = 0; k < N / 2; k++) {
      const angle = (-2 * Math.PI * k) / N;
      const twiddle = { re: Math.cos(angle), im: Math.sin(angle) };
      const ok = {
        re: odd[k].re * twiddle.re - odd[k].im * twiddle.im,
        im: odd[k].re * twiddle.im + odd[k].im * twiddle.re
      };
      X[k] = { re: even[k].re + ok.re, im: even[k].im + ok.im };
      X[k + N / 2] = { re: even[k].re - ok.re, im: even[k].im - ok.im };
    }
    return X;
  }

  dft(x) {
    const N = x.length;
    const X = new Array(N);
    for (let k = 0; k < N; k++) {
      let sumRe = 0, sumIm = 0;
      for (let n = 0; n < N; n++) {
        const angle = (-2 * Math.PI * k * n) / N;
        sumRe += x[n].re * Math.cos(angle) - x[n].im * Math.sin(angle);
        sumIm += x[n].re * Math.sin(angle) + x[n].im * Math.cos(angle);
      }
      X[k] = { re: sumRe, im: sumIm };
    }
    return X;
  }

  transmit(dataSymbols) {
    const nSymbols = Math.floor(dataSymbols.length / this.nData);
    const txTimeFrames = [];
    const txFreqFrames = [];

    for (let s = 0; s < nSymbols; s++) {
      const freqFrame = new Array(this.nFFT).fill(null).map(() => ({ re: 0, im: 0 }));
      for (let d = 0; d < this.nData; d++) {
        freqFrame[this.dataIndices[d]] = dataSymbols[s * this.nData + d];
      }
      for (let p = 0; p < this.nPilots; p++) {
        freqFrame[this.pilotIndices[p]] = { ...this.pilotValue };
      }
      freqFrame[0] = { re: 0, im: 0 };
      txFreqFrames.push(freqFrame);

      const timeFrame = this.ifft(freqFrame);
      const cp = timeFrame.slice(this.nFFT - this.cpLength);
      const timeWithCp = cp.concat(timeFrame);
      txTimeFrames.push(timeWithCp);
    }

    const txSignal = [];
    txTimeFrames.forEach(f => f.forEach(sample => txSignal.push(sample)));
    return { txSignal, txFreqFrames, txTimeFrames };
  }

  generateRayleighChannel() {
    const taps = [];
    for (let i = 0; i < this.pdp.length; i++) {
      const sigma = Math.sqrt(this.pdp[i] / 2.0);
      const u1 = Math.max(1e-10, Math.random());
      const u2 = Math.random();
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      const z1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2);
      taps.push({ re: z0 * sigma, im: z1 * sigma });
    }
    return taps;
  }

  passChannel(txSignal, channelTaps, snrDb, channelType = 'RAYLEIGH') {
    let rxConv = [];
    let actualTaps = channelTaps;

    if (channelType.toUpperCase() === 'AWGN') {
      rxConv = txSignal.map(s => ({ ...s }));
      actualTaps = [{ re: 1.0, im: 0.0 }];
    } else {
      const N = txSignal.length;
      const L = channelTaps.length;
      rxConv = new Array(N).fill(null).map(() => ({ re: 0, im: 0 }));

      for (let n = 0; n < N; n++) {
        let reSum = 0, imSum = 0;
        for (let l = 0; l < L; l++) {
          if (n - l >= 0) {
            const x = txSignal[n - l];
            const h = channelTaps[l];
            reSum += x.re * h.re - x.im * h.im;
            imSum += x.re * h.im + x.im * h.re;
          }
        }
        rxConv[n] = { re: reSum, im: imSum };
      }
    }

    let pSignal = 0;
    rxConv.forEach(s => pSignal += s.re * s.re + s.im * s.im);
    pSignal /= rxConv.length;

    const snrLinear = Math.pow(10, snrDb / 10.0);
    const noisePower = pSignal / Math.max(1e-12, snrLinear);
    const sigmaNoise = Math.sqrt(noisePower / 2.0);

    const rxNoisy = rxConv.map(s => {
      const u1 = Math.max(1e-10, Math.random());
      const u2 = Math.random();
      const n0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * sigmaNoise;
      const n1 = Math.sqrt(-2.0 * Math.log(u1)) * Math.sin(2.0 * Math.PI * u2) * sigmaNoise;
      return { re: s.re + n0, im: s.im + n1 };
    });

    return { rxNoisy, actualTaps, noisePower };
  }

  receive(rxSignal, nSymbols) {
    const frameLen = this.nFFT + this.cpLength;
    const rxFreqFrames = [];

    for (let s = 0; s < nSymbols; s++) {
      const frameWithCp = rxSignal.slice(s * frameLen, (s + 1) * frameLen);
      if (frameWithCp.length < frameLen) break;
      const timeFrame = frameWithCp.slice(this.cpLength);
      
      const freqRaw = this.fft(timeFrame);
      const freqFrame = freqRaw.map(c => ({
        re: c.re / Math.sqrt(this.nFFT),
        im: c.im / Math.sqrt(this.nFFT)
      }));
      rxFreqFrames.push(freqFrame);
    }
    return rxFreqFrames;
  }

  estimateChannel(rxFreqFrames, method = 'LS', noisePower = 1e-3) {
    const HEst = [];
    const nSymbols = rxFreqFrames.length;

    for (let s = 0; s < nSymbols; s++) {
      const rxFreq = rxFreqFrames[s];
      const hPilots = [];

      for (let p = 0; p < this.nPilots; p++) {
        const pIdx = this.pilotIndices[p];
        const Y = rxFreq[pIdx];
        const X = this.pilotValue;
        const xMagSq = X.re * X.re + X.im * X.im;
        const hLS = {
          re: (Y.re * X.re + Y.im * X.im) / xMagSq,
          im: (Y.im * X.re - Y.re * X.im) / xMagSq
        };
        hPilots.push({ idx: pIdx, ...hLS });
      }

      const hFrame = new Array(this.nFFT);
      for (let k = 0; k < this.nFFT; k++) {
        let left = hPilots[0];
        let right = hPilots[hPilots.length - 1];

        for (let p = 0; p < hPilots.length - 1; p++) {
          if (k >= hPilots[p].idx && k <= hPilots[p + 1].idx) {
            left = hPilots[p];
            right = hPilots[p + 1];
            break;
          }
        }

        let interpRe, interpIm;
        if (left.idx === right.idx) {
          interpRe = left.re;
          interpIm = left.im;
        } else {
          const alpha = (k - left.idx) / (right.idx - left.idx);
          interpRe = left.re + alpha * (right.re - left.re);
          interpIm = left.im + alpha * (right.im - left.im);
        }

        if (method.toUpperCase() === 'MMSE') {
          const magSq = interpRe * interpRe + interpIm * interpIm;
          const factor = magSq / (magSq + noisePower);
          interpRe *= factor;
          interpIm *= factor;
        }

        hFrame[k] = { re: interpRe, im: interpIm };
      }
      HEst.push(hFrame);
    }
    return HEst;
  }

  equalize(rxFreqFrames, HEst, method = 'ZF', snrDb = 20) {
    const snrLinear = Math.pow(10, snrDb / 10.0);
    const equalizedDataSymbols = [];
    const equalizedFreqFrames = [];

    for (let s = 0; s < rxFreqFrames.length; s++) {
      const Y = rxFreqFrames[s];
      const H = HEst[s];
      const eqFrame = new Array(this.nFFT);

      for (let k = 0; k < this.nFFT; k++) {
        const yk = Y[k];
        const hk = H[k];
        const hMagSq = hk.re * hk.re + hk.im * hk.im;

        let xHat = { re: 0, im: 0 };
        if (method.toUpperCase() === 'ZF') {
          const denom = Math.max(1e-10, hMagSq);
          xHat = {
            re: (yk.re * hk.re + yk.im * hk.im) / denom,
            im: (yk.im * hk.re - yk.re * hk.im) / denom
          };
        } else {
          const denom = hMagSq + (1.0 / snrLinear);
          xHat = {
            re: (yk.re * hk.re + yk.im * hk.im) / denom,
            im: (yk.im * hk.re - yk.re * hk.im) / denom
          };
        }
        eqFrame[k] = xHat;
      }

      equalizedFreqFrames.push(eqFrame);
      for (let d = 0; d < this.nData; d++) {
        equalizedDataSymbols.push(eqFrame[this.dataIndices[d]]);
      }
    }
    return { equalizedDataSymbols, equalizedFreqFrames };
  }

  computeBER(txBits, rxBits) {
    let bitErrors = 0;
    const len = Math.min(txBits.length, rxBits.length);
    for (let i = 0; i < len; i++) {
      if (txBits[i] !== rxBits[i]) bitErrors++;
    }
    return { ber: bitErrors / len, bitErrors, totalBits: len };
  }
}

if (typeof window !== 'undefined') {
  window.OFDMTransceiverJS = OFDMTransceiverJS;
}

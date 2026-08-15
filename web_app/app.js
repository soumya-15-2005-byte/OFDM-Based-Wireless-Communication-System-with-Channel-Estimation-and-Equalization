/**
 * Main Application Orchestrator & Real-Time Canvas Renderers
 * Project: OFDM-Based Wireless Communication System with Channel Estimation and Equalization
 */

document.addEventListener('DOMContentLoaded', () => {
  const modulationSelect = document.getElementById('modulationSelect');
  const channelSelect = document.getElementById('channelSelect');
  const snrRange = document.getElementById('snrRange');
  const snrVal = document.getElementById('snrVal');
  const estimationSelect = document.getElementById('estimationSelect');
  const equalizationSelect = document.getElementById('equalizationSelect');
  const fftSelect = document.getElementById('fftSelect');
  const btnRunBerSweep = document.getElementById('btnRunBerSweep');

  const statBer = document.getElementById('statBer');
  const statErrors = document.getElementById('statErrors');
  const statSnr = document.getElementById('statSnr');
  const statRate = document.getElementById('statRate');

  const constellationCanvas = document.getElementById('constellationCanvas');
  const waveformCanvas = document.getElementById('waveformCanvas');
  const channelCanvas = document.getElementById('channelCanvas');

  const imgTxCanvas = document.getElementById('imgTxCanvas');
  const imgRxCanvas = document.getElementById('imgRxCanvas');
  const imgEqCanvas = document.getElementById('imgEqCanvas');

  let simParams = {
    nFFT: parseInt(fftSelect.value),
    cpLength: 16,
    modulation: modulationSelect.value,
    channelType: channelSelect.value,
    snrDb: parseFloat(snrRange.value),
    estMethod: estimationSelect.value,
    eqMethod: equalizationSelect.value
  };

  let berChart = null;

  initChart();
  initImageDemo();
  runRealtimeSimulation();

  snrRange.addEventListener('input', (e) => {
    simParams.snrDb = parseFloat(e.target.value);
    snrVal.textContent = `${simParams.snrDb} dB`;
    statSnr.textContent = `${simParams.snrDb} dB`;
    runRealtimeSimulation();
  });

  modulationSelect.addEventListener('change', (e) => {
    simParams.modulation = e.target.value;
    runRealtimeSimulation();
  });

  channelSelect.addEventListener('change', (e) => {
    simParams.channelType = e.target.value;
    runRealtimeSimulation();
  });

  estimationSelect.addEventListener('change', (e) => {
    simParams.estMethod = e.target.value;
    runRealtimeSimulation();
  });

  equalizationSelect.addEventListener('change', (e) => {
    simParams.eqMethod = e.target.value;
    runRealtimeSimulation();
  });

  fftSelect.addEventListener('change', (e) => {
    simParams.nFFT = parseInt(e.target.value);
    simParams.cpLength = Math.max(16, Math.floor(simParams.nFFT / 4));
    runRealtimeSimulation();
  });

  btnRunBerSweep.addEventListener('click', () => {
    runBerSweep();
  });

  function runRealtimeSimulation() {
    const tx = new OFDMTransceiverJS({
      nFFT: simParams.nFFT,
      cpLength: simParams.cpLength,
      modulation: simParams.modulation,
      pilotSpacing: 4
    });

    const nSymbols = 40;
    const bits = tx.generateBits(nSymbols);
    const dataSymbols = tx.mapBitsToSymbols(bits);

    const { txSignal, txFreqFrames, txTimeFrames } = tx.transmit(dataSymbols);
    const taps = tx.generateRayleighChannel();
    const { rxNoisy, actualTaps, noisePower } = tx.passChannel(txSignal, taps, simParams.snrDb, simParams.channelType);

    const rxFreqFrames = tx.receive(rxNoisy, nSymbols);
    const HEst = tx.estimateChannel(rxFreqFrames, simParams.estMethod, noisePower);
    const { equalizedDataSymbols, equalizedFreqFrames } = tx.equalize(rxFreqFrames, HEst, simParams.eqMethod, simParams.snrDb);

    const rxBits = tx.demodulateSymbolsToBits(equalizedDataSymbols);
    const { ber, bitErrors, totalBits } = tx.computeBER(bits, rxBits);

    statBer.textContent = ber.toFixed(5);
    statBer.className = ber < 0.01 ? 'val green' : ber < 0.08 ? 'val amber' : 'val red';
    statErrors.textContent = `${bitErrors} / ${totalBits}`;
    
    const eff = (simParams.modulation === 'BPSK' ? 1.0 : 2.0) * (tx.nData / tx.nFFT);
    statRate.textContent = `${eff.toFixed(2)} bits/Hz`;

    renderConstellations(dataSymbols, rxFreqFrames, equalizedDataSymbols, tx);
    renderWaveform(txTimeFrames[0], simParams.cpLength);
    renderChannelResponse(tx, actualTaps, HEst[0]);
    updateImageTransmissionDemo(tx, taps, simParams.snrDb, simParams.channelType, simParams.estMethod, simParams.eqMethod);
  }

  function renderConstellations(txSyms, rxFrames, eqSyms, tx) {
    const ctx = constellationCanvas.getContext('2d');
    const w = constellationCanvas.width = constellationCanvas.parentElement.clientWidth;
    const h = constellationCanvas.height = constellationCanvas.parentElement.clientHeight;

    ctx.clearRect(0, 0, w, h);

    const margin = 30;
    const size = Math.min(w, h) - margin * 2;
    const cx = w / 2;
    const cy = h / 2;
    const scale = size / 3.5;

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - size / 2, cy); ctx.lineTo(cx + size / 2, cy);
    ctx.moveTo(cx, cy - size / 2); ctx.lineTo(cx, cy + size / 2);
    ctx.stroke();

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.beginPath();
    ctx.arc(cx, cy, scale, 0, 2 * Math.PI);
    ctx.stroke();

    const rxDataSyms = [];
    rxFrames.forEach(frame => {
      tx.dataIndices.forEach(idx => rxDataSyms.push(frame[idx]));
    });

    ctx.fillStyle = 'rgba(239, 68, 68, 0.4)';
    rxDataSyms.slice(0, 300).forEach(s => {
      const px = cx + s.re * scale;
      const py = cy - s.im * scale;
      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.fillStyle = 'rgba(16, 185, 129, 0.85)';
    eqSyms.slice(0, 300).forEach(s => {
      const px = cx + s.re * scale;
      const py = cy - s.im * scale;
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.fillStyle = '#60a5fa';
    ctx.shadowColor = '#3b82f6';
    ctx.shadowBlur = 8;
    txSyms.slice(0, 100).forEach(s => {
      const px = cx + s.re * scale;
      const py = cy - s.im * scale;
      ctx.beginPath();
      ctx.arc(px, py, 5, 0, 2 * Math.PI);
      ctx.fill();
    });
    ctx.shadowBlur = 0;
  }

  function renderWaveform(frameWithCp, cpLen) {
    const ctx = waveformCanvas.getContext('2d');
    const w = waveformCanvas.width = waveformCanvas.parentElement.clientWidth;
    const h = waveformCanvas.height = waveformCanvas.parentElement.clientHeight;

    ctx.clearRect(0, 0, w, h);

    const N = frameWithCp.length;
    const stepX = w / N;
    const cy = h / 2;
    const ampScale = (h / 3) / 1.5;

    const cpWidth = cpLen * stepX;
    ctx.fillStyle = 'rgba(6, 182, 212, 0.15)';
    ctx.fillRect(0, 0, cpWidth, h);

    ctx.strokeStyle = 'rgba(6, 182, 212, 0.6)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(cpWidth, 0); ctx.lineTo(cpWidth, h);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#06b6d4';
    ctx.font = '11px sans-serif';
    ctx.fillText(`Cyclic Prefix (${cpLen} samples)`, 10, 20);
    ctx.fillText('Payload', cpWidth + 10, 20);

    ctx.strokeStyle = '#f8fafc';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const px = i * stepX;
      const py = cy - frameWithCp[i].re * ampScale;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.strokeStyle = 'rgba(167, 139, 250, 0.6)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const px = i * stepX;
      const py = cy - frameWithCp[i].im * ampScale;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  function renderChannelResponse(tx, actualTaps, HEstFrame) {
    const ctx = channelCanvas.getContext('2d');
    const w = channelCanvas.width = channelCanvas.parentElement.clientWidth;
    const h = channelCanvas.height = channelCanvas.parentElement.clientHeight;

    ctx.clearRect(0, 0, w, h);

    const HTrue = tx.fft(actualTaps.concat(new Array(tx.nFFT - actualTaps.length).fill({ re: 0, im: 0 })));
    const HTrueMag = HTrue.map(c => Math.sqrt(c.re * c.re + c.im * c.im));
    const HEstMag = HEstFrame.map(c => Math.sqrt(c.re * c.re + c.im * c.im));

    const maxMag = Math.max(...HTrueMag, ...HEstMag, 1.5);
    const margin = 30;
    const drawW = w - margin * 2;
    const drawH = h - margin * 2;
    const stepX = drawW / (tx.nFFT - 1);

    ctx.strokeStyle = '#3b82f6';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    for (let k = 0; k < tx.nFFT; k++) {
      const px = margin + k * stepX;
      const py = h - margin - (HTrueMag[k] / maxMag) * drawH;
      if (k === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 1.8;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    for (let k = 0; k < tx.nFFT; k++) {
      const px = margin + k * stepX;
      const py = h - margin - (HEstMag[k] / maxMag) * drawH;
      if (k === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#ef4444';
    tx.pilotIndices.forEach(idx => {
      const px = margin + idx * stepX;
      const py = h - margin - (HEstMag[idx] / maxMag) * drawH;
      ctx.beginPath();
      ctx.arc(px, py, 4, 0, 2 * Math.PI);
      ctx.fill();
    });
  }

  function initChart() {
    const ctx = document.getElementById('berChart').getContext('2d');
    const snrDbRange = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30];
    
    const awgnTheory = snrDbRange.map(snr => {
      const g = Math.pow(10, snr / 10.0);
      return Math.max(1e-5, 0.5 * Math.exp(-g));
    });

    const rayleighTheory = snrDbRange.map(snr => {
      const g = Math.pow(10, snr / 10.0);
      return 0.5 * (1 - Math.sqrt(g / (1 + g)));
    });

    berChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: snrDbRange,
        datasets: [
          { label: 'AWGN Theory', data: awgnTheory, borderColor: '#94a3b8', borderDash: [5, 5], pointRadius: 0 },
          { label: 'Rayleigh Theory', data: rayleighTheory, borderColor: '#ec4899', borderDash: [4, 4], pointRadius: 0 },
          { label: 'Current Setup (Simulated)', data: [], borderColor: '#10b981', backgroundColor: '#10b981', pointRadius: 5, borderWidth: 2 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: 'SNR (dB)', color: '#94a3b8' }, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } },
          y: { type: 'logarithmic', title: { display: true, text: 'Bit Error Rate (BER)', color: '#94a3b8' }, min: 1e-4, max: 1.0, grid: { color: 'rgba(255, 255, 255, 0.05)' }, ticks: { color: '#94a3b8' } }
        },
        plugins: { legend: { labels: { color: '#e2e8f0', font: { size: 11 } } } }
      }
    });
  }

  function runBerSweep() {
    const snrRange = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30];
    const simulatedBer = [];

    const tx = new OFDMTransceiverJS({
      nFFT: simParams.nFFT,
      cpLength: simParams.cpLength,
      modulation: simParams.modulation,
      pilotSpacing: 4
    });

    const nSymbols = 120;
    const bits = tx.generateBits(nSymbols);
    const dataSymbols = tx.mapBitsToSymbols(bits);
    const { txSignal } = tx.transmit(dataSymbols);

    snrRange.forEach(snr => {
      const taps = tx.generateRayleighChannel();
      const { rxNoisy, noisePower } = tx.passChannel(txSignal, taps, snr, simParams.channelType);
      const rxFreqFrames = tx.receive(rxNoisy, nSymbols);
      const HEst = tx.estimateChannel(rxFreqFrames, simParams.estMethod, noisePower);
      const { equalizedDataSymbols } = tx.equalize(rxFreqFrames, HEst, simParams.eqMethod, snr);
      const rxBits = tx.demodulateSymbolsToBits(equalizedDataSymbols);
      const { ber } = tx.computeBER(bits, rxBits);
      simulatedBer.push(Math.max(1e-4, ber));
    });

    berChart.data.datasets[2].data = simulatedBer;
    berChart.update();
  }

  function initImageDemo() {
    const ctxTx = imgTxCanvas.getContext('2d');
    ctxTx.fillStyle = '#0f172a';
    ctxTx.fillRect(0, 0, 128, 128);

    ctxTx.fillStyle = '#38bdf8';
    ctxTx.font = 'bold 22px sans-serif';
    ctxTx.fillText('OFDM', 24, 55);

    ctxTx.fillStyle = '#34d399';
    ctxTx.font = '12px sans-serif';
    ctxTx.fillText('Sig2Sig Demo', 24, 75);

    ctxTx.strokeStyle = '#f59e0b';
    ctxTx.lineWidth = 3;
    ctxTx.strokeRect(10, 10, 108, 108);
  }

  function updateImageTransmissionDemo(tx, taps, snrDb, channelType, estMethod, eqMethod) {
    const ctxTx = imgTxCanvas.getContext('2d');
    const ctxRx = imgRxCanvas.getContext('2d');
    const ctxEq = imgEqCanvas.getContext('2d');

    const imgData = ctxTx.getImageData(0, 0, 128, 128);
    const pixels = imgData.data;

    const bits = [];
    for (let i = 0; i < pixels.length; i += 4) {
      const r = pixels[i];
      const g = pixels[i + 1];
      const b = pixels[i + 2];
      for (let bIdx = 7; bIdx >= 0; bIdx--) bits.push((r >> bIdx) & 1);
      for (let bIdx = 7; bIdx >= 0; bIdx--) bits.push((g >> bIdx) & 1);
      for (let bIdx = 7; bIdx >= 0; bIdx--) bits.push((b >> bIdx) & 1);
    }

    const nSymbols = Math.ceil(bits.length / (tx.nData * tx.bitsPerSymbol));
    const paddedBits = new Uint8Array(nSymbols * tx.nData * tx.bitsPerSymbol);
    paddedBits.set(bits);

    const txSyms = tx.mapBitsToSymbols(paddedBits);
    const { txSignal } = tx.transmit(txSyms);
    const { rxNoisy, noisePower } = tx.passChannel(txSignal, taps, snrDb, channelType);

    const rxFreq = tx.receive(rxNoisy, nSymbols);

    const rawDataSyms = [];
    rxFreq.forEach(f => tx.dataIndices.forEach(idx => rawDataSyms.push(f[idx])));
    const rxBitsRaw = tx.demodulateSymbolsToBits(rawDataSyms);

    const HEst = tx.estimateChannel(rxFreq, estMethod, noisePower);
    const { equalizedDataSymbols } = tx.equalize(rxFreq, HEst, eqMethod, snrDb);
    const rxBitsEq = tx.demodulateSymbolsToBits(equalizedDataSymbols);

    const rxImgData = ctxRx.createImageData(128, 128);
    const eqImgData = ctxEq.createImageData(128, 128);

    let bitPtr = 0;
    for (let i = 0; i < 128 * 128 * 4; i += 4) {
      let rRaw = 0, gRaw = 0, bRaw = 0;
      for (let bIdx = 7; bIdx >= 0; bIdx--) rRaw |= (rxBitsRaw[bitPtr++] << bIdx);
      for (let bIdx = 7; bIdx >= 0; bIdx--) gRaw |= (rxBitsRaw[bitPtr++] << bIdx);
      for (let bIdx = 7; bIdx >= 0; bIdx--) bRaw |= (rxBitsRaw[bitPtr++] << bIdx);

      rxImgData.data[i] = rRaw;
      rxImgData.data[i + 1] = gRaw;
      rxImgData.data[i + 2] = bRaw;
      rxImgData.data[i + 3] = 255;
    }

    bitPtr = 0;
    for (let i = 0; i < 128 * 128 * 4; i += 4) {
      let rEq = 0, gEq = 0, bEq = 0;
      for (let bIdx = 7; bIdx >= 0; bIdx--) rEq |= (rxBitsEq[bitPtr++] << bIdx);
      for (let bIdx = 7; bIdx >= 0; bIdx--) gEq |= (rxBitsEq[bitPtr++] << bIdx);
      for (let bIdx = 7; bIdx >= 0; bIdx--) bEq |= (rxBitsEq[bitPtr++] << bIdx);

      eqImgData.data[i] = rEq;
      eqImgData.data[i + 1] = gEq;
      eqImgData.data[i + 2] = bEq;
      eqImgData.data[i + 3] = 255;
    }

    ctxRx.putImageData(rxImgData, 0, 0);
    ctxEq.putImageData(eqImgData, 0, 0);
  }
});

export type MLOptimizerType =
  | 'batch_gd'
  | 'sgd'
  | 'minibatch_gd'
  | 'momentum'
  | 'rmsprop'
  | 'adam';

export interface DatasetTable {
  name: string;
  headers: string[];
  rows: Record<string, number>[];
}

export interface RegressionMetrics {
  mae: number;
  mse: number;
  r2: number;
}

export interface EpochRecord {
  epoch: number;
  trainMse: number;
  valMse: number;
  weightsNorm: number;
  gradNorm: number;
  weightsPreview: number[];
  bias: number;
}

export interface MLTrainingResult {
  optimizer: MLOptimizerType;
  epochs: number;
  learningRate: number;
  batchSize: number;
  standardized: boolean;
  degree: number;
  l2Lambda: number;
  featureNames: string[];
  targetName: string;
  weights: number[];
  bias: number;
  rawWeights: number[];
  rawBias: number;
  history: EpochRecord[];
  trainMetrics: RegressionMetrics;
  valMetrics: RegressionMetrics;
  yTrainActual: number[];
  yTrainPred: number[];
  yValActual: number[];
  yValPred: number[];
  xTrainPrimary: number[];
  xValPrimary: number[];
  olsBenchmark: {
    weights: number[];
    bias: number;
    trainMetrics: RegressionMetrics;
    valMetrics: RegressionMetrics;
  } | null;
  status: 'converged' | 'completed' | 'diverged';
  statusMessage: string;
}

function createRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s += 0x6d2b79f5;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t ^= t + Math.imul(t ^ (t >>> 7), 61 | t);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussianNoise(rng: () => number): number {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

export function getBuiltinDatasets(): DatasetTable[] {
  const rng1 = createRng(42);
  const studyRows: Record<string, number>[] = [];
  for (let i = 0; i < 60; i++) {
    const hours = 1 + (i / 59) * 9;
    const attendance = 65 + rng1() * 35;
    const score = 32 + 5.6 * hours + 0.18 * attendance + gaussianNoise(rng1) * 3.2;
    studyRows.push({
      study_hours: Number(hours.toFixed(2)),
      attendance_pct: Number(attendance.toFixed(1)),
      exam_score: Number(Math.min(100, Math.max(0, score)).toFixed(2)),
    });
  }

  const rng2 = createRng(108);
  const housingRows: Record<string, number>[] = [];
  for (let i = 0; i < 80; i++) {
    const area = 650 + rng2() * 2600;
    const rooms = Math.max(1, Math.round(area / 580 + (rng2() - 0.5) * 1.4));
    const price = 45 + 0.145 * area + 12.5 * rooms + gaussianNoise(rng2) * 18;
    housingRows.push({
      area_sqft: Math.round(area),
      rooms,
      price_k_usd: Number(price.toFixed(2)),
    });
  }

  const rng3 = createRng(777);
  const waveRows: Record<string, number>[] = [];
  for (let i = 0; i < 24; i++) {
    const x = -2.8 + (i / 23) * 5.6;
    const y = Math.sin(1.25 * x) + 0.25 * x + gaussianNoise(rng3) * 0.42;
    waveRows.push({
      x: Number(x.toFixed(3)),
      y: Number(y.toFixed(4)),
    });
  }

  return [
    {
      name: 'Study Hours & Attendance vs Exam Score',
      headers: ['study_hours', 'attendance_pct', 'exam_score'],
      rows: studyRows,
    },
    {
      name: 'Housing Area & Rooms vs Price ($k)',
      headers: ['area_sqft', 'rooms', 'price_k_usd'],
      rows: housingRows,
    },
    {
      name: 'Nonlinear Signal (Overfitting / Underfitting Lab)',
      headers: ['x', 'y'],
      rows: waveRows,
    },
  ];
}

export function parseCsvDataset(csvText: string, fileName = 'Uploaded CSV'): DatasetTable {
  const lines = csvText
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length < 4) {
    throw new Error('CSV must contain a header row and at least 3 data rows.');
  }

  const headers = lines[0].split(',').map((h) => h.trim().replace(/^"|"$/g, ''));
  if (headers.length < 2) {
    throw new Error('CSV must contain at least 2 columns (input and target).');
  }

  const numericMask = headers.map(() => true);
  const rawRows: string[][] = [];

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map((c) => c.trim().replace(/^"|"$/g, ''));
    if (cols.length !== headers.length) continue;
    rawRows.push(cols);
    cols.forEach((c, idx) => {
      const val = Number(c);
      if (c === '' || !Number.isFinite(val)) {
        numericMask[idx] = false;
      }
    });
  }

  const numericHeaders = headers.filter((_, idx) => numericMask[idx]);
  if (numericHeaders.length < 2) {
    throw new Error('CSV needs at least 2 strictly numeric columns for regression.');
  }

  const rows: Record<string, number>[] = rawRows.map((cols) => {
    const rec: Record<string, number> = {};
    headers.forEach((h, idx) => {
      if (numericMask[idx]) {
        rec[h] = Number(cols[idx]);
      }
    });
    return rec;
  });

  if (rows.length < 4) {
    throw new Error('Not enough valid numeric rows in CSV (minimum 4 required).');
  }

  return {
    name: fileName,
    headers: numericHeaders,
    rows,
  };
}

export function computeRegressionMetrics(
  yTrue: number[],
  yPred: number[]
): RegressionMetrics {
  const n = yTrue.length;
  if (n === 0) return { mae: NaN, mse: NaN, r2: NaN };

  let absSum = 0;
  let sqSum = 0;
  let meanY = 0;
  for (let i = 0; i < n; i++) meanY += yTrue[i];
  meanY /= n;

  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    const err = yTrue[i] - yPred[i];
    absSum += Math.abs(err);
    sqSum += err * err;
    const diff = yTrue[i] - meanY;
    ssTot += diff * diff;
  }

  const mae = absSum / n;
  const mse = sqSum / n;
  const r2 = ssTot > 1e-15 ? 1 - sqSum / ssTot : sqSum <= 1e-15 ? 1 : 0;

  return { mae, mse, r2 };
}

/**
 * Solves (A + ridge*I) * x = b via Gaussian elimination with partial pivoting.
 */
function solveLinearSystem(A: number[][], b: number[]): number[] | null {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    let maxVal = Math.abs(M[col][col]);
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > maxVal) {
        maxVal = Math.abs(M[r][col]);
        pivotRow = r;
      }
    }
    if (maxVal < 1e-12) return null;
    if (pivotRow !== col) {
      const tmp = M[col];
      M[col] = M[pivotRow];
      M[pivotRow] = tmp;
    }

    const pivot = M[col][col];
    for (let j = col; j <= n; j++) M[col][j] /= pivot;

    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = M[r][col];
      for (let j = col; j <= n; j++) {
        M[r][j] -= factor * M[col][j];
      }
    }
  }

  return M.map((row) => row[n]);
}

function buildFeatureMatrix(
  rows: Record<string, number>[],
  inputCols: string[],
  degree: number
): { X: number[][]; expandedNames: string[] } {
  if (degree === 1 || inputCols.length > 1) {
    return {
      X: rows.map((r) => inputCols.map((c) => r[c])),
      expandedNames: [...inputCols],
    };
  }
  const col = inputCols[0];
  const expandedNames: string[] = [];
  for (let d = 1; d <= degree; d++) {
    expandedNames.push(d === 1 ? col : `${col}^${d}`);
  }
  const X = rows.map((r) => {
    const val = r[col];
    const feats: number[] = [];
    for (let d = 1; d <= degree; d++) {
      feats.push(Math.pow(val, d));
    }
    return feats;
  });
  return { X, expandedNames };
}

export interface MLTrainingConfig {
  dataset: DatasetTable;
  inputCols: string[];
  targetCol: string;
  optimizer: MLOptimizerType;
  learningRate: number;
  epochs: number;
  batchSize?: number;
  trainRatio?: number;
  standardize?: boolean;
  degree?: number;
  l2Lambda?: number;
  seed?: number;
}

export function trainLinearRegressionFromScratch(
  config: MLTrainingConfig
): MLTrainingResult {
  const {
    dataset,
    inputCols,
    targetCol,
    optimizer,
    learningRate,
    epochs,
    batchSize = 16,
    trainRatio = 0.75,
    standardize = true,
    degree = 1,
    l2Lambda = 0,
    seed = 2026,
  } = config;

  if (inputCols.length === 0) {
    throw new Error('Select at least one input feature column.');
  }
  if (!Number.isFinite(learningRate) || learningRate <= 0) {
    throw new Error('Learning rate must be strictly positive.');
  }
  if (!Number.isInteger(epochs) || epochs < 1 || epochs > 5000) {
    throw new Error('Epochs must be an integer between 1 and 5000.');
  }
  if (!Number.isFinite(l2Lambda) || l2Lambda < 0) {
    throw new Error('L2 regularization λ must be non-negative (λ >= 0).');
  }
  if (!Number.isInteger(degree) || degree < 1 || degree > 12) {
    throw new Error('Polynomial degree must be an integer between 1 and 12.');
  }
  if (!Number.isInteger(batchSize) || batchSize < 1) {
    throw new Error('Batch size must be a positive integer.');
  }
  if (!Number.isFinite(trainRatio) || trainRatio <= 0.05 || trainRatio >= 0.95) {
    throw new Error('Train split ratio must lie strictly in (0.05, 0.95).');
  }

  // Deterministic shuffle & train/val split
  const rng = createRng(seed);
  const indices = dataset.rows.map((_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    const tmp = indices[i];
    indices[i] = indices[j];
    indices[j] = tmp;
  }

  const shuffledRows = indices.map((i) => dataset.rows[i]);
  const splitIdx = Math.max(
    2,
    Math.min(shuffledRows.length - 2, Math.floor(shuffledRows.length * trainRatio))
  );
  const trainRows = shuffledRows.slice(0, splitIdx);
  const valRows = shuffledRows.slice(splitIdx);

  const { X: XTrainRaw, expandedNames } = buildFeatureMatrix(
    trainRows,
    inputCols,
    degree
  );
  const { X: XValRaw } = buildFeatureMatrix(valRows, inputCols, degree);

  const yTrain = trainRows.map((r) => r[targetCol]);
  const yVal = valRows.map((r) => r[targetCol]);
  const nTrain = XTrainRaw.length;
  const nVal = XValRaw.length;
  const p = expandedNames.length;

  // Compute training-set statistics for standardization
  const means = new Array<number>(p).fill(0);
  const stds = new Array<number>(p).fill(1);
  if (standardize) {
    for (let j = 0; j < p; j++) {
      let sum = 0;
      for (let i = 0; i < nTrain; i++) sum += XTrainRaw[i][j];
      means[j] = sum / nTrain;

      let sq = 0;
      for (let i = 0; i < nTrain; i++) {
        const d = XTrainRaw[i][j] - means[j];
        sq += d * d;
      }
      const s = Math.sqrt(sq / nTrain);
      stds[j] = s > 1e-10 ? s : 1;
    }
  }

  const XTrain = XTrainRaw.map((row) =>
    row.map((v, j) => (standardize ? (v - means[j]) / stds[j] : v))
  );
  const XVal = XValRaw.map((row) =>
    row.map((v, j) => (standardize ? (v - means[j]) / stds[j] : v))
  );

  // Initialize parameters at zero
  const w = new Array<number>(p).fill(0);
  let b = 0;

  // Optimizer state
  const vw = new Array<number>(p).fill(0);
  let vb = 0;
  const sw = new Array<number>(p).fill(0);
  let sb = 0;
  const mw = new Array<number>(p).fill(0);
  let mb = 0;
  let adamStep = 0;

  const beta = 0.9;
  const rho = 0.9;
  const beta1 = 0.9;
  const beta2 = 0.999;
  const eps = 1e-8;

  const predict = (X: number[][]) =>
    X.map((row) => {
      let s = b;
      for (let j = 0; j < p; j++) s += w[j] * row[j];
      return s;
    });

  const history: EpochRecord[] = [];
  let status: MLTrainingResult['status'] = 'completed';
  let statusMessage = `Completed ${epochs} epochs with ${optimizer.toUpperCase()}.`;

  const applyBatchUpdate = (batchIndices: number[]) => {
    const m = batchIndices.length;
    const gw = new Array<number>(p).fill(0);
    let gb = 0;

    for (const idx of batchIndices) {
      const row = XTrain[idx];
      let pred = b;
      for (let j = 0; j < p; j++) pred += w[j] * row[j];
      const err = pred - yTrain[idx];
      gb += (2 / m) * err;
      for (let j = 0; j < p; j++) {
        gw[j] += (2 / m) * err * row[j];
      }
    }

    for (let j = 0; j < p; j++) {
      gw[j] += 2 * l2Lambda * w[j];
    }

    if (optimizer === 'batch_gd' || optimizer === 'sgd' || optimizer === 'minibatch_gd') {
      for (let j = 0; j < p; j++) w[j] -= learningRate * gw[j];
      b -= learningRate * gb;
    } else if (optimizer === 'momentum') {
      for (let j = 0; j < p; j++) {
        vw[j] = beta * vw[j] - learningRate * gw[j];
        w[j] += vw[j];
      }
      vb = beta * vb - learningRate * gb;
      b += vb;
    } else if (optimizer === 'rmsprop') {
      for (let j = 0; j < p; j++) {
        sw[j] = rho * sw[j] + (1 - rho) * gw[j] * gw[j];
        w[j] -= (learningRate / (Math.sqrt(sw[j]) + eps)) * gw[j];
      }
      sb = rho * sb + (1 - rho) * gb * gb;
      b -= (learningRate / (Math.sqrt(sb) + eps)) * gb;
    } else if (optimizer === 'adam') {
      adamStep += 1;
      for (let j = 0; j < p; j++) {
        mw[j] = beta1 * mw[j] + (1 - beta1) * gw[j];
        sw[j] = beta2 * sw[j] + (1 - beta2) * gw[j] * gw[j];
        const mHat = mw[j] / (1 - Math.pow(beta1, adamStep));
        const sHat = sw[j] / (1 - Math.pow(beta2, adamStep));
        w[j] -= (learningRate / (Math.sqrt(sHat) + eps)) * mHat;
      }
      mb = beta1 * mb + (1 - beta1) * gb;
      sb = beta2 * sb + (1 - beta2) * gb * gb;
      const mbHat = mb / (1 - Math.pow(beta1, adamStep));
      const sbHat = sb / (1 - Math.pow(beta2, adamStep));
      b -= (learningRate / (Math.sqrt(sbHat) + eps)) * mbHat;
    }

    return Math.hypot(...gw, gb);
  };

  const computeFullGradNorm = () => {
    const gw = new Array<number>(p).fill(0);
    let gb = 0;
    for (let i = 0; i < nTrain; i++) {
      const row = XTrain[i];
      let pred = b;
      for (let j = 0; j < p; j++) pred += w[j] * row[j];
      const err = pred - yTrain[i];
      gb += (2 / nTrain) * err;
      for (let j = 0; j < p; j++) {
        gw[j] += (2 / nTrain) * err * row[j];
      }
    }
    for (let j = 0; j < p; j++) {
      gw[j] += 2 * l2Lambda * w[j];
    }
    return Math.hypot(...gw, gb);
  };

  const allTrainIndices = Array.from({ length: nTrain }, (_, i) => i);

  // Epoch 0: initial state before any gradient updates
  {
    const initTrainMse = computeRegressionMetrics(yTrain, predict(XTrain)).mse;
    const initValMse = computeRegressionMetrics(yVal, predict(XVal)).mse;
    history.push({
      epoch: 0,
      trainMse: initTrainMse,
      valMse: initValMse,
      weightsNorm: Math.hypot(...w),
      gradNorm: computeFullGradNorm(),
      weightsPreview: [...w],
      bias: b,
    });
  }

  for (let epoch = 1; epoch <= epochs; epoch++) {
    let gradNorm = 0;
    if (optimizer === 'sgd') {
      const order = [...allTrainIndices];
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
      for (const idx of order) {
        gradNorm = applyBatchUpdate([idx]);
      }
    } else if (optimizer === 'minibatch_gd') {
      const order = [...allTrainIndices];
      for (let i = order.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
      const effBatch = Math.max(1, Math.min(nTrain, batchSize));
      for (let start = 0; start < nTrain; start += effBatch) {
        gradNorm = applyBatchUpdate(order.slice(start, start + effBatch));
      }
    } else {
      applyBatchUpdate(allTrainIndices);
      gradNorm = computeFullGradNorm();
    }

    const trainPred = predict(XTrain);
    const valPred = predict(XVal);
    const trainMse = computeRegressionMetrics(yTrain, trainPred).mse;
    const valMse = computeRegressionMetrics(yVal, valPred).mse;

    if (
      !Number.isFinite(trainMse) ||
      !Number.isFinite(valMse) ||
      !Number.isFinite(gradNorm) ||
      trainMse > 1e15
    ) {
      status = 'diverged';
      statusMessage = `Training diverged at epoch ${epoch} (learning rate α = ${learningRate} is too large).`;
      break;
    }

    history.push({
      epoch,
      trainMse,
      valMse,
      weightsNorm: Math.hypot(...w),
      gradNorm,
      weightsPreview: [...w],
      bias: b,
    });

    if (epoch > 5 && gradNorm < 1e-7 && optimizer !== 'sgd' && optimizer !== 'minibatch_gd') {
      status = 'converged';
      statusMessage = `Converged at epoch ${epoch} (||∇L|| = ${gradNorm.toExponential(2)}).`;
      break;
    }
  }

  const yTrainPred = predict(XTrain);
  const yValPred = predict(XVal);
  const trainMetrics = computeRegressionMetrics(yTrain, yTrainPred);
  const valMetrics = computeRegressionMetrics(yVal, yValPred);

  // Map standardized weights back to original feature units
  const rawWeights = w.map((wj, j) => (standardize ? wj / stds[j] : wj));
  let rawBias = b;
  if (standardize) {
    for (let j = 0; j < p; j++) {
      rawBias -= (w[j] * means[j]) / stds[j];
    }
  }

  // Exact Closed-Form OLS Benchmark on XTrain (with tiny ridge 1e-10 + l2Lambda)
  let olsBenchmark: MLTrainingResult['olsBenchmark'] = null;
  const dimAug = p + 1;
  const XtX: number[][] = Array.from({ length: dimAug }, () =>
    new Array<number>(dimAug).fill(0)
  );
  const Xty = new Array<number>(dimAug).fill(0);

  for (let i = 0; i < nTrain; i++) {
    const aug = [...XTrain[i], 1];
    for (let r = 0; r < dimAug; r++) {
      Xty[r] += aug[r] * yTrain[i];
      for (let c = 0; c < dimAug; c++) {
        XtX[r][c] += aug[r] * aug[c];
      }
    }
  }
  for (let r = 0; r < p; r++) {
    XtX[r][r] += nTrain * l2Lambda + 1e-10;
  }
  XtX[p][p] += 1e-10;

  const thetaOls = solveLinearSystem(XtX, Xty);
  if (thetaOls) {
    const wOls = thetaOls.slice(0, p);
    const bOls = thetaOls[p];
    const predOls = (X: number[][]) =>
      X.map((row) => {
        let s = bOls;
        for (let j = 0; j < p; j++) s += wOls[j] * row[j];
        return s;
      });
    olsBenchmark = {
      weights: wOls,
      bias: bOls,
      trainMetrics: computeRegressionMetrics(yTrain, predOls(XTrain)),
      valMetrics: computeRegressionMetrics(yVal, predOls(XVal)),
    };
  }

  return {
    optimizer,
    epochs: history.length - 1,
    learningRate,
    batchSize,
    standardized: standardize,
    degree,
    l2Lambda,
    featureNames: expandedNames,
    targetName: targetCol,
    weights: [...w],
    bias: b,
    rawWeights,
    rawBias,
    history,
    trainMetrics,
    valMetrics,
    yTrainActual: yTrain,
    yTrainPred,
    yValActual: yVal,
    yValPred,
    xTrainPrimary: trainRows.map((r) => r[inputCols[0]]),
    xValPrimary: valRows.map((r) => r[inputCols[0]]),
    olsBenchmark,
    status,
    statusMessage,
  };
}

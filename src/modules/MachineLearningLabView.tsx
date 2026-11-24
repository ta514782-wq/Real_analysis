import { Upload } from 'lucide-react';
import { ChangeEvent, useMemo, useState } from 'react';
import { MathTex } from '../components/MathTex';
import { MetricCard } from '../components/MetricCard';
import { PlotlyChart } from '../components/PlotlyChart';
import {
  DatasetTable,
  MLOptimizerType,
  getBuiltinDatasets,
  parseCsvDataset,
  trainLinearRegressionFromScratch,
} from '../core/mlEngine';

const OPTIMIZER_OPTIONS: { id: MLOptimizerType; label: string; color: string }[] = [
  { id: 'batch_gd', label: 'Batch GD', color: '#06B6D4' },
  { id: 'sgd', label: 'Stochastic GD (SGD)', color: '#F59E0B' },
  { id: 'minibatch_gd', label: 'Mini-Batch GD', color: '#38BDF8' },
  { id: 'momentum', label: 'Momentum', color: '#A78BFA' },
  { id: 'rmsprop', label: 'RMSProp', color: '#F43F5E' },
  { id: 'adam', label: 'Adam', color: '#10B981' },
];

export function MachineLearningLabView() {
  const builtinDatasets = useMemo(() => getBuiltinDatasets(), []);
  const [datasets, setDatasets] = useState<DatasetTable[]>(builtinDatasets);
  const [selectedDatasetIdx, setSelectedDatasetIdx] = useState(0);
  const [csvError, setCsvError] = useState<string | null>(null);

  const activeDataset = datasets[selectedDatasetIdx] ?? datasets[0];

  const [inputCols, setInputCols] = useState<string[]>(() =>
    activeDataset.headers.slice(0, activeDataset.headers.length - 1)
  );
  const [targetCol, setTargetCol] = useState<string>(
    () => activeDataset.headers[activeDataset.headers.length - 1]
  );

  const [optimizer, setOptimizer] = useState<MLOptimizerType>('batch_gd');
  const [learningRate, setLearningRate] = useState(0.08);
  const [epochs, setEpochs] = useState(250);
  const [batchSize, setBatchSize] = useState(12);
  const [trainRatio, setTrainRatio] = useState(0.75);
  const [standardize, setStandardize] = useState(true);
  const [degree, setDegree] = useState(1);
  const [l2Lambda, setL2Lambda] = useState(0);
  const [compareAllOptimizers, setCompareAllOptimizers] = useState(false);

  function selectDataset(idx: number) {
    const ds = datasets[idx];
    if (!ds) return;
    setSelectedDatasetIdx(idx);
    const defaultTarget = ds.headers[ds.headers.length - 1];
    const defaultInputs = ds.headers.filter((h) => h !== defaultTarget);
    setInputCols(defaultInputs);
    setTargetCol(defaultTarget);
    setDegree(idx === 2 ? 3 : 1);
    setLearningRate(idx === 2 ? 0.15 : 0.08);
    setCsvError(null);
  }

  function handleCsvUpload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setCsvError(null);
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const text = typeof reader.result === 'string' ? reader.result : '';
        const parsed = parseCsvDataset(text, file.name);
        setDatasets((prev) => {
          const next = [...prev, parsed];
          const newIdx = next.length - 1;
          setSelectedDatasetIdx(newIdx);
          const tCol = parsed.headers[parsed.headers.length - 1];
          setTargetCol(tCol);
          setInputCols(parsed.headers.filter((h) => h !== tCol));
          setDegree(1);
          return next;
        });
      } catch (err) {
        setCsvError(err instanceof Error ? err.message : 'Failed to parse CSV file.');
      }
    };
    reader.readAsText(file);
    e.target.value = '';
  }

  function toggleInputCol(col: string) {
    if (col === targetCol) return;
    setInputCols((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  }

  const trainingState = useMemo(() => {
    try {
      const effectiveDegree = inputCols.length === 1 ? degree : 1;
      const primary = trainLinearRegressionFromScratch({
        dataset: activeDataset,
        inputCols,
        targetCol,
        optimizer,
        learningRate,
        epochs,
        batchSize,
        trainRatio,
        standardize,
        degree: effectiveDegree,
        l2Lambda,
      });

      const comparisons = compareAllOptimizers
        ? OPTIMIZER_OPTIONS.map((opt) => ({
            meta: opt,
            result: trainLinearRegressionFromScratch({
              dataset: activeDataset,
              inputCols,
              targetCol,
              optimizer: opt.id,
              learningRate,
              epochs,
              batchSize,
              trainRatio,
              standardize,
              degree: effectiveDegree,
              l2Lambda,
            }),
          }))
        : [];

      return { primary, comparisons, error: null as string | null };
    } catch (err) {
      return {
        primary: null,
        comparisons: [],
        error: err instanceof Error ? err.message : 'Training configuration error.',
      };
    }
  }, [
    activeDataset,
    inputCols,
    targetCol,
    optimizer,
    learningRate,
    epochs,
    batchSize,
    trainRatio,
    standardize,
    degree,
    l2Lambda,
    compareAllOptimizers,
  ]);

  const lossCurveSpec = useMemo((): {
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
  } => {
    if (!trainingState.primary) return { data: [], layout: {} };

    if (compareAllOptimizers && trainingState.comparisons.length > 0) {
      return {
        data: trainingState.comparisons.map((c) => ({
          x: c.result.history.map((h) => h.epoch),
          y: c.result.history.map((h) => h.trainMse),
          type: 'scatter',
          mode: 'lines',
          name: c.meta.label,
          line: { color: c.meta.color, width: 2 },
        })),
        layout: {
          title: {
            text: '6-Optimizer Training Loss Comparison (MSE)',
            font: { size: 13, color: '#E2E8F0' },
          },
          xaxis: { title: { text: 'Epoch' } },
          yaxis: { title: { text: 'Training MSE' }, type: 'log' },
        },
      };
    }

    const hist = trainingState.primary.history;
    return {
      data: [
        {
          x: hist.map((h) => h.epoch),
          y: hist.map((h) => h.trainMse),
          type: 'scatter',
          mode: 'lines',
          name: 'Training MSE',
          line: { color: '#06B6D4', width: 2.2 },
        },
        {
          x: hist.map((h) => h.epoch),
          y: hist.map((h) => h.valMse),
          type: 'scatter',
          mode: 'lines',
          name: 'Validation MSE',
          line: { color: '#F59E0B', width: 2, dash: 'dash' },
        },
      ],
      layout: {
        title: {
          text: `Train vs Validation Loss Curve (${trainingState.primary.optimizer.toUpperCase()})`,
          font: { size: 13, color: '#E2E8F0' },
        },
        xaxis: { title: { text: 'Epoch' } },
        yaxis: { title: { text: 'Mean Squared Error (MSE)' } },
      },
    };
  }, [trainingState, compareAllOptimizers]);

  const scatterSpec = useMemo((): {
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
  } => {
    const res = trainingState.primary;
    if (!res) return { data: [], layout: {} };

    // If 1 input feature, show regression fit curve on (x, y)
    if (inputCols.length === 1) {
      const allX = [...res.xTrainPrimary, ...res.xValPrimary];
      const minX = Math.min(...allX);
      const maxX = Math.max(...allX);
      const curveX: number[] = [];
      const curveY: number[] = [];
      const steps = 120;
      for (let i = 0; i <= steps; i++) {
        const xv = minX + (i / steps) * (maxX - minX);
        let yHat = res.rawBias;
        for (let d = 0; d < res.rawWeights.length; d++) {
          yHat += res.rawWeights[d] * Math.pow(xv, d + 1);
        }
        curveX.push(xv);
        curveY.push(yHat);
      }

      return {
        data: [
          {
            x: res.xTrainPrimary,
            y: res.yTrainActual,
            type: 'scatter',
            mode: 'markers',
            name: 'Train Data',
            marker: { color: '#38BDF8', size: 7, opacity: 0.85 },
          },
          {
            x: res.xValPrimary,
            y: res.yValActual,
            type: 'scatter',
            mode: 'markers',
            name: 'Validation Data',
            marker: { color: '#F59E0B', size: 8, symbol: 'diamond' },
          },
          {
            x: curveX,
            y: curveY,
            type: 'scatter',
            mode: 'lines',
            name: `GD Model (deg=${res.degree})`,
            line: { color: '#10B981', width: 2.5 },
          },
        ],
        layout: {
          title: {
            text: `${targetCol} vs ${inputCols[0]} (Train/Val & Model Fit)`,
            font: { size: 13, color: '#E2E8F0' },
          },
          xaxis: { title: { text: inputCols[0] } },
          yaxis: { title: { text: targetCol } },
        },
      };
    }

    // Multi-feature: Actual vs Predicted
    const allActual = [...res.yTrainActual, ...res.yValActual];
    const minVal = Math.min(...allActual);
    const maxVal = Math.max(...allActual);

    return {
      data: [
        {
          x: res.yTrainActual,
          y: res.yTrainPred,
          type: 'scatter',
          mode: 'markers',
          name: 'Train Split',
          marker: { color: '#38BDF8', size: 7, opacity: 0.8 },
        },
        {
          x: res.yValActual,
          y: res.yValPred,
          type: 'scatter',
          mode: 'markers',
          name: 'Validation Split',
          marker: { color: '#F59E0B', size: 8, symbol: 'diamond' },
        },
        {
          x: [minVal, maxVal],
          y: [minVal, maxVal],
          type: 'scatter',
          mode: 'lines',
          name: 'Ideal y = ŷ',
          line: { color: '#10B981', width: 1.5, dash: 'dot' },
        },
      ],
      layout: {
        title: {
          text: 'Actual vs Predicted Values',
          font: { size: 13, color: '#E2E8F0' },
        },
        xaxis: { title: { text: `Actual ${targetCol}` } },
        yaxis: { title: { text: `Predicted ${targetCol}` } },
      },
    };
  }, [trainingState.primary, inputCols, targetCol]);

  const modelEquationTex = useMemo(() => {
    const res = trainingState.primary;
    if (!res) return '';
    const terms = res.rawWeights.map((wj, idx) => {
      const name = res.featureNames[idx].replace(/_/g, '\\_');
      const sign = wj >= 0 ? (idx === 0 ? '' : ' + ') : ' - ';
      return `${sign}${Math.abs(wj).toFixed(3)}\\cdot\\text{${name}}`;
    });
    const bSign = res.rawBias >= 0 ? ' + ' : ' - ';
    return `\\hat{y} = ${terms.join('')}${bSign}${Math.abs(res.rawBias).toFixed(3)}`;
  }, [trainingState.primary]);

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      {/* Left Configuration */}
      <div className="space-y-5 xl:col-span-4">
        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5">
          <h2 className="font-display text-base font-semibold text-slate-100">
            Module 4 · Machine Learning Laboratory
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Train Linear and Polynomial Regression from scratch using Gradient Descent variants and benchmark against exact OLS.
          </p>

          <div className="mt-4 space-y-3">
            <div>
              <label htmlFor="ml-dataset" className="block text-xs font-medium text-slate-300">
                Dataset Selection
              </label>
              <select
                id="ml-dataset"
                value={selectedDatasetIdx}
                onChange={(e) => selectDataset(Number(e.target.value))}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 text-xs text-slate-100 focus:border-cyan-500 focus:outline-none"
              >
                {datasets.map((ds, idx) => (
                  <option key={ds.name + idx} value={idx}>
                    {ds.name} ({ds.rows.length} rows)
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="flex cursor-pointer items-center justify-center gap-2 rounded border border-dashed border-slate-700 bg-[#0B0F17] px-3 py-2 text-xs text-slate-300 hover:border-cyan-500/60 hover:text-cyan-300">
                <Upload className="h-3.5 w-3.5" />
                <span>Upload Custom CSV Dataset</span>
                <input
                  type="file"
                  accept=".csv,text/csv"
                  onChange={handleCsvUpload}
                  className="hidden"
                />
              </label>
              {csvError && <p className="mt-1 text-xs text-rose-400">{csvError}</p>}
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="ml-target" className="block text-xs font-medium text-slate-300">
                  Target Column (y)
                </label>
                <select
                  id="ml-target"
                  value={targetCol}
                  onChange={(e) => {
                    const nextT = e.target.value;
                    setTargetCol(nextT);
                    setInputCols((prev) => {
                      const filtered = prev.filter((c) => c !== nextT);
                      return filtered.length > 0
                        ? filtered
                        : activeDataset.headers.filter((h) => h !== nextT).slice(0, 1);
                    });
                  }}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-2.5 py-1.5 font-mono text-xs text-slate-100"
                >
                  {activeDataset.headers.map((h) => (
                    <option key={h} value={h}>
                      {h}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label htmlFor="ml-opt" className="block text-xs font-medium text-slate-300">
                  Optimizer
                </label>
                <select
                  id="ml-opt"
                  value={optimizer}
                  onChange={(e) => setOptimizer(e.target.value as MLOptimizerType)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-2.5 py-1.5 font-mono text-xs text-slate-100"
                >
                  {OPTIMIZER_OPTIONS.map((opt) => (
                    <option key={opt.id} value={opt.id}>
                      {opt.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div>
              <div className="text-xs font-medium text-slate-300">
                Input Features (X)
              </div>
              <div className="mt-1.5 flex flex-wrap gap-1.5">
                {activeDataset.headers
                  .filter((h) => h !== targetCol)
                  .map((h) => {
                    const active = inputCols.includes(h);
                    return (
                      <button
                        key={h}
                        type="button"
                        onClick={() => toggleInputCol(h)}
                        className={`rounded border px-2.5 py-1 font-mono text-xs transition-colors ${
                          active
                            ? 'border-cyan-500/60 bg-cyan-500/15 text-cyan-300'
                            : 'border-slate-800 bg-[#0B0F17] text-slate-400 hover:text-slate-200'
                        }`}
                      >
                        {h}
                      </button>
                    );
                  })}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="ml-lr" className="block text-xs font-medium text-slate-300">
                  Learning Rate α
                </label>
                <input
                  id="ml-lr"
                  type="number"
                  step="0.01"
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-2.5 py-1.5 font-mono text-xs text-slate-100"
                />
              </div>
              <div>
                <label htmlFor="ml-epochs" className="block text-xs font-medium text-slate-300">
                  Epochs
                </label>
                <input
                  id="ml-epochs"
                  type="number"
                  min={10}
                  max={2000}
                  step={20}
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-2.5 py-1.5 font-mono text-xs text-slate-100"
                />
              </div>

              {optimizer === 'minibatch_gd' && (
                <div>
                  <label htmlFor="ml-bs" className="block text-xs font-medium text-slate-300">
                    Batch Size
                  </label>
                  <input
                    id="ml-bs"
                    type="number"
                    min={2}
                    max={64}
                    value={batchSize}
                    onChange={(e) => setBatchSize(Number(e.target.value))}
                    className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-2.5 py-1.5 font-mono text-xs text-slate-100"
                  />
                </div>
              )}

              <div>
                <label htmlFor="ml-split" className="block text-xs font-medium text-slate-300">
                  Train Split ({Math.round(trainRatio * 100)}%)
                </label>
                <input
                  id="ml-split"
                  type="range"
                  min={0.5}
                  max={0.9}
                  step={0.05}
                  value={trainRatio}
                  onChange={(e) => setTrainRatio(Number(e.target.value))}
                  className="mt-2 w-full accent-cyan-500"
                />
              </div>

              {inputCols.length === 1 && (
                <div>
                  <label htmlFor="ml-deg" className="block text-xs font-medium text-slate-300">
                    Poly Degree (d = {degree})
                  </label>
                  <input
                    id="ml-deg"
                    type="range"
                    min={1}
                    max={9}
                    step={1}
                    value={degree}
                    onChange={(e) => setDegree(Number(e.target.value))}
                    className="mt-2 w-full accent-amber-500"
                  />
                </div>
              )}

              <div>
                <label htmlFor="ml-l2" className="block text-xs font-medium text-slate-300">
                  L2 Regularization λ
                </label>
                <input
                  id="ml-l2"
                  type="number"
                  min={0}
                  step="0.01"
                  value={l2Lambda}
                  onChange={(e) => setL2Lambda(Number(e.target.value))}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-2.5 py-1.5 font-mono text-xs text-slate-100"
                />
              </div>
            </div>

            <div className="space-y-2 pt-1 text-xs text-slate-300">
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={standardize}
                  onChange={(e) => setStandardize(e.target.checked)}
                  className="accent-cyan-500"
                />
                Standardize features (Z-score on training split)
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={compareAllOptimizers}
                  onChange={(e) => setCompareAllOptimizers(e.target.checked)}
                  className="accent-emerald-500"
                />
                Compare all 6 optimizers (Batch, SGD, Mini-Batch, Momentum, RMSProp, Adam)
              </label>
            </div>
          </div>
        </div>

        {/* Model Equation */}
        {trainingState.primary && (
          <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 space-y-3">
            <h3 className="font-display text-sm font-semibold text-slate-100">
              Learned Model Equation
            </h3>
            <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
              <MathTex block tex={modelEquationTex} />
            </div>
            <div className="text-xs text-slate-400">
              {trainingState.primary.statusMessage}
              {degree >= 6 &&
                trainingState.primary.valMetrics.mse >
                  trainingState.primary.trainMetrics.mse * 1.4 && (
                  <span className="block mt-1.5 text-amber-300">
                    ▲ Overfitting indicator: high polynomial degree (d={degree}) lowers training MSE while validation MSE increases. Try d=3 or L2 regularization λ &gt; 0.
                  </span>
                )}
            </div>
          </div>
        )}
      </div>

      {/* Right Stage */}
      <div className="space-y-5 xl:col-span-8">
        {trainingState.error && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 p-4 text-sm text-rose-200">
            <span className="font-mono font-semibold">✖ Training Error: </span>
            {trainingState.error}
          </div>
        )}

        {trainingState.primary && (
          <>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <MetricCard
                label="Validation R²"
                value={
                  Number.isFinite(trainingState.primary.valMetrics.r2)
                    ? trainingState.primary.valMetrics.r2.toFixed(4)
                    : 'NaN'
                }
                detail={`Train R² = ${trainingState.primary.trainMetrics.r2.toFixed(4)}`}
                tone={trainingState.primary.valMetrics.r2 > 0.7 ? 'emerald' : 'amber'}
              />
              <MetricCard
                label="Validation MSE"
                value={
                  Number.isFinite(trainingState.primary.valMetrics.mse)
                    ? trainingState.primary.valMetrics.mse.toFixed(4)
                    : 'NaN'
                }
                detail={`Train MSE = ${trainingState.primary.trainMetrics.mse.toFixed(4)}`}
                tone="cyan"
              />
              <MetricCard
                label="Validation MAE"
                value={
                  Number.isFinite(trainingState.primary.valMetrics.mae)
                    ? trainingState.primary.valMetrics.mae.toFixed(4)
                    : 'NaN'
                }
                detail={`Train MAE = ${trainingState.primary.trainMetrics.mae.toFixed(4)}`}
              />
              <MetricCard
                label="OLS Analytical R²"
                value={
                  trainingState.primary.olsBenchmark
                    ? trainingState.primary.olsBenchmark.valMetrics.r2.toFixed(4)
                    : '—'
                }
                detail={
                  trainingState.primary.olsBenchmark
                    ? `OLS Val MSE = ${trainingState.primary.olsBenchmark.valMetrics.mse.toFixed(4)}`
                    : 'Collinear matrix'
                }
              />
            </div>

            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              <PlotlyChart
                data={lossCurveSpec.data}
                layout={lossCurveSpec.layout}
                height={380}
              />
              <PlotlyChart
                data={scatterSpec.data}
                layout={scatterSpec.layout}
                height={380}
              />
            </div>

            {/* 6-Optimizer Comparison Table */}
            {compareAllOptimizers && trainingState.comparisons.length > 0 && (
              <div className="rounded-lg border border-slate-800 bg-[#0F141D]">
                <div className="border-b border-slate-800 px-4 py-3">
                  <h3 className="font-display text-sm font-semibold text-slate-100">
                    Optimizer Algorithm Benchmark (α = {learningRate}, {epochs} epochs)
                  </h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left font-mono text-xs tabular-nums">
                    <thead className="bg-[#0B0F17] text-slate-400">
                      <tr>
                        <th className="px-4 py-2.5 font-medium">Optimizer</th>
                        <th className="px-4 py-2.5 font-medium">Train MSE</th>
                        <th className="px-4 py-2.5 font-medium">Val MSE</th>
                        <th className="px-4 py-2.5 font-medium">Val MAE</th>
                        <th className="px-4 py-2.5 font-medium">Val R²</th>
                        <th className="px-4 py-2.5 font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/80 text-slate-200">
                      {trainingState.comparisons.map((c) => (
                        <tr key={c.meta.id} className="hover:bg-slate-800/30">
                          <td className="px-4 py-2.5 font-sans font-medium text-slate-100">
                            <span
                              className="mr-2 inline-block h-2 w-2 rounded-full"
                              style={{ backgroundColor: c.meta.color }}
                            />
                            {c.meta.label}
                          </td>
                          <td className="px-4 py-2.5">
                            {Number.isFinite(c.result.trainMetrics.mse)
                              ? c.result.trainMetrics.mse.toFixed(4)
                              : 'Diverged'}
                          </td>
                          <td className="px-4 py-2.5">
                            {Number.isFinite(c.result.valMetrics.mse)
                              ? c.result.valMetrics.mse.toFixed(4)
                              : 'Diverged'}
                          </td>
                          <td className="px-4 py-2.5">
                            {Number.isFinite(c.result.valMetrics.mae)
                              ? c.result.valMetrics.mae.toFixed(4)
                              : '—'}
                          </td>
                          <td className="px-4 py-2.5 text-cyan-300">
                            {Number.isFinite(c.result.valMetrics.r2)
                              ? c.result.valMetrics.r2.toFixed(4)
                              : '—'}
                          </td>
                          <td className="px-4 py-2.5 font-sans">
                            <span
                              className={
                                c.result.status === 'diverged'
                                  ? 'text-rose-400'
                                  : 'text-emerald-400'
                              }
                            >
                              {c.result.status === 'diverged' ? '✖ Diverged' : '● Complete'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Parameter Updates Table */}
            <div className="rounded-lg border border-slate-800 bg-[#0F141D]">
              <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
                <h3 className="font-display text-sm font-semibold text-slate-100">
                  Training Parameter Updates Log
                </h3>
                <span className="font-mono text-xs text-slate-400">
                  Sampled epochs
                </span>
              </div>
              <div className="max-h-64 overflow-auto">
                <table className="w-full text-left font-mono text-xs tabular-nums">
                  <thead className="sticky top-0 bg-[#0B0F17] text-slate-400">
                    <tr>
                      <th className="px-4 py-2 font-medium">Epoch</th>
                      <th className="px-4 py-2 font-medium">Train MSE</th>
                      <th className="px-4 py-2 font-medium">Val MSE</th>
                      <th className="px-4 py-2 font-medium">||w||</th>
                      <th className="px-4 py-2 font-medium">Bias b</th>
                      <th className="px-4 py-2 font-medium">||∇L||</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/80 text-slate-200">
                    {trainingState.primary.history
                      .filter(
                        (_, i, arr) =>
                          i < 10 || i === arr.length - 1 || i % Math.max(1, Math.floor(arr.length / 25)) === 0
                      )
                      .map((row) => (
                        <tr key={row.epoch} className="hover:bg-slate-800/30">
                          <td className="px-4 py-2 text-slate-400">{row.epoch}</td>
                          <td className="px-4 py-2">{row.trainMse.toFixed(4)}</td>
                          <td className="px-4 py-2">{row.valMse.toFixed(4)}</td>
                          <td className="px-4 py-2">{row.weightsNorm.toFixed(4)}</td>
                          <td className="px-4 py-2">{row.bias.toFixed(4)}</td>
                          <td className="px-4 py-2 text-cyan-300">
                            {row.gradNorm.toExponential(2)}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

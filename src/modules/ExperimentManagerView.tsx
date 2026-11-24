import { Download, Plus, Trash2 } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { MetricCard } from '../components/MetricCard';
import { PlotlyChart } from '../components/PlotlyChart';
import { analyzeExpression } from '../core/mathematics';
import {
  OptimizerAlgorithm,
  runGradientDescent1D,
  runGradientDescent2D,
} from '../core/optimizers';

export interface SavedExperiment {
  id: string;
  name: string;
  createdAt: string;
  expression: string;
  dimension: 1 | 2;
  algorithm: OptimizerAlgorithm;
  alpha: number;
  x0: number;
  y0: number;
  maxIterations: number;
  tolerance: number;
  totalIterations: number;
  finalX: number;
  finalY?: number;
  finalLoss: number;
  finalGradNorm: number;
  status: string;
  lossCurve: number[];
  selectedForPlot: boolean;
}

const STORAGE_KEY = 'gradai_experiments_v1';
const PLOT_COLORS = ['#06B6D4', '#10B981', '#F59E0B', '#A78BFA', '#F43F5E', '#38BDF8'];

function loadStoredExperiments(): SavedExperiment[] {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is SavedExperiment =>
        Boolean(item) &&
        typeof item === 'object' &&
        typeof item.id === 'string' &&
        typeof item.name === 'string' &&
        typeof item.expression === 'string' &&
        typeof item.alpha === 'number' &&
        typeof item.finalLoss === 'number' &&
        Array.isArray(item.lossCurve)
    );
  } catch {
    return [];
  }
}

function sanitizeCsvTextCell(value: string): string {
  const safe = /^[=+\-@\t\r]/.test(value) ? `'${value}` : value;
  return `"${safe.replace(/"/g, '""')}"`;
}

export function ExperimentManagerView() {
  const [experiments, setExperiments] = useState<SavedExperiment[]>(() =>
    loadStoredExperiments()
  );
  const [nameInput, setNameInput] = useState('Run #1 · Quadratic GD');
  const [exprInput, setExprInput] = useState('x**2 + 2*y**2');
  const [x0Input, setX0Input] = useState('4');
  const [y0Input, setY0Input] = useState('-3');
  const [alphaInput, setAlphaInput] = useState('0.1');
  const [maxIterInput, setMaxIterInput] = useState('150');
  const [tolInput, setTolInput] = useState('1e-6');
  const [algorithm, setAlgorithm] = useState<OptimizerAlgorithm>('gd');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(experiments));
    } catch {
      // Ignore storage quota errors
    }
  }, [experiments]);

  function handleRunAndSave() {
    setErrorMsg(null);
    const x0 = Number(x0Input);
    const y0 = Number(y0Input);
    const alpha = Number(alphaInput);
    const maxIter = Number(maxIterInput);
    const tol = Number(tolInput);

    try {
      const analysis = analyzeExpression(exprInput);
      if (analysis.dimension === 1) {
        const res = runGradientDescent1D(exprInput, x0, alpha, maxIter, tol);
        const item: SavedExperiment = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          name: nameInput.trim() || `1D GD (${exprInput})`,
          createdAt: new Date().toISOString().slice(11, 19),
          expression: res.expression,
          dimension: 1,
          algorithm: 'gd',
          alpha,
          x0,
          y0: 0,
          maxIterations: maxIter,
          tolerance: tol,
          totalIterations: res.totalIterations,
          finalX: res.finalX,
          finalLoss: res.finalFx,
          finalGradNorm: Math.abs(res.finalGrad),
          status: res.statusLabel,
          lossCurve: res.history.map((h) => h.fx),
          selectedForPlot: true,
        };
        setExperiments((prev) => [item, ...prev]);
      } else {
        const res = runGradientDescent2D(
          exprInput,
          x0,
          y0,
          alpha,
          maxIter,
          tol,
          algorithm
        );
        const item: SavedExperiment = {
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
          name: nameInput.trim() || `${algorithm.toUpperCase()} (${exprInput})`,
          createdAt: new Date().toISOString().slice(11, 19),
          expression: res.expression,
          dimension: 2,
          algorithm,
          alpha,
          x0,
          y0,
          maxIterations: maxIter,
          tolerance: tol,
          totalIterations: res.totalIterations,
          finalX: res.finalX,
          finalY: res.finalY,
          finalLoss: res.finalFxy,
          finalGradNorm: res.finalGradNorm,
          status: res.statusLabel,
          lossCurve: res.history.map((h) => h.fxy),
          selectedForPlot: true,
        };
        setExperiments((prev) => [item, ...prev]);
      }
      setNameInput(`Run #${experiments.length + 2}`);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : 'Failed to run experiment.');
    }
  }

  function togglePlotSelection(id: string) {
    setExperiments((prev) =>
      prev.map((e) => (e.id === id ? { ...e, selectedForPlot: !e.selectedForPlot } : e))
    );
  }

  function deleteExperiment(id: string) {
    setExperiments((prev) => prev.filter((e) => e.id !== id));
  }

  function exportCsv() {
    if (experiments.length === 0) return;
    const headers = [
      'Name',
      'Time',
      'Expression',
      'Dimension',
      'Optimizer',
      'Alpha',
      'x0',
      'y0',
      'Iterations',
      'FinalX',
      'FinalY',
      'FinalLoss',
      'FinalGradNorm',
      'Status',
    ];
    const rows = experiments.map((e) =>
      [
        sanitizeCsvTextCell(e.name),
        e.createdAt,
        sanitizeCsvTextCell(e.expression),
        e.dimension,
        e.algorithm.toUpperCase(),
        e.alpha,
        e.x0,
        e.y0,
        e.totalIterations,
        e.finalX,
        e.finalY ?? '',
        e.finalLoss,
        e.finalGradNorm,
        sanitizeCsvTextCell(e.status),
      ].join(',')
    );
    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'gradai_experiments.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  const comparisonPlotSpec = useMemo((): {
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
  } => {
    const selected = experiments.filter((e) => e.selectedForPlot);
    return {
      data: selected.map((exp, idx) => ({
        x: exp.lossCurve.map((_, i) => i),
        y: exp.lossCurve.map((v) => Math.max(Math.abs(v), 1e-16)),
        type: 'scatter',
        mode: 'lines',
        name: `${exp.name} (α=${exp.alpha})`,
        line: { color: PLOT_COLORS[idx % PLOT_COLORS.length], width: 2 },
      })),
      layout: {
        title: {
          text: 'Saved Experiments Convergence Overlay (Log Objective Magnitude)',
          font: { size: 13, color: '#E2E8F0' },
        },
        xaxis: { title: { text: 'Iteration n' } },
        yaxis: { title: { text: '|f(X_n)|' }, type: 'log' },
      },
    };
  }, [experiments]);

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      {/* Left Config & Save */}
      <div className="space-y-5 xl:col-span-4">
        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5">
          <h2 className="font-display text-base font-semibold text-slate-100">
            Experiment Manager
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Execute and persist optimization configurations locally to compare convergence curves and export CSV summaries.
          </p>

          <div className="mt-4 space-y-3">
            <div>
              <label htmlFor="em-name" className="block text-xs font-medium text-slate-300">
                Experiment Label
              </label>
              <input
                id="em-name"
                type="text"
                value={nameInput}
                onChange={(e) => setNameInput(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <label htmlFor="em-expr" className="block text-xs font-medium text-slate-300">
                Expression f(x) or f(x, y)
              </label>
              <input
                id="em-expr"
                type="text"
                value={exprInput}
                onChange={(e) => setExprInput(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="em-x0" className="block text-xs font-medium text-slate-300">
                  x₀
                </label>
                <input
                  id="em-x0"
                  type="number"
                  step="any"
                  value={x0Input}
                  onChange={(e) => setX0Input(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100"
                />
              </div>
              <div>
                <label htmlFor="em-y0" className="block text-xs font-medium text-slate-300">
                  y₀ (if 2D)
                </label>
                <input
                  id="em-y0"
                  type="number"
                  step="any"
                  value={y0Input}
                  onChange={(e) => setY0Input(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100"
                />
              </div>
              <div>
                <label htmlFor="em-alpha" className="block text-xs font-medium text-slate-300">
                  Learning Rate α
                </label>
                <input
                  id="em-alpha"
                  type="number"
                  step="0.01"
                  value={alphaInput}
                  onChange={(e) => setAlphaInput(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100"
                />
              </div>
              <div>
                <label htmlFor="em-algo" className="block text-xs font-medium text-slate-300">
                  Optimizer (2D)
                </label>
                <select
                  id="em-algo"
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value as OptimizerAlgorithm)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-2.5 py-2 font-mono text-xs text-slate-100"
                >
                  <option value="gd">Batch GD</option>
                  <option value="momentum">Momentum</option>
                  <option value="rmsprop">RMSProp</option>
                  <option value="adam">Adam</option>
                </select>
              </div>
              <div>
                <label htmlFor="em-maxiter" className="block text-xs font-medium text-slate-300">
                  Max Iterations
                </label>
                <input
                  id="em-maxiter"
                  type="number"
                  value={maxIterInput}
                  onChange={(e) => setMaxIterInput(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100"
                />
              </div>
              <div>
                <label htmlFor="em-tol" className="block text-xs font-medium text-slate-300">
                  Tolerance ε
                </label>
                <input
                  id="em-tol"
                  type="text"
                  value={tolInput}
                  onChange={(e) => setTolInput(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100"
                />
              </div>
            </div>

            {errorMsg && (
              <div className="rounded border border-rose-500/40 bg-rose-950/30 p-3 text-xs text-rose-200">
                ✖ {errorMsg}
              </div>
            )}

            <button
              type="button"
              onClick={handleRunAndSave}
              className="flex w-full items-center justify-center gap-2 rounded-md bg-cyan-500 px-4 py-2.5 text-xs font-semibold text-slate-950 transition-colors hover:bg-cyan-400"
            >
              <Plus className="h-4 w-4" />
              Run & Save Experiment
            </button>
          </div>
        </div>
      </div>

      {/* Right Stage */}
      <div className="space-y-5 xl:col-span-8">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
          <MetricCard
            label="Saved Experiments"
            value={String(experiments.length)}
            detail="Persisted in browser local storage"
            tone="cyan"
          />
          <MetricCard
            label="Plotted Overlays"
            value={String(experiments.filter((e) => e.selectedForPlot).length)}
            detail="Toggle checkboxes below"
          />
          <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-[#0F141D] p-4">
            <div>
              <div className="text-xs font-medium text-slate-400">Export & Cleanup</div>
              <div className="mt-1 text-xs text-slate-500">Download CSV log</div>
            </div>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={exportCsv}
                disabled={experiments.length === 0}
                className="flex items-center gap-1.5 rounded border border-slate-700 bg-[#0B0F17] px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-cyan-500/50 disabled:opacity-40"
              >
                <Download className="h-3.5 w-3.5" />
                CSV
              </button>
              <button
                type="button"
                onClick={() => setExperiments([])}
                disabled={experiments.length === 0}
                className="rounded border border-slate-700 bg-[#0B0F17] px-2.5 py-1.5 text-xs text-rose-300 hover:border-rose-500/50 disabled:opacity-40"
              >
                Clear
              </button>
            </div>
          </div>
        </div>

        <PlotlyChart
          data={comparisonPlotSpec.data}
          layout={comparisonPlotSpec.layout}
          height={340}
        />

        <div className="rounded-lg border border-slate-800 bg-[#0F141D]">
          <div className="border-b border-slate-800 px-4 py-3">
            <h3 className="font-display text-sm font-semibold text-slate-100">
              Experiment History
            </h3>
          </div>
          {experiments.length === 0 ? (
            <div className="px-4 py-6 text-xs text-slate-400">
              No saved experiments yet. Configure parameters on the left and click &ldquo;Run &amp; Save Experiment&rdquo; to log and compare runs.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs tabular-nums">
                <thead className="bg-[#0B0F17] text-slate-400">
                  <tr>
                    <th className="px-3 py-2.5 font-medium">Plot</th>
                    <th className="px-3 py-2.5 font-medium">Name</th>
                    <th className="px-3 py-2.5 font-medium">f</th>
                    <th className="px-3 py-2.5 font-medium">Opt / α</th>
                    <th className="px-3 py-2.5 font-medium">Iters</th>
                    <th className="px-3 py-2.5 font-medium">Final Loss</th>
                    <th className="px-3 py-2.5 font-medium">Status</th>
                    <th className="px-3 py-2.5 font-medium">Del</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80 text-slate-200">
                  {experiments.map((exp) => (
                    <tr key={exp.id} className="hover:bg-slate-800/30">
                      <td className="px-3 py-2">
                        <input
                          type="checkbox"
                          checked={exp.selectedForPlot}
                          onChange={() => togglePlotSelection(exp.id)}
                          className="accent-cyan-500"
                        />
                      </td>
                      <td className="px-3 py-2 font-sans font-medium text-slate-100">
                        {exp.name}
                      </td>
                      <td className="px-3 py-2 text-cyan-300">{exp.expression}</td>
                      <td className="px-3 py-2">
                        {exp.algorithm.toUpperCase()} · {exp.alpha}
                      </td>
                      <td className="px-3 py-2">{exp.totalIterations}</td>
                      <td className="px-3 py-2">{exp.finalLoss.toExponential(3)}</td>
                      <td className="px-3 py-2 font-sans text-slate-300">{exp.status}</td>
                      <td className="px-3 py-2">
                        <button
                          type="button"
                          onClick={() => deleteExperiment(exp.id)}
                          aria-label={`Delete ${exp.name}`}
                          className="text-slate-500 hover:text-rose-400"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

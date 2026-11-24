import { useMemo, useState } from 'react';
import { MathTex } from '../components/MathTex';
import { MetricCard } from '../components/MetricCard';
import { PlotlyChart } from '../components/PlotlyChart';
import { analyzeExpression, findCriticalPoints2D } from '../core/mathematics';
import {
  GradientDescentResult2D,
  OptimizerAlgorithm,
  runGradientDescent2D,
} from '../core/optimizers';

const MULTIVAR_PRESETS = [
  {
    label: 'x² + y²',
    expr: 'x**2 + y**2',
    x0: 3.5,
    y0: -2.8,
    alpha: 0.12,
    note: 'Isotropic convex paraboloid; global minimum at (0, 0).',
  },
  {
    label: 'x² + 2y²',
    expr: 'x**2 + 2*y**2',
    x0: 3.5,
    y0: -2.8,
    alpha: 0.15,
    note: 'Elliptic bowl; steeper along y-axis than x-axis.',
  },
  {
    label: 'x² - y²',
    expr: 'x**2 - y**2',
    x0: 2.5,
    y0: 0.35,
    alpha: 0.08,
    note: 'Hyperbolic saddle at (0, 0); descends along y while stabilizing along x.',
  },
  {
    label: 'x² + xy + y²',
    expr: 'x**2 + x*y + y**2',
    x0: 3.5,
    y0: -2.0,
    alpha: 0.15,
    note: 'Coupled positive-definite quadratic form with rotated principal axes.',
  },
  {
    label: 'x² + 8y²',
    expr: 'x**2 + 8*y**2',
    x0: 3.8,
    y0: 1.5,
    alpha: 0.1,
    note: 'Ill-conditioned valley demonstrating GD zig-zagging vs Momentum/Adam.',
  },
];

const ALGO_COLORS: Record<OptimizerAlgorithm, string> = {
  gd: '#06B6D4',
  momentum: '#F59E0B',
  rmsprop: '#A78BFA',
  adam: '#10B981',
};

const ALGO_LABELS: Record<OptimizerAlgorithm, string> = {
  gd: 'Batch GD',
  momentum: 'Momentum',
  rmsprop: 'RMSProp',
  adam: 'Adam',
};

export function Multivariable3DLabView() {
  const [exprInput, setExprInput] = useState('x**2 + 2*y**2');
  const [x0Input, setX0Input] = useState('3.5');
  const [y0Input, setY0Input] = useState('-2.8');
  const [compareSecondStart, setCompareSecondStart] = useState(false);
  const [x0AltInput, setX0AltInput] = useState('-3.2');
  const [y0AltInput, setY0AltInput] = useState('2.6');
  const [alphaInput, setAlphaInput] = useState('0.12');
  const [maxIterInput, setMaxIterInput] = useState('120');
  const [selectedAlgos, setSelectedAlgos] = useState<OptimizerAlgorithm[]>(['gd', 'momentum', 'adam']);
  const [showVectorField, setShowVectorField] = useState(true);
  const [domainBound, setDomainBound] = useState(4.5);

  const analysisState = useMemo(() => {
    try {
      const data = analyzeExpression(exprInput, { allowedDimensions: [2] });
      return { data, error: null as string | null };
    } catch (err) {
      return {
        data: null,
        error: err instanceof Error ? err.message : 'Enter a valid multivariable function f(x, y).',
      };
    }
  }, [exprInput]);

  const criticalPoints = useMemo(() => {
    if (!analysisState.data) return [];
    return findCriticalPoints2D(analysisState.data, domainBound);
  }, [analysisState.data, domainBound]);

  const runs = useMemo((): {
    items: { label: string; color: string; result: GradientDescentResult2D }[];
    error: string | null;
  } => {
    if (!analysisState.data) return { items: [], error: analysisState.error };

    const x0 = Number(x0Input);
    const y0 = Number(y0Input);
    const alpha = Number(alphaInput);
    const maxIter = Number(maxIterInput);

    if (selectedAlgos.length === 0) {
      return { items: [], error: 'Select at least one optimizer to trace.' };
    }

    try {
      const items: { label: string; color: string; result: GradientDescentResult2D }[] = [];
      for (const algo of selectedAlgos) {
        const res = runGradientDescent2D(exprInput, x0, y0, alpha, maxIter, 1e-6, algo);
        items.push({
          label: `${ALGO_LABELS[algo]} (${x0}, ${y0})`,
          color: ALGO_COLORS[algo],
          result: res,
        });
      }

      if (compareSecondStart) {
        const x0Alt = Number(x0AltInput);
        const y0Alt = Number(y0AltInput);
        const primaryAlgo = selectedAlgos[0];
        const resAlt = runGradientDescent2D(
          exprInput,
          x0Alt,
          y0Alt,
          alpha,
          maxIter,
          1e-6,
          primaryAlgo
        );
        items.push({
          label: `${ALGO_LABELS[primaryAlgo]} Alt (${x0Alt}, ${y0Alt})`,
          color: '#F43F5E',
          result: resAlt,
        });
      }

      return { items, error: null };
    } catch (err) {
      return {
        items: [],
        error: err instanceof Error ? err.message : 'Simulation parameter error.',
      };
    }
  }, [
    analysisState,
    exprInput,
    x0Input,
    y0Input,
    compareSecondStart,
    x0AltInput,
    y0AltInput,
    alphaInput,
    maxIterInput,
    selectedAlgos,
  ]);

  const gridData = useMemo(() => {
    const a = analysisState.data;
    if (!a) return null;
    const gridN = 45;
    const gx: number[] = [];
    const gy: number[] = [];
    const gz: number[][] = [];
    for (let i = 0; i < gridN; i++) {
      const v = -domainBound + (i / (gridN - 1)) * (2 * domainBound);
      gx.push(v);
      gy.push(v);
    }
    for (let j = 0; j < gridN; j++) {
      const row: number[] = [];
      for (let i = 0; i < gridN; i++) {
        const val = a.evalF(gx[i], gy[j]);
        row.push(Number.isFinite(val) && Math.abs(val) < 1e4 ? val : NaN);
      }
      gz.push(row);
    }
    return { gx, gy, gz };
  }, [analysisState.data, domainBound]);

  const surface3DSpec = useMemo((): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } => {
    if (!gridData || !analysisState.data) return { data: [], layout: {} };

    const traces: Plotly.Data[] = [
      {
        x: gridData.gx,
        y: gridData.gy,
        z: gridData.gz,
        type: 'surface',
        colorscale: 'Cividis',
        opacity: 0.85,
        showscale: false,
        name: 'f(x, y)',
      } as Plotly.Data,
    ];

    for (const run of runs.items) {
      const hist = run.result.history.filter(
        (h) => Math.abs(h.x) <= domainBound * 1.5 && Math.abs(h.y) <= domainBound * 1.5
      );
      traces.push({
        x: hist.map((h) => h.x),
        y: hist.map((h) => h.y),
        z: hist.map((h) => h.fxy + 0.05),
        type: 'scatter3d',
        mode: 'lines+markers',
        name: run.label,
        line: { color: run.color, width: 4 },
        marker: { size: 3.5, color: run.color },
      } as Plotly.Data);
    }

    if (criticalPoints.length > 0) {
      traces.push({
        x: criticalPoints.map((c) => c.x),
        y: criticalPoints.map((c) => c.y),
        z: criticalPoints.map((c) => c.fxy + 0.1),
        text: criticalPoints.map((c) => c.hessian.classification),
        type: 'scatter3d',
        mode: 'markers',
        name: 'Critical Points',
        marker: { size: 6, symbol: 'diamond', color: '#FFFFFF' },
      } as Plotly.Data);
    }

    return {
      data: traces,
      layout: {
        title: {
          text: 'Interactive 3D Surface & Optimization Trajectories',
          font: { size: 13, color: '#E2E8F0' },
        },
        margin: { t: 40, r: 10, b: 10, l: 10 },
        scene: {
          xaxis: { title: { text: 'x' }, gridcolor: '#1E293B', color: '#94A3B8' },
          yaxis: { title: { text: 'y' }, gridcolor: '#1E293B', color: '#94A3B8' },
          zaxis: { title: { text: 'f(x, y)' }, gridcolor: '#1E293B', color: '#94A3B8' },
          bgcolor: '#0B0F17',
        },
      },
    };
  }, [gridData, analysisState.data, runs.items, criticalPoints, domainBound]);

  const contour2DSpec = useMemo((): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } => {
    if (!gridData || !analysisState.data) return { data: [], layout: {} };

    const traces: Plotly.Data[] = [
      {
        x: gridData.gx,
        y: gridData.gy,
        z: gridData.gz,
        type: 'contour',
        colorscale: 'Cividis',
        contours: { coloring: 'heatmap' },
        name: 'f(x, y)',
      },
    ];

    for (const run of runs.items) {
      const hist = run.result.history;
      traces.push({
        x: hist.map((h) => h.x),
        y: hist.map((h) => h.y),
        type: 'scatter',
        mode: 'lines+markers',
        name: run.label,
        line: { color: run.color, width: 2.2 },
        marker: { size: 5, color: run.color },
      });
    }

    if (criticalPoints.length > 0) {
      traces.push({
        x: criticalPoints.map((c) => c.x),
        y: criticalPoints.map((c) => c.y),
        text: criticalPoints.map(
          (c) => `${c.hessian.classification} (${c.x.toFixed(2)}, ${c.y.toFixed(2)})`
        ),
        hoverinfo: 'text',
        type: 'scatter',
        mode: 'markers',
        name: 'Critical Points',
        marker: {
          size: 10,
          symbol: 'diamond',
          color: '#FFFFFF',
          line: { color: '#07090E', width: 1.5 },
        },
      });
    }

    const annotations: Partial<Plotly.Annotation>[] = [];
    if (showVectorField) {
      const vecGrid = 9;
      const step = (2 * domainBound) / (vecGrid + 1);
      const maxArrowLen = step * 0.65;
      for (let i = 1; i <= vecGrid; i++) {
        for (let j = 1; j <= vecGrid; j++) {
          const px = -domainBound + i * step;
          const py = -domainBound + j * step;
          const [gx, gy] = analysisState.data.evalGrad(px, py);
          const norm = Math.hypot(gx, gy);
          if (!Number.isFinite(norm) || norm < 1e-6) continue;
          const len = Math.min(maxArrowLen, maxArrowLen * Math.tanh(norm / 4));
          const dx = (-gx / norm) * len;
          const dy = (-gy / norm) * len;
          annotations.push({
            x: px + dx,
            y: py + dy,
            ax: px,
            ay: py,
            xref: 'x',
            yref: 'y',
            axref: 'x',
            ayref: 'y',
            showarrow: true,
            arrowhead: 2,
            arrowsize: 0.9,
            arrowwidth: 1.2,
            arrowcolor: 'rgba(226, 232, 240, 0.55)',
          });
        }
      }
    }

    return {
      data: traces,
      layout: {
        title: {
          text: '2D Contour Map, Descent Vector Field (-∇f) & Paths',
          font: { size: 13, color: '#E2E8F0' },
        },
        xaxis: { title: { text: 'x' }, range: [-domainBound, domainBound] },
        yaxis: { title: { text: 'y' }, range: [-domainBound, domainBound] },
        annotations,
      },
    };
  }, [gridData, analysisState.data, runs.items, criticalPoints, showVectorField, domainBound]);

  const lossComparisonSpec = useMemo((): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } => {
    return {
      data: runs.items.map((run) => ({
        x: run.result.history.map((h) => h.iteration),
        y: run.result.history.map((h) => h.fxy),
        type: 'scatter',
        mode: 'lines',
        name: run.label,
        line: { color: run.color, width: 2.2 },
      })),
      layout: {
        title: {
          text: 'Optimizer Trajectory Convergence Comparison f(X_n)',
          font: { size: 13, color: '#E2E8F0' },
        },
        xaxis: { title: { text: 'Iteration n' } },
        yaxis: { title: { text: 'f(x_n, y_n)' } },
      },
    };
  }, [runs.items]);

  function toggleAlgo(algo: OptimizerAlgorithm) {
    setSelectedAlgos((prev) =>
      prev.includes(algo) ? prev.filter((a) => a !== algo) : [...prev, algo]
    );
  }

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      {/* Left Configuration */}
      <div className="space-y-5 xl:col-span-4">
        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5">
          <h2 className="font-display text-base font-semibold text-slate-100">
            Module 3 · 3D Multivariable Lab
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Explore 3D loss surfaces, gradient vector fields, critical points, and multi-optimizer path comparisons.
          </p>

          <div className="mt-4">
            <label className="block text-xs font-medium text-slate-300">
              Benchmark Surfaces
            </label>
            <div className="mt-1.5 grid grid-cols-2 gap-1.5">
              {MULTIVAR_PRESETS.map((p) => (
                <button
                  key={p.expr}
                  type="button"
                  title={p.note}
                  onClick={() => {
                    setExprInput(p.expr);
                    setX0Input(String(p.x0));
                    setY0Input(String(p.y0));
                    setAlphaInput(String(p.alpha));
                  }}
                  className={`truncate rounded border px-2.5 py-1.5 text-left font-mono text-xs transition-colors ${
                    exprInput === p.expr
                      ? 'border-cyan-500/60 bg-cyan-500/10 text-cyan-300'
                      : 'border-slate-800 bg-[#0B0F17] text-slate-300 hover:border-slate-700 hover:text-slate-100'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-4 space-y-3">
            <div>
              <label htmlFor="m3-expr" className="block text-xs font-medium text-slate-300">
                Multivariable Function f(x, y)
              </label>
              <input
                id="m3-expr"
                type="text"
                value={exprInput}
                onChange={(e) => setExprInput(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="m3-x0" className="block text-xs font-medium text-slate-300">
                  Start x₀
                </label>
                <input
                  id="m3-x0"
                  type="number"
                  step="any"
                  value={x0Input}
                  onChange={(e) => setX0Input(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <label htmlFor="m3-y0" className="block text-xs font-medium text-slate-300">
                  Start y₀
                </label>
                <input
                  id="m3-y0"
                  type="number"
                  step="any"
                  value={y0Input}
                  onChange={(e) => setY0Input(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <label htmlFor="m3-alpha" className="block text-xs font-medium text-slate-300">
                  Learning Rate α
                </label>
                <input
                  id="m3-alpha"
                  type="number"
                  step="0.01"
                  value={alphaInput}
                  onChange={(e) => setAlphaInput(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <label htmlFor="m3-maxiter" className="block text-xs font-medium text-slate-300">
                  Max Iterations
                </label>
                <input
                  id="m3-maxiter"
                  type="number"
                  step="10"
                  value={maxIterInput}
                  onChange={(e) => setMaxIterInput(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between text-xs">
                <label htmlFor="m3-bound" className="font-medium text-slate-300">
                  Surface Domain [-R, R]²
                </label>
                <span className="font-mono text-cyan-300 tabular-nums">R = {domainBound}</span>
              </div>
              <input
                id="m3-bound"
                type="range"
                min={2.5}
                max={10}
                step={0.5}
                value={domainBound}
                onChange={(e) => setDomainBound(Number(e.target.value))}
                className="mt-1.5 w-full accent-cyan-500"
              />
            </div>

            <div>
              <div className="text-xs font-medium text-slate-300">
                Compare Optimization Paths
              </div>
              <div className="mt-1.5 grid grid-cols-2 gap-2">
                {(['gd', 'momentum', 'rmsprop', 'adam'] as OptimizerAlgorithm[]).map((algo) => {
                  const checked = selectedAlgos.includes(algo);
                  return (
                    <label
                      key={algo}
                      className={`flex cursor-pointer items-center gap-2 rounded border px-2.5 py-1.5 text-xs ${
                        checked
                          ? 'border-slate-600 bg-[#0B0F17] text-slate-100'
                          : 'border-slate-800 bg-[#0B0F17]/60 text-slate-400'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={checked}
                        onChange={() => toggleAlgo(algo)}
                        className="accent-cyan-500"
                      />
                      <span
                        className="inline-block h-2 w-2 rounded-full"
                        style={{ backgroundColor: ALGO_COLORS[algo] }}
                      />
                      <span>{ALGO_LABELS[algo]}</span>
                    </label>
                  );
                })}
              </div>
            </div>

            <div className="space-y-2 pt-1 text-xs text-slate-300">
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={showVectorField}
                  onChange={(e) => setShowVectorField(e.target.checked)}
                  className="accent-cyan-500"
                />
                Overlay gradient descent vector field (-∇f)
              </label>
              <label className="flex cursor-pointer items-center gap-2">
                <input
                  type="checkbox"
                  checked={compareSecondStart}
                  onChange={(e) => setCompareSecondStart(e.target.checked)}
                  className="accent-rose-500"
                />
                Compare secondary starting point
              </label>
            </div>

            {compareSecondStart && (
              <div className="grid grid-cols-2 gap-3 rounded border border-slate-800 bg-[#0B0F17] p-3">
                <div>
                  <label htmlFor="m3-x0-alt" className="block text-xs text-slate-400">
                    Alt Start x₀
                  </label>
                  <input
                    id="m3-x0-alt"
                    type="number"
                    step="any"
                    value={x0AltInput}
                    onChange={(e) => setX0AltInput(e.target.value)}
                    className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-2.5 py-1.5 font-mono text-xs text-slate-100"
                  />
                </div>
                <div>
                  <label htmlFor="m3-y0-alt" className="block text-xs text-slate-400">
                    Alt Start y₀
                  </label>
                  <input
                    id="m3-y0-alt"
                    type="number"
                    step="any"
                    value={y0AltInput}
                    onChange={(e) => setY0AltInput(e.target.value)}
                    className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-2.5 py-1.5 font-mono text-xs text-slate-100"
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Partial Derivatives & Real Analysis Definitions */}
        {analysisState.data && (
          <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 space-y-3">
            <h3 className="font-display text-sm font-semibold text-slate-100">
              Partial Derivatives & Critical-Point Guide
            </h3>
            <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
              <MathTex
                block
                tex={`\\frac{\\partial f}{\\partial x} = ${analysisState.data.dfDxLatex}, \\quad \\frac{\\partial f}{\\partial y} = ${analysisState.data.dfDyLatex}`}
              />
            </div>
            <div className="space-y-2 text-xs leading-relaxed text-slate-300">
              <div>
                <span className="font-semibold text-slate-100">Gradient (∇f): </span>
                Points in the direction of steepest ascent; Gradient Descent moves along −∇f.
              </div>
              <div>
                <span className="font-semibold text-slate-100">Critical Point: </span>
                A point where ∇f(x*, y*) = 0 (both partial derivatives vanish).
              </div>
              <div>
                <span className="font-semibold text-slate-100">Minimum vs Maximum vs Saddle: </span>
                At a critical point, positive Hessian eigenvalues yield a local minimum, negative eigenvalues yield a local maximum, and opposite-sign eigenvalues yield a saddle point.
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Right Visualization Stage */}
      <div className="space-y-5 xl:col-span-8">
        {runs.error && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 p-4 text-sm text-rose-200">
            <span className="font-mono font-semibold">✖ Input Error: </span>
            {runs.error}
          </div>
        )}

        {analysisState.data && (
          <>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <MetricCard
                label="Detected Critical Points"
                value={String(criticalPoints.length)}
                unit={`in [-${domainBound}, ${domainBound}]²`}
                tone="cyan"
                detail={
                  criticalPoints[0]
                    ? `${criticalPoints[0].hessian.classification} at (${criticalPoints[0].x.toFixed(2)}, ${criticalPoints[0].y.toFixed(2)})`
                    : 'No stationary root in window'
                }
              />
              <MetricCard
                label="Active Paths"
                value={String(runs.items.length)}
                detail="Simultaneous trajectories"
              />
              <MetricCard
                label="Best Final Loss"
                value={
                  runs.items.length > 0
                    ? Math.min(...runs.items.map((r) => r.result.finalFxy)).toExponential(3)
                    : '—'
                }
                tone="emerald"
              />
              <MetricCard
                label="Converged Paths"
                value={`${
                  runs.items.filter(
                    (r) =>
                      r.result.status === 'converged_grad' ||
                      r.result.status === 'converged_step'
                  ).length
                } / ${runs.items.length}`}
                tone="default"
              />
            </div>

            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              <PlotlyChart
                data={surface3DSpec.data}
                layout={surface3DSpec.layout}
                height={430}
              />
              <PlotlyChart
                data={contour2DSpec.data}
                layout={contour2DSpec.layout}
                height={430}
              />
            </div>

            <PlotlyChart
              data={lossComparisonSpec.data}
              layout={lossComparisonSpec.layout}
              height={300}
            />

            {/* Path Comparison Summary Table */}
            <div className="rounded-lg border border-slate-800 bg-[#0F141D]">
              <div className="border-b border-slate-800 px-4 py-3">
                <h3 className="font-display text-sm font-semibold text-slate-100">
                  Optimization Path Comparison Summary
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs tabular-nums">
                  <thead className="bg-[#0B0F17] text-slate-400">
                    <tr>
                      <th className="px-4 py-2.5 font-medium">Path / Optimizer</th>
                      <th className="px-4 py-2.5 font-medium">Iterations</th>
                      <th className="px-4 py-2.5 font-medium">Final (x*, y*)</th>
                      <th className="px-4 py-2.5 font-medium">Final f(X*)</th>
                      <th className="px-4 py-2.5 font-medium">||∇f||</th>
                      <th className="px-4 py-2.5 font-medium">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/80 text-slate-200">
                    {runs.items.map((item) => {
                      const ok =
                        item.result.status === 'converged_grad' ||
                        item.result.status === 'converged_step';
                      const div =
                        item.result.status === 'diverged' ||
                        item.result.status === 'numerical_error';
                      return (
                        <tr key={item.label} className="hover:bg-slate-800/30">
                          <td className="px-4 py-2.5 font-sans font-medium text-slate-100">
                            <span
                              className="mr-2 inline-block h-2 w-2 rounded-full"
                              style={{ backgroundColor: item.color }}
                            />
                            {item.label}
                          </td>
                          <td className="px-4 py-2.5">{item.result.totalIterations}</td>
                          <td className="px-4 py-2.5">
                            ({item.result.finalX.toFixed(4)}, {item.result.finalY.toFixed(4)})
                          </td>
                          <td className="px-4 py-2.5">{item.result.finalFxy.toExponential(3)}</td>
                          <td className="px-4 py-2.5">
                            {item.result.finalGradNorm.toExponential(3)}
                          </td>
                          <td className="px-4 py-2.5">
                            <span
                              className={
                                ok
                                  ? 'text-emerald-400'
                                  : div
                                  ? 'text-rose-400'
                                  : 'text-amber-400'
                              }
                            >
                              {ok ? '● Converged' : div ? '✖ Diverged' : '▲ Max Iter'}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
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

import { useMemo, useState } from 'react';
import { MathTex } from '../components/MathTex';
import { MetricCard } from '../components/MetricCard';
import { PlotlyChart } from '../components/PlotlyChart';
import { analyzeExpression } from '../core/mathematics';
import {
  GradientDescentResult1D,
  GradientDescentResult2D,
  runGradientDescent1D,
  runGradientDescent2D,
} from '../core/optimizers';

interface RegimeClassification {
  label: string;
  tone: 'emerald' | 'cyan' | 'amber' | 'rose';
  note: string;
}

function classify1DRegime(res: GradientDescentResult1D): RegimeClassification {
  if (res.status === 'diverged' || res.status === 'numerical_error') {
    return {
      label: '✖ Divergent / Unstable',
      tone: 'rose',
      note: 'Step size exceeded local stability bound; iterates diverge or overflow.',
    };
  }
  // Check sign alternation in iterates (oscillation)
  let signFlips = 0;
  for (let i = 1; i < Math.min(res.history.length, 20); i++) {
    const prev = res.history[i - 1].grad;
    const curr = res.history[i].grad;
    if (prev * curr < 0) signFlips++;
  }
  if (signFlips >= 3) {
    return {
      label: '▲ Oscillatory Convergence',
      tone: 'amber',
      note: 'Gradient changes sign across consecutive steps, overshooting the minimum while damping.',
    };
  }
  if (res.status === 'max_iterations' || res.totalIterations > 250) {
    return {
      label: '▲ Slow Convergence',
      tone: 'amber',
      note: 'Monotonic descent, but small step size requires many iterations to approach limit.',
    };
  }
  return {
    label: '● Stable Convergence',
    tone: 'emerald',
    note: 'Smooth contraction toward stationary limit within tolerance.',
  };
}

function classify2DRegime(res: GradientDescentResult2D): RegimeClassification {
  if (res.status === 'diverged' || res.status === 'numerical_error') {
    return {
      label: '✖ Divergent / Unstable',
      tone: 'rose',
      note: 'Step size exceeded local curvature stability bound; sequence diverged.',
    };
  }
  if (res.status === 'max_iterations' || res.totalIterations > 250) {
    return {
      label: '▲ Slow Convergence',
      tone: 'amber',
      note: 'Slow contraction; anisotropic curvature or small alpha limits progress.',
    };
  }
  return {
    label: '● Stable Convergence',
    tone: 'emerald',
    note: 'Norm ||∇f|| contracted below stopping tolerance.',
  };
}

const PRESET_OBJECTIVES = [
  { label: 'x² (Quadratic Limit Proof)', expr: 'x**2', x0: 10, y0: 0, alphas: '0.01, 0.1, 0.9, 1.1' },
  { label: 'x⁴ - 3x² + x (Non-Convex)', expr: 'x**4 - 3*x**2 + x', x0: 2.2, y0: 0, alphas: '0.01, 0.05, 0.12, 0.25' },
  { label: '0.5x² + 2sin(x)', expr: '0.5*x**2 + 2*sin(x)', x0: 3.0, y0: 0, alphas: '0.02, 0.15, 0.6, 1.2' },
  { label: 'x² + 5y² (2D Valley)', expr: 'x**2 + 5*y**2', x0: 4, y0: 2, alphas: '0.02, 0.1, 0.19, 0.25' },
];

const SERIES_COLORS = ['#38BDF8', '#10B981', '#F59E0B', '#F43F5E', '#A78BFA'];

export function ConvergenceAnalyzerView() {
  const [exprInput, setExprInput] = useState('x**2');
  const [x0Input, setX0Input] = useState('10');
  const [y0Input, setY0Input] = useState('0');
  const [alphasInput, setAlphasInput] = useState('0.01, 0.1, 0.9, 1.1');
  const [maxIter, setMaxIter] = useState(120);
  const [tolerance, setTolerance] = useState(1e-6);

  const parsedExpr = useMemo(() => {
    try {
      return { data: analyzeExpression(exprInput), error: null as string | null };
    } catch (err) {
      return {
        data: null,
        error: err instanceof Error ? err.message : 'Invalid expression.',
      };
    }
  }, [exprInput]);

  const alphaList = useMemo(() => {
    return alphasInput
      .split(',')
      .map((s) => Number(s.trim()))
      .filter((n) => Number.isFinite(n) && n > 0)
      .slice(0, 5);
  }, [alphasInput]);

  const { experiments, simError } = useMemo(() => {
    if (!parsedExpr.data) return { experiments: [], simError: null as string | null };
    if (alphaList.length === 0) {
      return {
        experiments: [],
        simError: 'Enter at least one positive finite learning rate α (e.g., 0.01, 0.1).',
      };
    }
    const x0 = Number(x0Input);
    const y0 = Number(y0Input);
    if (!Number.isFinite(x0) || !Number.isFinite(y0)) {
      return { experiments: [], simError: 'Initial coordinates x0 and y0 must be finite numbers.' };
    }

    try {
      const items = alphaList.map((alpha, idx) => {
        const color = SERIES_COLORS[idx % SERIES_COLORS.length];
        if (parsedExpr.data!.dimension === 1) {
          const res = runGradientDescent1D(exprInput, x0, alpha, maxIter, tolerance);
          return {
            dim: 1 as const,
            alpha,
            color,
            res,
            regime: classify1DRegime(res),
          };
        }
        const res = runGradientDescent2D(exprInput, x0, y0, alpha, maxIter, tolerance, 'gd');
        return {
          dim: 2 as const,
          alpha,
          color,
          res,
          regime: classify2DRegime(res),
        };
      });
      return { experiments: items, simError: null as string | null };
    } catch (err) {
      return {
        experiments: [],
        simError: err instanceof Error ? err.message : 'Invalid simulation parameters.',
      };
    }
  }, [parsedExpr.data, alphaList, exprInput, x0Input, y0Input, maxIter, tolerance]);

  const toleranceSensitivity = useMemo(() => {
    if (!parsedExpr.data || experiments.length === 0 || simError) return [];
    const x0 = Number(x0Input);
    const y0 = Number(y0Input);
    if (!Number.isFinite(x0) || !Number.isFinite(y0)) return [];

    // Pick first converging alpha or first alpha as representative
    const rep =
      experiments.find((e) => e.regime.tone === 'emerald')?.alpha ?? experiments[0].alpha;
    const tols = [1e-2, 1e-4, 1e-6, 1e-8];
    try {
      return tols.map((tol) => {
        if (parsedExpr.data!.dimension === 1) {
          const r = runGradientDescent1D(exprInput, x0, rep, 1000, tol);
          return {
            tol,
            alpha: rep,
            iters: r.totalIterations,
            finalVal: r.finalFx,
            finalGrad: Math.abs(r.finalGrad),
            status: r.statusLabel,
          };
        }
        const r = runGradientDescent2D(exprInput, x0, y0, rep, 1000, tol, 'gd');
        return {
          tol,
          alpha: rep,
          iters: r.totalIterations,
          finalVal: r.finalFxy,
          finalGrad: r.finalGradNorm,
          status: r.statusLabel,
        };
      });
    } catch {
      return [];
    }
  }, [parsedExpr.data, experiments, simError, exprInput, x0Input, y0Input]);

  const sequencePlot = useMemo((): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } => {
    const traces: Plotly.Data[] = experiments.map((exp) => {
      if (exp.dim === 1) {
        const hist = exp.res.history.slice(0, 60);
        return {
          x: hist.map((h) => h.iteration),
          y: hist.map((h) => h.x),
          type: 'scatter',
          mode: 'lines+markers',
          name: `α = ${exp.alpha} (${exp.regime.label.slice(2)})`,
          line: { color: exp.color, width: 2 },
          marker: { size: 4 },
        };
      }
      const hist = exp.res.history.slice(0, 60);
      return {
        x: hist.map((h) => h.iteration),
        y: hist.map((h) => h.fxy),
        type: 'scatter',
        mode: 'lines+markers',
        name: `α = ${exp.alpha}`,
        line: { color: exp.color, width: 2 },
        marker: { size: 4 },
      };
    });

    return {
      data: traces,
      layout: {
        title: {
          text:
            parsedExpr.data?.dimension === 1
              ? 'Iterate Sequence x_n vs Iteration n (First 60 Steps)'
              : 'Objective Sequence f(X_n) vs Iteration n (First 60 Steps)',
          font: { size: 13, color: '#E2E8F0' },
        },
        xaxis: { title: { text: 'Iteration n' } },
        yaxis: {
          title: { text: parsedExpr.data?.dimension === 1 ? 'x_n' : 'f(X_n)' },
        },
      },
    };
  }, [experiments, parsedExpr.data]);

  const errorDecayPlot = useMemo((): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } => {
    const traces: Plotly.Data[] = experiments.map((exp) => {
      if (exp.dim === 1) {
        const hist = exp.res.history;
        return {
          x: hist.map((h) => h.iteration),
          y: hist.map((h) => Math.max(h.absGrad, 1e-16)),
          type: 'scatter',
          mode: 'lines',
          name: `α = ${exp.alpha}`,
          line: { color: exp.color, width: 2 },
        };
      }
      const hist = exp.res.history;
      return {
        x: hist.map((h) => h.iteration),
        y: hist.map((h) => Math.max(h.gradNorm, 1e-16)),
        type: 'scatter',
        mode: 'lines',
        name: `α = ${exp.alpha}`,
        line: { color: exp.color, width: 2 },
      };
    });

    return {
      data: traces,
      layout: {
        title: {
          text: 'Log-Scale Gradient Norm ||∇f(X_n)|| Convergence Rate',
          font: { size: 13, color: '#E2E8F0' },
        },
        xaxis: { title: { text: 'Iteration n' } },
        yaxis: { title: { text: '||∇f||' }, type: 'log' },
      },
    };
  }, [experiments]);

  const bestExperiment = useMemo(() => {
    const converged = experiments.filter((e) => e.regime.tone === 'emerald');
    if (converged.length === 0) return null;
    return converged.sort((a, b) => a.res.totalIterations - b.res.totalIterations)[0];
  }, [experiments]);

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      {/* Left Configuration */}
      <div className="space-y-5 xl:col-span-4">
        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5">
          <h2 className="font-display text-base font-semibold text-slate-100">
            Module 5 · Convergence & Learning Rate Analyzer
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Compare learning rates, initial points, and tolerances to observe contraction, oscillation, and divergence.
          </p>

          <div className="mt-4 grid grid-cols-2 gap-1.5">
            {PRESET_OBJECTIVES.map((p) => (
              <button
                key={p.expr}
                type="button"
                onClick={() => {
                  setExprInput(p.expr);
                  setX0Input(String(p.x0));
                  setY0Input(String(p.y0));
                  setAlphasInput(p.alphas);
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

          <div className="mt-4 space-y-3">
            <div>
              <label htmlFor="m5-expr" className="block text-xs font-medium text-slate-300">
                Objective Function
              </label>
              <input
                id="m5-expr"
                type="text"
                value={exprInput}
                onChange={(e) => setExprInput(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <label htmlFor="m5-alphas" className="block text-xs font-medium text-slate-300">
                Learning Rates α (comma-separated, up to 5)
              </label>
              <input
                id="m5-alphas"
                type="text"
                value={alphasInput}
                onChange={(e) => setAlphasInput(e.target.value)}
                placeholder="0.01, 0.1, 0.9, 1.1"
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="m5-x0" className="block text-xs font-medium text-slate-300">
                  Initial x₀
                </label>
                <input
                  id="m5-x0"
                  type="number"
                  step="any"
                  value={x0Input}
                  onChange={(e) => setX0Input(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>
              {parsedExpr.data?.dimension === 2 && (
                <div>
                  <label htmlFor="m5-y0" className="block text-xs font-medium text-slate-300">
                    Initial y₀
                  </label>
                  <input
                    id="m5-y0"
                    type="number"
                    step="any"
                    value={y0Input}
                    onChange={(e) => setY0Input(e.target.value)}
                    className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                  />
                </div>
              )}
              <div>
                <label htmlFor="m5-maxiter" className="block text-xs font-medium text-slate-300">
                  Max Iterations
                </label>
                <input
                  id="m5-maxiter"
                  type="number"
                  min={10}
                  max={2000}
                  value={maxIter}
                  onChange={(e) => setMaxIter(Number(e.target.value))}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <label htmlFor="m5-tol" className="block text-xs font-medium text-slate-300">
                  Tolerance ε
                </label>
                <input
                  id="m5-tol"
                  type="number"
                  step="any"
                  value={tolerance}
                  onChange={(e) => setTolerance(Number(e.target.value))}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Real Analysis Caution Box */}
        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 space-y-2.5 text-xs leading-relaxed text-slate-300">
          <h3 className="font-display text-sm font-semibold text-slate-100">
            Real Analysis Convergence Principles
          </h3>
          <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
            <MathTex block tex="x_{n+1} = (1 - 2\alpha)x_n \implies x_n = (1 - 2\alpha)^n x_0" />
          </div>
          <p>
            For <span className="font-mono text-cyan-300">f(x) = x²</span>, the sequence converges to{' '}
            <span className="font-mono">0</span> for every finite <span className="font-mono">x₀</span>{' '}
            if and only if <span className="font-mono">|1 - 2α| &lt; 1</span> (that is,{' '}
            <span className="font-mono">0 &lt; α &lt; 1</span>). For{' '}
            <span className="font-mono">0.5 &lt; α &lt; 1</span>, iterates oscillate across zero while converging; for{' '}
            <span className="font-mono">α &gt; 1</span>, the sequence diverges.
          </p>
          <p className="text-slate-400">
            General Gradient Descent is <span className="text-slate-200 font-medium">not guaranteed to converge</span> for arbitrary functions, non-Lipschitz gradients, or poor initialization. Any effective learning rate below is supported solely by numerical experiments on the chosen function.
          </p>
        </div>
      </div>

      {/* Right Stage */}
      <div className="space-y-5 xl:col-span-8">
        {(parsedExpr.error || simError) && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 p-4 text-sm text-rose-200">
            <span className="font-mono font-semibold">✖ Validation Error: </span>
            {parsedExpr.error ?? simError}
          </div>
        )}

        {experiments.length > 0 && (
          <>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <MetricCard
                label="Tested Learning Rates"
                value={String(experiments.length)}
                detail={alphaList.map((a) => `α=${a}`).join(' · ')}
                tone="cyan"
              />
              <MetricCard
                label="Empirically Fastest α"
                value={bestExperiment ? `α = ${bestExperiment.alpha}` : 'None'}
                detail={
                  bestExperiment
                    ? `Converged in ${bestExperiment.res.totalIterations} steps`
                    : 'No tested α converged'
                }
                tone={bestExperiment ? 'emerald' : 'amber'}
              />
              <MetricCard
                label="Divergent Regimes"
                value={String(experiments.filter((e) => e.regime.tone === 'rose').length)}
                detail="Violated curvature bound"
                tone={
                  experiments.some((e) => e.regime.tone === 'rose') ? 'rose' : 'default'
                }
              />
              <MetricCard
                label="Oscillatory / Slow"
                value={String(experiments.filter((e) => e.regime.tone === 'amber').length)}
                detail="Sub-optimal contraction"
                tone="amber"
              />
            </div>

            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              <PlotlyChart
                data={sequencePlot.data}
                layout={sequencePlot.layout}
                height={380}
              />
              <PlotlyChart
                data={errorDecayPlot.data}
                layout={errorDecayPlot.layout}
                height={380}
              />
            </div>

            {/* Learning Rate Regime Table */}
            <div className="rounded-lg border border-slate-800 bg-[#0F141D]">
              <div className="border-b border-slate-800 px-4 py-3">
                <h3 className="font-display text-sm font-semibold text-slate-100">
                  Sequence Limit & Regime Comparison
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs tabular-nums">
                  <thead className="bg-[#0B0F17] text-slate-400">
                    <tr>
                      <th className="px-4 py-2.5 font-medium">α</th>
                      <th className="px-4 py-2.5 font-medium">Regime</th>
                      <th className="px-4 py-2.5 font-medium">Iters</th>
                      <th className="px-4 py-2.5 font-medium">Final Point</th>
                      <th className="px-4 py-2.5 font-medium">Final Loss</th>
                      <th className="px-4 py-2.5 font-medium">Observation</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/80 text-slate-200">
                    {experiments.map((exp) => {
                      const toneCls =
                        exp.regime.tone === 'emerald'
                          ? 'text-emerald-400'
                          : exp.regime.tone === 'rose'
                          ? 'text-rose-400'
                          : 'text-amber-400';
                      return (
                        <tr key={exp.alpha} className="hover:bg-slate-800/30">
                          <td className="px-4 py-2.5 font-semibold" style={{ color: exp.color }}>
                            {exp.alpha}
                          </td>
                          <td className={`px-4 py-2.5 whitespace-nowrap ${toneCls}`}>
                            {exp.regime.label}
                          </td>
                          <td className="px-4 py-2.5">{exp.res.totalIterations}</td>
                          <td className="px-4 py-2.5">
                            {exp.dim === 1
                              ? exp.res.finalX.toExponential(3)
                              : `(${exp.res.finalX.toFixed(3)}, ${exp.res.finalY.toFixed(3)})`}
                          </td>
                          <td className="px-4 py-2.5">
                            {(exp.dim === 1 ? exp.res.finalFx : exp.res.finalFxy).toExponential(3)}
                          </td>
                          <td className="px-4 py-2.5 font-sans text-slate-400">
                            {exp.regime.note}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Tolerance Sensitivity */}
            {toleranceSensitivity.length > 0 && (
              <div className="rounded-lg border border-slate-800 bg-[#0F141D]">
                <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
                  <h3 className="font-display text-sm font-semibold text-slate-100">
                    Stopping Tolerance Sensitivity (α = {toleranceSensitivity[0].alpha})
                  </h3>
                  <span className="font-mono text-xs text-slate-400">
                    Max 1,000 iterations
                  </span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left font-mono text-xs tabular-nums">
                    <thead className="bg-[#0B0F17] text-slate-400">
                      <tr>
                        <th className="px-4 py-2.5 font-medium">Tolerance ε</th>
                        <th className="px-4 py-2.5 font-medium">Iterations Needed</th>
                        <th className="px-4 py-2.5 font-medium">Final f(X*)</th>
                        <th className="px-4 py-2.5 font-medium">Final ||∇f||</th>
                        <th className="px-4 py-2.5 font-medium">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/80 text-slate-200">
                      {toleranceSensitivity.map((row) => (
                        <tr key={row.tol} className="hover:bg-slate-800/30">
                          <td className="px-4 py-2">{row.tol.toExponential(0)}</td>
                          <td className="px-4 py-2 text-cyan-300">{row.iters}</td>
                          <td className="px-4 py-2">{row.finalVal.toExponential(3)}</td>
                          <td className="px-4 py-2">{row.finalGrad.toExponential(3)}</td>
                          <td className="px-4 py-2 font-sans text-slate-300">{row.status}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

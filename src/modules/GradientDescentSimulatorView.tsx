import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
  SkipForward,
} from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { MathTex } from '../components/MathTex';
import { MetricCard } from '../components/MetricCard';
import { PlotlyChart } from '../components/PlotlyChart';
import { analyzeExpression } from '../core/mathematics';
import {
  OptimizerAlgorithm,
  StoppingCriterion,
  runGradientDescent1D,
  runGradientDescent2D,
} from '../core/optimizers';

const SIM_PRESETS = [
  { label: 'Test A · x^2 (α=0.1)', expr: 'x^2', x0: 10, y0: 0, alpha: 0.1, maxIter: 150 },
  { label: 'Test B · x^2 (α=0.01)', expr: 'x^2', x0: 10, y0: 0, alpha: 0.01, maxIter: 500 },
  { label: 'Test C · x^2 (α=1.1 Diverge)', expr: 'x^2', x0: 10, y0: 0, alpha: 1.1, maxIter: 40 },
  { label: 'Double-Well · x^4 - 3*x^2 + x', expr: 'x^4 - 3*x^2 + x', x0: 2.2, y0: 0, alpha: 0.04, maxIter: 120 },
  { label: 'Elliptic 2D · x^2 + 2*y^2', expr: 'x^2 + 2*y^2', x0: 4, y0: -3, alpha: 0.12, maxIter: 150 },
  { label: 'Coupled 2D · x^2 + x*y + y^2', expr: 'x^2 + x*y + y^2', x0: 4, y0: 2.5, alpha: 0.15, maxIter: 150 },
];

export function GradientDescentSimulatorView() {
  const [exprInput, setExprInput] = useState('x^2');
  const [x0Input, setX0Input] = useState('10');
  const [y0Input, setY0Input] = useState('0');
  const [alphaInput, setAlphaInput] = useState('0.1');
  const [maxIterInput, setMaxIterInput] = useState('150');
  const [tolInput, setTolInput] = useState('1e-6');
  const [stoppingCriterion, setStoppingCriterion] = useState<StoppingCriterion>('both');
  const [algorithm, setAlgorithm] = useState<OptimizerAlgorithm>('gd');

  const [runNonce, setRunNonce] = useState(0);
  const [stepIndex, setStepIndex] = useState<number>(Infinity);
  const [isPlaying, setIsPlaying] = useState(false);

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

  const simulation = useMemo(() => {
    void runNonce;
    if (!parsedExpr.data) {
      return { error: parsedExpr.error, dim: 1 as const, res1D: null, res2D: null };
    }
    const x0 = Number(x0Input);
    const y0 = Number(y0Input);
    const alpha = Number(alphaInput);
    const maxIter = Number(maxIterInput);
    const tol = Number(tolInput);

    try {
      if (parsedExpr.data.dimension === 1) {
        const res1D = runGradientDescent1D(
          exprInput,
          x0,
          alpha,
          maxIter,
          tol,
          stoppingCriterion
        );
        return { error: null, dim: 1 as const, res1D, res2D: null };
      }
      const res2D = runGradientDescent2D(
        exprInput,
        x0,
        y0,
        alpha,
        maxIter,
        tol,
        algorithm,
        {},
        stoppingCriterion
      );
      return { error: null, dim: 2 as const, res1D: null, res2D };
    } catch (err) {
      return {
        error: err instanceof Error ? err.message : 'Invalid optimizer configuration.',
        dim: parsedExpr.data.dimension,
        res1D: null,
        res2D: null,
      };
    }
  }, [
    parsedExpr,
    exprInput,
    x0Input,
    y0Input,
    alphaInput,
    maxIterInput,
    tolInput,
    stoppingCriterion,
    algorithm,
    runNonce,
  ]);

  const totalRecorded =
    simulation.res1D?.history.length ?? simulation.res2D?.history.length ?? 0;
  const maxStep = Math.max(0, totalRecorded - 1);
  const activeStep = Math.min(stepIndex, maxStep);

  useEffect(() => {
    if (!isPlaying) return;
    if (activeStep >= maxStep) {
      setIsPlaying(false);
      return;
    }
    const timer = window.setInterval(() => {
      setStepIndex((prev) => {
        const next = (Number.isFinite(prev) ? prev : 0) + 1;
        if (next >= maxStep) {
          setIsPlaying(false);
          return maxStep;
        }
        return next;
      });
    }, 180);
    return () => window.clearInterval(timer);
  }, [isPlaying, activeStep, maxStep]);

  const trajectoryPlot = useMemo((): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } => {
    if (!parsedExpr.data) return { data: [], layout: {} };

    if (simulation.dim === 1 && simulation.res1D) {
      const fullHist = simulation.res1D.history;
      const visibleHist = fullHist.slice(0, activeStep + 1);
      const xs = fullHist.map((h) => h.x);
      const maxAbs = Math.min(120, Math.max(4, ...xs.map(Math.abs)) * 1.15);

      const curveX: number[] = [];
      const curveY: number[] = [];
      const samples = 240;
      for (let i = 0; i <= samples; i++) {
        const xv = -maxAbs + (i / samples) * (2 * maxAbs);
        const yv = parsedExpr.data.evalF(xv);
        curveX.push(xv);
        curveY.push(Number.isFinite(yv) ? yv : NaN);
      }

      const currentPt = visibleHist[visibleHist.length - 1];

      return {
        data: [
          {
            x: curveX,
            y: curveY,
            type: 'scatter',
            mode: 'lines',
            name: 'f(x)',
            line: { color: '#38BDF8', width: 2.2 },
          },
          {
            x: visibleHist.map((h) => h.x),
            y: visibleHist.map((h) => h.fx),
            type: 'scatter',
            mode: 'lines+markers',
            name: `Trajectory (0..${activeStep})`,
            line: { color: '#F59E0B', width: 1.5, dash: 'dot' },
            marker: { size: 6, color: '#F59E0B' },
          },
          ...(currentPt
            ? [
                {
                  x: [currentPt.x],
                  y: [currentPt.fx],
                  type: 'scatter' as const,
                  mode: 'markers' as const,
                  name: `Current x_${activeStep}`,
                  marker: {
                    size: 11,
                    color: '#10B981',
                    line: { color: '#07090E', width: 2 },
                  },
                },
              ]
            : []),
        ],
        layout: {
          title: {
            text: `Optimization Trajectory (Step ${activeStep} of ${maxStep})`,
            font: { size: 13, color: '#E2E8F0' },
          },
          xaxis: { title: { text: 'x' } },
          yaxis: { title: { text: 'f(x)' } },
        },
      };
    }

    if (simulation.dim === 2 && simulation.res2D) {
      const fullHist = simulation.res2D.history;
      const visibleHist = fullHist.slice(0, activeStep + 1);
      const xs = fullHist.map((h) => h.x);
      const ys = fullHist.map((h) => h.y);
      const bound = Math.min(25, Math.max(4, ...xs.map(Math.abs), ...ys.map(Math.abs)) * 1.25);

      const gridN = 45;
      const gx: number[] = [];
      const gy: number[] = [];
      const gz: number[][] = [];
      for (let i = 0; i < gridN; i++) {
        const v = -bound + (i / (gridN - 1)) * (2 * bound);
        gx.push(v);
        gy.push(v);
      }
      for (let j = 0; j < gridN; j++) {
        const row: number[] = [];
        for (let i = 0; i < gridN; i++) {
          const val = parsedExpr.data.evalF(gx[i], gy[j]);
          row.push(Number.isFinite(val) ? val : NaN);
        }
        gz.push(row);
      }

      const currentPt = visibleHist[visibleHist.length - 1];

      return {
        data: [
          {
            x: gx,
            y: gy,
            z: gz,
            type: 'contour',
            colorscale: 'Cividis',
            contours: { coloring: 'heatmap' },
            name: 'f(x, y)',
          },
          {
            x: visibleHist.map((h) => h.x),
            y: visibleHist.map((h) => h.y),
            type: 'scatter',
            mode: 'lines+markers',
            name: `Path (0..${activeStep})`,
            line: { color: '#06B6D4', width: 2.2 },
            marker: { size: 6, color: '#F59E0B' },
          },
          ...(currentPt
            ? [
                {
                  x: [currentPt.x],
                  y: [currentPt.y],
                  type: 'scatter' as const,
                  mode: 'markers' as const,
                  name: `Current X_${activeStep}`,
                  marker: {
                    size: 11,
                    color: '#10B981',
                    line: { color: '#07090E', width: 2 },
                  },
                },
              ]
            : []),
        ],
        layout: {
          title: {
            text: `2D Contour Trajectory (${simulation.res2D.algorithm.toUpperCase()} · Step ${activeStep}/${maxStep})`,
            font: { size: 13, color: '#E2E8F0' },
          },
          xaxis: { title: { text: 'x' } },
          yaxis: { title: { text: 'y' } },
        },
      };
    }

    return { data: [], layout: {} };
  }, [parsedExpr.data, simulation, activeStep, maxStep]);

  const convergencePlot = useMemo((): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } => {
    const res1 = simulation.res1D;
    const res2 = simulation.res2D;
    if (!res1 && !res2) return { data: [], layout: {} };

    const iters = res1
      ? res1.history.slice(0, activeStep + 1).map((h) => h.iteration)
      : res2!.history.slice(0, activeStep + 1).map((h) => h.iteration);
    const lossVals = res1
      ? res1.history.slice(0, activeStep + 1).map((h) => h.fx)
      : res2!.history.slice(0, activeStep + 1).map((h) => h.fxy);
    const gradVals = res1
      ? res1.history.slice(0, activeStep + 1).map((h) => Math.max(h.absGrad, 1e-16))
      : res2!.history.slice(0, activeStep + 1).map((h) => Math.max(h.gradNorm, 1e-16));

    return {
      data: [
        {
          x: iters,
          y: lossVals,
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Objective f(X_n)',
          line: { color: '#06B6D4', width: 2 },
          marker: { size: 4 },
        },
        {
          x: iters,
          y: gradVals,
          type: 'scatter',
          mode: 'lines',
          name: '||∇f(X_n)||',
          yaxis: 'y2',
          line: { color: '#F59E0B', width: 1.8, dash: 'dot' },
        },
      ],
      layout: {
        title: {
          text: 'Convergence Curves: Objective Value & Gradient Norm',
          font: { size: 13, color: '#E2E8F0' },
        },
        xaxis: { title: { text: 'Iteration n' } },
        yaxis: { title: { text: 'f(X_n)' } },
        yaxis2: {
          title: { text: '||∇f(X_n)||' },
          overlaying: 'y',
          side: 'right',
          type: 'log',
          gridcolor: 'rgba(0,0,0,0)',
          tickfont: { family: 'JetBrains Mono, monospace', size: 10, color: '#F59E0B' },
        },
      },
    };
  }, [simulation, activeStep]);

  const activeResult = simulation.res1D ?? simulation.res2D;
  const statusTone =
    activeResult?.status === 'converged_grad' || activeResult?.status === 'converged_step'
      ? 'emerald'
      : activeResult?.status === 'diverged' || activeResult?.status === 'numerical_error'
      ? 'rose'
      : 'amber';
  const statusGlyph = statusTone === 'emerald' ? '●' : statusTone === 'rose' ? '✖' : '▲';

  const current1D = simulation.res1D?.history[activeStep];
  const next1D = simulation.res1D?.history[activeStep + 1];
  const current2D = simulation.res2D?.history[activeStep];
  const next2D = simulation.res2D?.history[activeStep + 1];

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      {/* Left Configuration Column */}
      <div className="space-y-5 xl:col-span-4">
        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5">
          <h2 className="font-display text-base font-semibold text-slate-100">
            Module 2 · Gradient Descent Simulator
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Execute genuine Gradient Descent from scratch and inspect each step of the iterate sequence.
          </p>

          <div className="mt-3 rounded border border-slate-800 bg-[#0B0F17] px-3 py-2">
            <MathTex
              block
              tex={
                parsedExpr.data?.dimension === 2
                  ? 'X_{n+1} = X_n - \\alpha \\nabla f(X_n)'
                  : "x_{n+1} = x_n - \\alpha f'(x_n)"
              }
            />
          </div>

          <div className="mt-4">
            <label className="block text-xs font-medium text-slate-300">
              Scenarios & Presets
            </label>
            <div className="mt-1.5 grid grid-cols-2 gap-1.5">
              {SIM_PRESETS.map((p) => (
                <button
                  key={p.label}
                  type="button"
                  onClick={() => {
                    setIsPlaying(false);
                    setExprInput(p.expr);
                    setX0Input(String(p.x0));
                    setY0Input(String(p.y0));
                    setAlphaInput(String(p.alpha));
                    setMaxIterInput(String(p.maxIter));
                    setStepIndex(Infinity);
                  }}
                  className="truncate rounded border border-slate-800 bg-[#0B0F17] px-2.5 py-1.5 text-left font-mono text-xs text-slate-300 transition-colors hover:border-cyan-500/50 hover:text-cyan-200"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-4 space-y-3">
            <div>
              <label htmlFor="m2-expr" className="block text-xs font-medium text-slate-300">
                Objective Function
              </label>
              <input
                id="m2-expr"
                type="text"
                value={exprInput}
                onChange={(e) => {
                  setIsPlaying(false);
                  setExprInput(e.target.value);
                  setStepIndex(Infinity);
                }}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="m2-x0" className="block text-xs font-medium text-slate-300">
                  Initial x₀
                </label>
                <input
                  id="m2-x0"
                  type="number"
                  step="any"
                  value={x0Input}
                  onChange={(e) => setX0Input(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              {parsedExpr.data?.dimension === 2 && (
                <div>
                  <label htmlFor="m2-y0" className="block text-xs font-medium text-slate-300">
                    Initial y₀
                  </label>
                  <input
                    id="m2-y0"
                    type="number"
                    step="any"
                    value={y0Input}
                    onChange={(e) => setY0Input(e.target.value)}
                    className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                  />
                </div>
              )}

              <div>
                <label htmlFor="m2-alpha" className="block text-xs font-medium text-slate-300">
                  Learning Rate α
                </label>
                <input
                  id="m2-alpha"
                  type="number"
                  step="0.01"
                  value={alphaInput}
                  onChange={(e) => setAlphaInput(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label htmlFor="m2-maxiter" className="block text-xs font-medium text-slate-300">
                  Max Iterations
                </label>
                <input
                  id="m2-maxiter"
                  type="number"
                  step="10"
                  value={maxIterInput}
                  onChange={(e) => setMaxIterInput(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label htmlFor="m2-tol" className="block text-xs font-medium text-slate-300">
                  Tolerance ε
                </label>
                <input
                  id="m2-tol"
                  type="text"
                  value={tolInput}
                  onChange={(e) => setTolInput(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label htmlFor="m2-stop" className="block text-xs font-medium text-slate-300">
                  Stopping Criterion
                </label>
                <select
                  id="m2-stop"
                  value={stoppingCriterion}
                  onChange={(e) => setStoppingCriterion(e.target.value as StoppingCriterion)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-2.5 py-2 font-mono text-xs text-slate-100 focus:border-cyan-500 focus:outline-none"
                >
                  <option value="both">Gradient or Step</option>
                  <option value="gradient">Gradient Norm ≤ ε</option>
                  <option value="step">Step Size ≤ ε</option>
                  <option value="fixed">Fixed Max Iter</option>
                </select>
              </div>
            </div>

            {parsedExpr.data?.dimension === 2 && (
              <div>
                <label htmlFor="m2-algo" className="block text-xs font-medium text-slate-300">
                  Update Rule
                </label>
                <select
                  id="m2-algo"
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value as OptimizerAlgorithm)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-xs text-slate-100 focus:border-cyan-500 focus:outline-none"
                >
                  <option value="gd">Batch Gradient Descent</option>
                  <option value="momentum">Momentum</option>
                  <option value="rmsprop">RMSProp</option>
                  <option value="adam">Adam</option>
                </select>
              </div>
            )}

            {/* Playback & Rerun Controls */}
            <div className="space-y-2 pt-2">
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={() => {
                    setIsPlaying(false);
                    setRunNonce((n) => n + 1);
                    setStepIndex(0);
                    setIsPlaying(true);
                  }}
                  className="flex flex-1 items-center justify-center gap-1.5 rounded-md bg-cyan-500 px-3 py-2 text-xs font-semibold text-slate-950 transition-colors hover:bg-cyan-400 whitespace-nowrap"
                >
                  <Play className="h-3.5 w-3.5" />
                  Start Animation
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setIsPlaying(false);
                    setRunNonce((n) => n + 1);
                    setStepIndex(Infinity);
                  }}
                  className="flex items-center gap-1.5 rounded-md border border-slate-700 bg-[#0B0F17] px-3 py-2 text-xs font-medium text-slate-200 hover:border-slate-600 whitespace-nowrap"
                >
                  <SkipForward className="h-3.5 w-3.5" />
                  Rerun All
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setIsPlaying(false);
                    setExprInput('x^2');
                    setX0Input('10');
                    setY0Input('0');
                    setAlphaInput('0.1');
                    setMaxIterInput('150');
                    setTolInput('1e-6');
                    setStoppingCriterion('both');
                    setAlgorithm('gd');
                    setStepIndex(Infinity);
                  }}
                  className="flex items-center gap-1.5 rounded-md border border-slate-700 bg-[#0B0F17] px-3 py-2 text-xs font-medium text-slate-300 hover:border-slate-600 whitespace-nowrap"
                >
                  <RotateCcw className="h-3.5 w-3.5" />
                  Reset
                </button>
              </div>

              {maxStep > 0 && (
                <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400">Step Playback</span>
                    <span className="font-mono text-cyan-300 tabular-nums">
                      n = {activeStep} / {maxStep}
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={maxStep}
                    value={activeStep}
                    onChange={(e) => {
                      setIsPlaying(false);
                      setStepIndex(Number(e.target.value));
                    }}
                    className="mt-2 w-full accent-cyan-500"
                  />
                  <div className="mt-2 flex items-center justify-between gap-2">
                    <button
                      type="button"
                      onClick={() => {
                        setIsPlaying(false);
                        setStepIndex(Math.max(0, activeStep - 1));
                      }}
                      className="flex items-center gap-1 rounded border border-slate-700 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-800"
                    >
                      <ChevronLeft className="h-3.5 w-3.5" />
                      Prev
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        if (activeStep >= maxStep) setStepIndex(0);
                        setIsPlaying((p) => !p);
                      }}
                      className="flex items-center gap-1 rounded border border-slate-700 px-3 py-1 text-xs text-cyan-300 hover:bg-slate-800"
                    >
                      {isPlaying ? (
                        <>
                          <Pause className="h-3.5 w-3.5" /> Pause
                        </>
                      ) : (
                        <>
                          <Play className="h-3.5 w-3.5" /> Play
                        </>
                      )}
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        setIsPlaying(false);
                        setStepIndex(Math.min(maxStep, activeStep + 1));
                      }}
                      className="flex items-center gap-1 rounded border border-slate-700 px-2.5 py-1 text-xs text-slate-300 hover:bg-slate-800"
                    >
                      Next
                      <ChevronRight className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Step-by-Step Numerical Explanation */}
        {activeResult && (current1D || current2D) && (
          <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5">
            <h3 className="font-display text-sm font-semibold text-slate-100">
              Step-by-Step Explanation (Iteration {activeStep})
            </h3>
            {current1D && (
              <div className="mt-3 space-y-2 text-xs text-slate-300">
                <div className="rounded border border-slate-800 bg-[#0B0F17] p-3 font-mono tabular-nums">
                  <div>x_{current1D.iteration} = {current1D.x.toFixed(6)}</div>
                  <div>f(x_{current1D.iteration}) = {current1D.fx.toExponential(4)}</div>
                  <div>f&apos;(x_{current1D.iteration}) = {current1D.grad.toExponential(4)}</div>
                </div>
                {next1D ? (
                  <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
                    <div className="text-slate-400">Next Update Calculation:</div>
                    <div className="mt-1 font-mono text-cyan-300 tabular-nums">
                      x_{next1D.iteration} = {current1D.x.toFixed(4)} − ({activeResult.alpha}) · (
                      {current1D.grad.toFixed(4)}) = {next1D.x.toFixed(6)}
                    </div>
                  </div>
                ) : (
                  <div className="text-slate-400">{activeResult.explanation}</div>
                )}
              </div>
            )}
            {current2D && (
              <div className="mt-3 space-y-2 text-xs text-slate-300">
                <div className="rounded border border-slate-800 bg-[#0B0F17] p-3 font-mono tabular-nums">
                  <div>
                    X_{current2D.iteration} = ({current2D.x.toFixed(5)}, {current2D.y.toFixed(5)})
                  </div>
                  <div>f(X_{current2D.iteration}) = {current2D.fxy.toExponential(4)}</div>
                  <div>
                    ∇f = ({current2D.gradX.toExponential(3)}, {current2D.gradY.toExponential(3)})
                  </div>
                  <div>||∇f|| = {current2D.gradNorm.toExponential(4)}</div>
                </div>
                {next2D ? (
                  <div className="rounded border border-slate-800 bg-[#0B0F17] p-3 font-mono text-cyan-300 tabular-nums">
                    X_{next2D.iteration} = ({next2D.x.toFixed(5)}, {next2D.y.toFixed(5)}) · Δ ={' '}
                    {next2D.stepDelta.toExponential(3)}
                  </div>
                ) : (
                  <div className="text-slate-400">{activeResult.explanation}</div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right Visualization & History */}
      <div className="space-y-5 xl:col-span-8">
        {simulation.error && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 p-4 text-sm text-rose-200">
            <span className="font-mono font-semibold">✖ Validation Error: </span>
            {simulation.error}
          </div>
        )}

        {activeResult && (
          <>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <MetricCard
                label="Convergence Status"
                value={`${statusGlyph} ${
                  statusTone === 'emerald'
                    ? 'Converged'
                    : statusTone === 'rose'
                    ? 'Diverged'
                    : 'Stopped'
                }`}
                detail={activeResult.statusLabel}
                tone={statusTone}
              />
              <MetricCard
                label="Total Iterations"
                value={String(activeResult.totalIterations)}
                unit={`/ ${activeResult.maxIterations}`}
                detail={`α = ${activeResult.alpha}`}
                tone="cyan"
              />
              <MetricCard
                label="Final Estimated Minimum"
                value={
                  simulation.dim === 1 && simulation.res1D
                    ? simulation.res1D.finalX.toExponential(3)
                    : simulation.res2D
                    ? `(${simulation.res2D.finalX.toFixed(3)}, ${simulation.res2D.finalY.toFixed(3)})`
                    : '—'
                }
                detail={
                  simulation.dim === 1 && simulation.res1D
                    ? `f(x*) = ${simulation.res1D.finalFx.toExponential(3)}`
                    : simulation.res2D
                    ? `f(X*) = ${simulation.res2D.finalFxy.toExponential(3)}`
                    : undefined
                }
              />
              <MetricCard
                label="Final Gradient"
                value={
                  simulation.dim === 1 && simulation.res1D
                    ? Math.abs(simulation.res1D.finalGrad).toExponential(3)
                    : simulation.res2D
                    ? simulation.res2D.finalGradNorm.toExponential(3)
                    : '—'
                }
                detail={`tol = ${activeResult.tolerance.toExponential(1)}`}
              />
            </div>

            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              <PlotlyChart
                data={trajectoryPlot.data}
                layout={trajectoryPlot.layout}
                height={380}
              />
              <PlotlyChart
                data={convergencePlot.data}
                layout={convergencePlot.layout}
                height={380}
              />
            </div>

            {/* Iteration-History Table */}
            <div className="rounded-lg border border-slate-800 bg-[#0F141D]">
              <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
                <h3 className="font-display text-sm font-semibold text-slate-100">
                  Iteration History (Click a row to inspect step)
                </h3>
                <span className="font-mono text-xs text-slate-400 tabular-nums">
                  Showing {Math.min(activeResult.history.length, 200)} of{' '}
                  {activeResult.history.length} states
                </span>
              </div>
              <div className="max-h-72 overflow-auto">
                <table className="w-full text-left font-mono text-xs tabular-nums">
                  <thead className="sticky top-0 bg-[#0B0F17] text-slate-400">
                    <tr>
                      <th className="px-4 py-2.5 font-medium">n</th>
                      <th className="px-4 py-2.5 font-medium">x_n</th>
                      {simulation.dim === 2 && <th className="px-4 py-2.5 font-medium">y_n</th>}
                      <th className="px-4 py-2.5 font-medium">f(X_n)</th>
                      <th className="px-4 py-2.5 font-medium">
                        {simulation.dim === 1 ? "f'(x_n)" : '||∇f(X_n)||'}
                      </th>
                      <th className="px-4 py-2.5 font-medium">Step Δ</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/70 text-slate-200">
                    {simulation.dim === 1 &&
                      simulation.res1D?.history.slice(0, 200).map((row) => (
                        <tr
                          key={row.iteration}
                          onClick={() => {
                            setIsPlaying(false);
                            setStepIndex(row.iteration);
                          }}
                          className={`cursor-pointer transition-colors ${
                            row.iteration === activeStep
                              ? 'bg-cyan-500/15 text-cyan-200'
                              : 'hover:bg-slate-800/30'
                          }`}
                        >
                          <td className="px-4 py-2 text-slate-400">{row.iteration}</td>
                          <td className="px-4 py-2">{row.x.toFixed(6)}</td>
                          <td className="px-4 py-2">{row.fx.toExponential(4)}</td>
                          <td className="px-4 py-2">{row.grad.toExponential(4)}</td>
                          <td className="px-4 py-2 text-slate-400">
                            {row.stepDelta.toExponential(3)}
                          </td>
                        </tr>
                      ))}
                    {simulation.dim === 2 &&
                      simulation.res2D?.history.slice(0, 200).map((row) => (
                        <tr
                          key={row.iteration}
                          onClick={() => {
                            setIsPlaying(false);
                            setStepIndex(row.iteration);
                          }}
                          className={`cursor-pointer transition-colors ${
                            row.iteration === activeStep
                              ? 'bg-cyan-500/15 text-cyan-200'
                              : 'hover:bg-slate-800/30'
                          }`}
                        >
                          <td className="px-4 py-2 text-slate-400">{row.iteration}</td>
                          <td className="px-4 py-2">{row.x.toFixed(5)}</td>
                          <td className="px-4 py-2">{row.y.toFixed(5)}</td>
                          <td className="px-4 py-2">{row.fxy.toExponential(4)}</td>
                          <td className="px-4 py-2">{row.gradNorm.toExponential(4)}</td>
                          <td className="px-4 py-2 text-slate-400">
                            {row.stepDelta.toExponential(3)}
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

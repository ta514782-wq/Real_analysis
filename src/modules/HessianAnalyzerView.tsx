import { useMemo, useState } from 'react';
import { MathTex } from '../components/MathTex';
import { MetricCard } from '../components/MetricCard';
import { PlotlyChart } from '../components/PlotlyChart';
import {
  analyzeExpression,
  analyzeHessian2D,
  findCriticalPoints2D,
  scanGridConvexity2D,
} from '../core/mathematics';

const HESSIAN_PRESETS = [
  { label: 'x² + y² (Convex Bowl)', expr: 'x**2 + y**2', x: 0, y: 0 },
  { label: 'x² + 10y² (Ill-Conditioned)', expr: 'x**2 + 10*y**2', x: 0, y: 0 },
  { label: 'x² - y² (Saddle Point)', expr: 'x**2 - y**2', x: 0, y: 0 },
  { label: '-x² - y² (Concave Dome)', expr: '-x**2 - y**2', x: 0, y: 0 },
  { label: 'x⁴ - 2x² + y² (Non-Convex)', expr: 'x**4 - 2*x**2 + y**2', x: 0, y: 0 },
  { label: '(1-x)² + 5(y-x²)² (Rosenbrock)', expr: '(1 - x)**2 + 5*(y - x**2)**2', x: 1, y: 1 },
];

export function HessianAnalyzerView() {
  const [exprInput, setExprInput] = useState('x**2 - y**2');
  const [evalX, setEvalX] = useState(0);
  const [evalY, setEvalY] = useState(0);
  const [scanBound, setScanBound] = useState(3);

  const analysisState = useMemo(() => {
    try {
      const data = analyzeExpression(exprInput, { allowedDimensions: [2] });
      return { data, error: null as string | null };
    } catch (err) {
      return {
        data: null,
        error: err instanceof Error ? err.message : 'Enter a valid function f(x, y).',
      };
    }
  }, [exprInput]);

  const hessianResult = useMemo(() => {
    if (!analysisState.data) return null;
    return analyzeHessian2D(analysisState.data, evalX, evalY);
  }, [analysisState.data, evalX, evalY]);

  const gridConvexity = useMemo(() => {
    if (!analysisState.data) return null;
    return scanGridConvexity2D(analysisState.data, scanBound, 25);
  }, [analysisState.data, scanBound]);

  const criticalPoints = useMemo(() => {
    if (!analysisState.data) return [];
    return findCriticalPoints2D(analysisState.data, scanBound);
  }, [analysisState.data, scanBound]);

  const eigenHeatmapSpec = useMemo((): {
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
  } => {
    if (!gridConvexity) return { data: [], layout: {} };
    return {
      data: [
        {
          x: gridConvexity.xs,
          y: gridConvexity.ys,
          z: gridConvexity.minEigenGrid,
          type: 'heatmap',
          colorscale: 'RdBu',
          zmid: 0,
          name: 'λ_min(x, y)',
        },
        {
          x: [evalX],
          y: [evalY],
          type: 'scatter',
          mode: 'markers',
          name: `Eval (${evalX.toFixed(2)}, ${evalY.toFixed(2)})`,
          marker: { size: 10, color: '#F59E0B', line: { color: '#07090E', width: 2 } },
        },
      ],
      layout: {
        title: {
          text: 'Minimum Hessian Eigenvalue λ_min(x, y) Across Domain',
          font: { size: 13, color: '#E2E8F0' },
        },
        xaxis: { title: { text: 'x' } },
        yaxis: { title: { text: 'y' } },
      },
    };
  }, [gridConvexity, evalX, evalY]);

  const quadraticTaylorSpec = useMemo((): {
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
  } => {
    if (!analysisState.data || !hessianResult) return { data: [], layout: {} };
    const f0 = analysisState.data.evalF(evalX, evalY);
    const [gx, gy] = analysisState.data.evalGrad(evalX, evalY);
    const [[h11, h12], [, h22]] = hessianResult.matrix;

    const span = 1.5;
    const gridN = 30;
    const xs: number[] = [];
    const ys: number[] = [];
    const zs: number[][] = [];
    for (let i = 0; i < gridN; i++) {
      xs.push(evalX - span + (i / (gridN - 1)) * (2 * span));
      ys.push(evalY - span + (i / (gridN - 1)) * (2 * span));
    }
    for (let j = 0; j < gridN; j++) {
      const row: number[] = [];
      for (let i = 0; i < gridN; i++) {
        const dx = xs[i] - evalX;
        const dy = ys[j] - evalY;
        const q =
          f0 +
          gx * dx +
          gy * dy +
          0.5 * (h11 * dx * dx + 2 * h12 * dx * dy + h22 * dy * dy);
        row.push(Number.isFinite(q) ? q : NaN);
      }
      zs.push(row);
    }

    return {
      data: [
        {
          x: xs,
          y: ys,
          z: zs,
          type: 'surface',
          colorscale: 'Viridis',
          showscale: false,
          opacity: 0.9,
        } as Plotly.Data,
      ],
      layout: {
        title: {
          text: `Local 2nd-Order Taylor Quadratic Model at (${evalX.toFixed(2)}, ${evalY.toFixed(2)})`,
          font: { size: 13, color: '#E2E8F0' },
        },
        margin: { t: 40, r: 10, b: 10, l: 10 },
        scene: {
          xaxis: { title: { text: 'x' }, gridcolor: '#1E293B', color: '#94A3B8' },
          yaxis: { title: { text: 'y' }, gridcolor: '#1E293B', color: '#94A3B8' },
          zaxis: { title: { text: 'Q(x, y)' }, gridcolor: '#1E293B', color: '#94A3B8' },
          bgcolor: '#0B0F17',
        },
      },
    };
  }, [analysisState.data, hessianResult, evalX, evalY]);

  const conditionNumber = useMemo(() => {
    if (!hessianResult) return null;
    const absEigs = hessianResult.eigenvalues.map(Math.abs);
    const maxA = Math.max(...absEigs);
    const minA = Math.min(...absEigs);
    if (minA < 1e-10) return Infinity;
    return maxA / minA;
  }, [hessianResult]);

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      {/* Left Configuration */}
      <div className="space-y-5 xl:col-span-4">
        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5">
          <h2 className="font-display text-base font-semibold text-slate-100">
            Hessian Matrix & Convexity Analyzer
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Analyze second-order partial derivatives, eigenvalues, positive/negative definiteness, and domain convexity.
          </p>

          <div className="mt-4 grid grid-cols-2 gap-1.5">
            {HESSIAN_PRESETS.map((p) => (
              <button
                key={p.expr}
                type="button"
                onClick={() => {
                  setExprInput(p.expr);
                  setEvalX(p.x);
                  setEvalY(p.y);
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
              <label htmlFor="hess-expr" className="block text-xs font-medium text-slate-300">
                Function f(x, y)
              </label>
              <input
                id="hess-expr"
                type="text"
                value={exprInput}
                onChange={(e) => setExprInput(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label htmlFor="hess-x" className="block text-xs font-medium text-slate-300">
                  Evaluation x
                </label>
                <input
                  id="hess-x"
                  type="number"
                  step="0.25"
                  value={evalX}
                  onChange={(e) => setEvalX(Number(e.target.value))}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>
              <div>
                <label htmlFor="hess-y" className="block text-xs font-medium text-slate-300">
                  Evaluation y
                </label>
                <input
                  id="hess-y"
                  type="number"
                  step="0.25"
                  value={evalY}
                  onChange={(e) => setEvalY(Number(e.target.value))}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>
            </div>

            <div>
              <div className="flex items-center justify-between text-xs">
                <label htmlFor="hess-bound" className="font-medium text-slate-300">
                  Grid Scan Radius [-R, R]²
                </label>
                <span className="font-mono text-cyan-300 tabular-nums">R = {scanBound}</span>
              </div>
              <input
                id="hess-bound"
                type="range"
                min={1.5}
                max={6}
                step={0.5}
                value={scanBound}
                onChange={(e) => setScanBound(Number(e.target.value))}
                className="mt-1.5 w-full accent-cyan-500"
              />
            </div>
          </div>
        </div>

        {/* Symbolic & Numeric Hessian Matrices */}
        {analysisState.data && hessianResult && (
          <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 space-y-3">
            <h3 className="font-display text-sm font-semibold text-slate-100">
              Symbolic & Evaluated Hessian
            </h3>
            <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
              <div className="text-xs text-slate-400">Symbolic H(x, y)</div>
              <MathTex
                block
                tex={`H(x, y) = \\begin{bmatrix} ${analysisState.data.d2fDx2Latex} & ${analysisState.data.d2fDxDyLatex} \\\\ ${analysisState.data.d2fDxDyLatex} & ${analysisState.data.d2fDy2Latex} \\end{bmatrix}`}
              />
            </div>
            <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
              <div className="text-xs text-slate-400">
                Numeric H({evalX.toFixed(2)}, {evalY.toFixed(2)})
              </div>
              <MathTex
                block
                tex={`H = \\begin{bmatrix} ${hessianResult.matrix[0][0].toFixed(3)} & ${hessianResult.matrix[0][1].toFixed(3)} \\\\ ${hessianResult.matrix[1][0].toFixed(3)} & ${hessianResult.matrix[1][1].toFixed(3)} \\end{bmatrix}`}
              />
            </div>
            <div className="text-xs leading-relaxed text-slate-300">
              <span className="font-semibold text-slate-100">Interpretation: </span>
              {hessianResult.convexityNote}
              {conditionNumber !== null &&
                Number.isFinite(conditionNumber) &&
                hessianResult.definiteness === 'Positive Definite' && (
                  <span>
                    {' '}
                    Condition number κ = {conditionNumber.toFixed(2)}; stable quadratic step bound is α &lt;{' '}
                    {(2 / Math.max(...hessianResult.eigenvalues)).toFixed(4)}.
                  </span>
                )}
            </div>
          </div>
        )}
      </div>

      {/* Right Stage */}
      <div className="space-y-5 xl:col-span-8">
        {analysisState.error && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 p-4 text-sm text-rose-200">
            <span className="font-mono font-semibold">✖ Input Error: </span>
            {analysisState.error}
          </div>
        )}

        {hessianResult && gridConvexity && (
          <>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <MetricCard
                label="Definiteness at Point"
                value={hessianResult.definiteness}
                detail={hessianResult.classification}
                tone={
                  hessianResult.definiteness === 'Positive Definite'
                    ? 'emerald'
                    : hessianResult.definiteness === 'Indefinite (Saddle)'
                    ? 'amber'
                    : hessianResult.definiteness === 'Negative Definite'
                    ? 'rose'
                    : 'default'
                }
              />
              <MetricCard
                label="Eigenvalues (λ₁, λ₂)"
                value={`${hessianResult.eigenvalues[0].toFixed(2)}, ${hessianResult.eigenvalues[1].toFixed(2)}`}
                detail={`det(H) = ${hessianResult.determinant.toFixed(3)} · tr(H) = ${hessianResult.trace.toFixed(3)}`}
                tone="cyan"
              />
              <MetricCard
                label="Domain Convexity Scan"
                value={
                  gridConvexity.globalClassification.includes('Convex')
                    ? '● Convex'
                    : gridConvexity.globalClassification.includes('Concave')
                    ? '▲ Concave'
                    : '✖ Non-Convex'
                }
                detail={gridConvexity.globalClassification}
                tone={
                  gridConvexity.globalClassification.includes('Convex')
                    ? 'emerald'
                    : 'amber'
                }
              />
              <MetricCard
                label="Condition Number κ"
                value={
                  conditionNumber !== null && Number.isFinite(conditionNumber)
                    ? conditionNumber.toFixed(2)
                    : '∞'
                }
                detail={`min λ = ${gridConvexity.minEigenvalue.toFixed(2)} on grid`}
              />
            </div>

            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              <PlotlyChart
                data={eigenHeatmapSpec.data}
                layout={eigenHeatmapSpec.layout}
                height={400}
              />
              <PlotlyChart
                data={quadraticTaylorSpec.data}
                layout={quadraticTaylorSpec.layout}
                height={400}
              />
            </div>

            {/* Detected Critical Points Table */}
            <div className="rounded-lg border border-slate-800 bg-[#0F141D]">
              <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
                <h3 className="font-display text-sm font-semibold text-slate-100">
                  Critical Points Classification (∇f = 0 in [-{scanBound}, {scanBound}]²)
                </h3>
                <span className="font-mono text-xs text-slate-400">
                  {criticalPoints.length} found
                </span>
              </div>
              {criticalPoints.length === 0 ? (
                <div className="px-4 py-4 text-xs text-slate-400">
                  No critical points detected inside [-{scanBound}, {scanBound}]².
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-left font-mono text-xs tabular-nums">
                    <thead className="bg-[#0B0F17] text-slate-400">
                      <tr>
                        <th className="px-4 py-2.5 font-medium">(x*, y*)</th>
                        <th className="px-4 py-2.5 font-medium">f(x*, y*)</th>
                        <th className="px-4 py-2.5 font-medium">λ₁, λ₂</th>
                        <th className="px-4 py-2.5 font-medium">Definiteness</th>
                        <th className="px-4 py-2.5 font-medium">Classification</th>
                        <th className="px-4 py-2.5 font-medium">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/80 text-slate-200">
                      {criticalPoints.map((pt, idx) => (
                        <tr key={idx} className="hover:bg-slate-800/30">
                          <td className="px-4 py-2.5">
                            ({pt.x.toFixed(4)}, {pt.y.toFixed(4)})
                          </td>
                          <td className="px-4 py-2.5">{pt.fxy.toFixed(4)}</td>
                          <td className="px-4 py-2.5">
                            {pt.hessian.eigenvalues[0].toFixed(2)},{' '}
                            {pt.hessian.eigenvalues[1].toFixed(2)}
                          </td>
                          <td className="px-4 py-2.5">{pt.hessian.definiteness}</td>
                          <td className="px-4 py-2.5 text-cyan-300">
                            {pt.hessian.classification}
                          </td>
                          <td className="px-4 py-2.5">
                            <button
                              type="button"
                              onClick={() => {
                                setEvalX(Number(pt.x.toFixed(3)));
                                setEvalY(Number(pt.y.toFixed(3)));
                              }}
                              className="text-cyan-400 hover:text-cyan-300 underline"
                            >
                              Inspect Point
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

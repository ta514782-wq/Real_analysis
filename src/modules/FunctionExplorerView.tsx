import { useMemo, useState } from 'react';
import { MathTex } from '../components/MathTex';
import { MetricCard } from '../components/MetricCard';
import { PlotlyChart } from '../components/PlotlyChart';
import {
  analyzeExpression,
  analyzeHessian2D,
  findStationaryPoints1D,
} from '../core/mathematics';

const FUNCTION_PRESETS = [
  { label: 'x**2', expr: 'x**2', desc: 'Convex parabola' },
  { label: 'x**3 - 4*x', expr: 'x**3 - 4*x', desc: 'Cubic with local min & max' },
  { label: 'sin(x)', expr: 'sin(x)', desc: 'Periodic trigonometric' },
  { label: 'exp(x)', expr: 'exp(x)', desc: 'Strictly increasing exponential' },
  { label: 'x**4 - 3*x**2 + x', expr: 'x**4 - 3*x**2 + x', desc: 'Double-well quartic' },
  { label: 'x**2 + y**2', expr: 'x**2 + y**2', desc: '2D isotropic paraboloid' },
  { label: 'x**2 - y**2', expr: 'x**2 - y**2', desc: '2D hyperbolic saddle' },
];

export function FunctionExplorerView() {
  const [exprInput, setExprInput] = useState('x**3 - 4*x');
  const [domainBound, setDomainBound] = useState(4);
  const [probeX, setProbeX] = useState(1.0);
  const [probeY, setProbeY] = useState(1.0);
  const [showDerivativeCurve, setShowDerivativeCurve] = useState(true);
  const [showTangentLine, setShowTangentLine] = useState(true);
  const [plotMode2D, setPlotMode2D] = useState<'contour' | 'surface'>('contour');

  const analysisState = useMemo(() => {
    try {
      const data = analyzeExpression(exprInput);
      return { data, error: null as string | null };
    } catch (err) {
      return {
        data: null,
        error: err instanceof Error ? err.message : 'Invalid expression.',
      };
    }
  }, [exprInput]);

  const stationary1D = useMemo(() => {
    if (!analysisState.data || analysisState.data.dimension !== 1) return [];
    return findStationaryPoints1D(analysisState.data, -domainBound, domainBound);
  }, [analysisState.data, domainBound]);

  const probeStats = useMemo(() => {
    const a = analysisState.data;
    if (!a) return null;
    if (a.dimension === 1) {
      const fx = a.evalF(probeX);
      const [fp] = a.evalGrad(probeX);
      const { fxx } = a.evalSecondPartials(probeX);
      return {
        dim: 1 as const,
        fx,
        fp,
        fxx,
      };
    }
    const fxy = a.evalF(probeX, probeY);
    const [gx, gy] = a.evalGrad(probeX, probeY);
    const hessian = analyzeHessian2D(a, probeX, probeY);
    return {
      dim: 2 as const,
      fxy,
      gx,
      gy,
      gradNorm: Math.hypot(gx, gy),
      hessian,
    };
  }, [analysisState.data, probeX, probeY]);

  const plotlySpec = useMemo((): { data: Plotly.Data[]; layout: Partial<Plotly.Layout> } => {
    const a = analysisState.data;
    if (!a) return { data: [], layout: {} };

    if (a.dimension === 1) {
      const xs: number[] = [];
      const fVals: number[] = [];
      const dfVals: number[] = [];
      const n = 300;
      for (let i = 0; i <= n; i++) {
        const xv = -domainBound + (i / n) * (2 * domainBound);
        const yv = a.evalF(xv);
        const dyv = a.evalGrad(xv)[0];
        xs.push(xv);
        fVals.push(Number.isFinite(yv) && Math.abs(yv) < 1e4 ? yv : NaN);
        dfVals.push(Number.isFinite(dyv) && Math.abs(dyv) < 1e4 ? dyv : NaN);
      }

      const traces: Plotly.Data[] = [
        {
          x: xs,
          y: fVals,
          type: 'scatter',
          mode: 'lines',
          name: 'f(x)',
          line: { color: '#38BDF8', width: 2.5 },
        },
      ];

      if (showDerivativeCurve) {
        traces.push({
          x: xs,
          y: dfVals,
          type: 'scatter',
          mode: 'lines',
          name: "f'(x)",
          line: { color: '#A78BFA', width: 1.75, dash: 'dash' },
        });
      }

      if (showTangentLine && probeStats?.dim === 1 && Number.isFinite(probeStats.fx) && Number.isFinite(probeStats.fp)) {
        const span = Math.max(0.8, domainBound * 0.25);
        const tx = [probeX - span, probeX + span];
        const ty = tx.map((xv) => probeStats.fx + probeStats.fp * (xv - probeX));
        traces.push({
          x: tx,
          y: ty,
          type: 'scatter',
          mode: 'lines',
          name: `Tangent at x=${probeX.toFixed(2)}`,
          line: { color: '#F59E0B', width: 2 },
        });
        traces.push({
          x: [probeX],
          y: [probeStats.fx],
          type: 'scatter',
          mode: 'markers',
          name: 'Probe Point',
          marker: { color: '#F59E0B', size: 10, line: { color: '#07090E', width: 1.5 } },
        });
      }

      if (stationary1D.length > 0) {
        traces.push({
          x: stationary1D.map((s) => s.x),
          y: stationary1D.map((s) => s.fx),
          text: stationary1D.map((s) => `${s.classification} (${s.x.toFixed(3)}, ${s.fx.toFixed(3)})`),
          hoverinfo: 'text',
          type: 'scatter',
          mode: 'markers',
          name: 'Stationary Points',
          marker: {
            size: 10,
            symbol: 'diamond',
            color: stationary1D.map((s) =>
              s.classification === 'Local Minimum'
                ? '#10B981'
                : s.classification === 'Local Maximum'
                ? '#F43F5E'
                : '#EAB308'
            ),
            line: { color: '#07090E', width: 1.5 },
          },
        });
      }

      return {
        data: traces,
        layout: {
          title: {
            text: `Function & Derivative Visualization on [${-domainBound}, ${domainBound}]`,
            font: { size: 13, color: '#E2E8F0' },
          },
          xaxis: { title: { text: 'x' } },
          yaxis: { title: { text: 'Value' } },
        },
      };
    }

    // 2D Visualization
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

    if (plotMode2D === 'surface') {
      const probeZ = a.evalF(probeX, probeY);
      return {
        data: [
          {
            x: gx,
            y: gy,
            z: gz,
            type: 'surface',
            colorscale: 'Cividis',
            opacity: 0.92,
            name: 'f(x, y)',
          } as Plotly.Data,
          {
            x: [probeX],
            y: [probeY],
            z: [Number.isFinite(probeZ) ? probeZ : 0],
            type: 'scatter3d',
            mode: 'markers',
            name: 'Probe (x, y)',
            marker: { size: 6, color: '#F59E0B' },
          } as Plotly.Data,
        ],
        layout: {
          title: {
            text: '3D Surface Preview f(x, y)',
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
    }

    const traces2D: Plotly.Data[] = [
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
        x: [probeX],
        y: [probeY],
        type: 'scatter',
        mode: 'markers',
        name: `Probe (${probeX.toFixed(2)}, ${probeY.toFixed(2)})`,
        marker: { size: 10, color: '#F59E0B', line: { color: '#07090E', width: 1.5 } },
      },
    ];

    if (probeStats?.dim === 2 && probeStats.gradNorm > 1e-8) {
      const scale = Math.min(domainBound * 0.35, 1.2) / Math.max(1, probeStats.gradNorm);
      traces2D.push({
        x: [probeX, probeX - probeStats.gx * scale],
        y: [probeY, probeY - probeStats.gy * scale],
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Descent Direction (-∇f)',
        line: { color: '#06B6D4', width: 2.5 },
        marker: { size: [4, 8], color: '#06B6D4' },
      });
    }

    return {
      data: traces2D,
      layout: {
        title: {
          text: '2D Contour Map & Steepest Descent Vector',
          font: { size: 13, color: '#E2E8F0' },
        },
        xaxis: { title: { text: 'x' } },
        yaxis: { title: { text: 'y' } },
      },
    };
  }, [analysisState.data, domainBound, probeX, probeY, showDerivativeCurve, showTangentLine, plotMode2D, probeStats, stationary1D]);

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      {/* Left Controls */}
      <div className="space-y-5 xl:col-span-4">
        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5">
          <h2 className="font-display text-base font-semibold text-slate-100">
            Module 1 · Function Explorer
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Inspect symbolic derivatives, tangent slopes, stationary points, and local curvature safely via AST parsing.
          </p>

          <div className="mt-4">
            <label className="block text-xs font-medium text-slate-300">
              Predefined Functions
            </label>
            <div className="mt-1.5 grid grid-cols-2 gap-1.5">
              {FUNCTION_PRESETS.map((p) => (
                <button
                  key={p.expr}
                  type="button"
                  onClick={() => setExprInput(p.expr)}
                  title={p.desc}
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
              <label htmlFor="m1-expr" className="block text-xs font-medium text-slate-300">
                Expression f(x) or f(x, y)
              </label>
              <input
                id="m1-expr"
                type="text"
                value={exprInput}
                onChange={(e) => setExprInput(e.target.value)}
                placeholder="e.g. x**3 - 4*x"
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <div className="flex items-center justify-between text-xs">
                <label htmlFor="m1-domain" className="font-medium text-slate-300">
                  Plot Domain [-R, R]
                </label>
                <span className="font-mono text-cyan-300 tabular-nums">R = {domainBound}</span>
              </div>
              <input
                id="m1-domain"
                type="range"
                min={2}
                max={12}
                step={1}
                value={domainBound}
                onChange={(e) => {
                  const val = Number(e.target.value);
                  setDomainBound(val);
                  setProbeX((px) => Math.max(-val, Math.min(val, px)));
                  setProbeY((py) => Math.max(-val, Math.min(val, py)));
                }}
                className="mt-1.5 w-full accent-cyan-500"
              />
            </div>

            <div>
              <div className="flex items-center justify-between text-xs">
                <label htmlFor="m1-probex" className="font-medium text-slate-300">
                  Probe Coordinate x
                </label>
                <span className="font-mono text-amber-300 tabular-nums">
                  x = {probeX.toFixed(2)}
                </span>
              </div>
              <input
                id="m1-probex"
                type="range"
                min={-domainBound}
                max={domainBound}
                step={0.05}
                value={probeX}
                onChange={(e) => setProbeX(Number(e.target.value))}
                className="mt-1.5 w-full accent-amber-500"
              />
            </div>

            {analysisState.data?.dimension === 2 && (
              <div>
                <div className="flex items-center justify-between text-xs">
                  <label htmlFor="m1-probey" className="font-medium text-slate-300">
                    Probe Coordinate y
                  </label>
                  <span className="font-mono text-amber-300 tabular-nums">
                    y = {probeY.toFixed(2)}
                  </span>
                </div>
                <input
                  id="m1-probey"
                  type="range"
                  min={-domainBound}
                  max={domainBound}
                  step={0.05}
                  value={probeY}
                  onChange={(e) => setProbeY(Number(e.target.value))}
                  className="mt-1.5 w-full accent-amber-500"
                />
              </div>
            )}

            {analysisState.data?.dimension === 1 ? (
              <div className="flex flex-wrap gap-4 pt-1 text-xs text-slate-300">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showDerivativeCurve}
                    onChange={(e) => setShowDerivativeCurve(e.target.checked)}
                    className="accent-cyan-500"
                  />
                  Show f&apos;(x) curve
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showTangentLine}
                    onChange={(e) => setShowTangentLine(e.target.checked)}
                    className="accent-amber-500"
                  />
                  Show tangent line
                </label>
              </div>
            ) : (
              <div className="flex items-center gap-1 rounded-md bg-[#0B0F17] p-1 border border-slate-800">
                <button
                  type="button"
                  onClick={() => setPlotMode2D('contour')}
                  className={`flex-1 rounded py-1.5 text-xs font-medium transition-colors ${
                    plotMode2D === 'contour'
                      ? 'bg-cyan-500/20 text-cyan-300'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  2D Contour
                </button>
                <button
                  type="button"
                  onClick={() => setPlotMode2D('surface')}
                  className={`flex-1 rounded py-1.5 text-xs font-medium transition-colors ${
                    plotMode2D === 'surface'
                      ? 'bg-cyan-500/20 text-cyan-300'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  3D Surface
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Symbolic Expressions */}
        {analysisState.data && (
          <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 space-y-3">
            <h3 className="font-display text-sm font-semibold text-slate-100">
              Symbolic Calculus
            </h3>
            <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
              <div className="text-xs text-slate-400">Function</div>
              <MathTex
                block
                tex={
                  analysisState.data.dimension === 1
                    ? `f(x) = ${analysisState.data.latexF}`
                    : `f(x, y) = ${analysisState.data.latexF}`
                }
              />
            </div>

            {analysisState.data.dimension === 1 ? (
              <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
                <div className="text-xs text-slate-400">Symbolic Derivatives</div>
                <MathTex block tex={`f'(x) = ${analysisState.data.dfDxLatex}`} />
                <MathTex block tex={`f''(x) = ${analysisState.data.d2fDx2Latex}`} />
              </div>
            ) : (
              <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
                <div className="text-xs text-slate-400">Gradient & Hessian</div>
                <MathTex
                  block
                  tex={`\\nabla f = \\begin{bmatrix} ${analysisState.data.dfDxLatex} \\\\ ${analysisState.data.dfDyLatex} \\end{bmatrix}`}
                />
                <MathTex
                  block
                  tex={`H = \\begin{bmatrix} ${analysisState.data.d2fDx2Latex} & ${analysisState.data.d2fDxDyLatex} \\\\ ${analysisState.data.d2fDxDyLatex} & ${analysisState.data.d2fDy2Latex} \\end{bmatrix}`}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right Stage */}
      <div className="space-y-5 xl:col-span-8">
        {analysisState.error && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 p-4 text-sm text-rose-200">
            <span className="font-mono font-semibold">✖ Validation Error: </span>
            {analysisState.error}
          </div>
        )}

        {probeStats && (
          <>
            {probeStats.dim === 1 ? (
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <MetricCard
                  label="Probe f(x)"
                  value={Number.isFinite(probeStats.fx) ? probeStats.fx.toFixed(4) : 'Undefined'}
                  detail={`at x = ${probeX.toFixed(2)}`}
                  tone="cyan"
                />
                <MetricCard
                  label="Slope f'(x)"
                  value={Number.isFinite(probeStats.fp) ? probeStats.fp.toFixed(4) : 'Undefined'}
                  detail={
                    probeStats.fp > 1e-4
                      ? 'Increasing (descent goes left)'
                      : probeStats.fp < -1e-4
                      ? 'Decreasing (descent goes right)'
                      : 'Near stationary point'
                  }
                />
                <MetricCard
                  label="Curvature f''(x)"
                  value={Number.isFinite(probeStats.fxx) ? probeStats.fxx.toFixed(4) : 'Undefined'}
                  detail={
                    probeStats.fxx > 0
                      ? 'Locally convex (cup up)'
                      : probeStats.fxx < 0
                      ? 'Locally concave (cup down)'
                      : 'Zero curvature'
                  }
                />
                <MetricCard
                  label="Stationary Points"
                  value={String(stationary1D.length)}
                  unit={`in [-${domainBound}, ${domainBound}]`}
                  detail={
                    stationary1D.length > 0
                      ? `${stationary1D.filter((s) => s.classification === 'Local Minimum').length} min · ${stationary1D.filter((s) => s.classification === 'Local Maximum').length} max`
                      : 'None found in interval'
                  }
                  tone="emerald"
                />
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                <MetricCard
                  label="Probe f(x, y)"
                  value={Number.isFinite(probeStats.fxy) ? probeStats.fxy.toFixed(4) : 'Undefined'}
                  detail={`at (${probeX.toFixed(2)}, ${probeY.toFixed(2)})`}
                  tone="cyan"
                />
                <MetricCard
                  label="Gradient Vector"
                  value={`(${probeStats.gx.toFixed(2)}, ${probeStats.gy.toFixed(2)})`}
                  detail={`||∇f|| = ${probeStats.gradNorm.toFixed(4)}`}
                />
                <MetricCard
                  label="Hessian Eigenvalues"
                  value={`[${probeStats.hessian.eigenvalues[0].toFixed(2)}, ${probeStats.hessian.eigenvalues[1].toFixed(2)}]`}
                  detail={`det(H) = ${probeStats.hessian.determinant.toFixed(2)}`}
                />
                <MetricCard
                  label="Local Curvature"
                  value={probeStats.hessian.definiteness}
                  detail={probeStats.hessian.classification}
                  tone={
                    probeStats.hessian.definiteness === 'Positive Definite'
                      ? 'emerald'
                      : probeStats.hessian.definiteness === 'Indefinite (Saddle)'
                      ? 'amber'
                      : 'default'
                  }
                />
              </div>
            )}

            <PlotlyChart data={plotlySpec.data} layout={plotlySpec.layout} height={440} />

            {probeStats.dim === 1 && (
              <div className="rounded-lg border border-slate-800 bg-[#0F141D]">
                <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
                  <h3 className="font-display text-sm font-semibold text-slate-100">
                    Detected Stationary Points (f&apos;(x) = 0)
                  </h3>
                  <span className="font-mono text-xs text-slate-400">
                    Interval [-{domainBound}, {domainBound}]
                  </span>
                </div>
                {stationary1D.length === 0 ? (
                  <div className="px-4 py-5 text-xs text-slate-400">
                    No stationary points detected in [-{domainBound}, {domainBound}]. For example,
                    functions like exp(x) are strictly monotonic and have no critical points.
                  </div>
                ) : (
                  <table className="w-full text-left font-mono text-xs tabular-nums">
                    <thead className="bg-[#0B0F17] text-slate-400">
                      <tr>
                        <th className="px-4 py-2.5 font-medium">x*</th>
                        <th className="px-4 py-2.5 font-medium">f(x*)</th>
                        <th className="px-4 py-2.5 font-medium">f&apos;&apos;(x*)</th>
                        <th className="px-4 py-2.5 font-medium">Classification</th>
                        <th className="px-4 py-2.5 font-medium">Inspect</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/80 text-slate-200">
                      {stationary1D.map((pt, idx) => (
                        <tr key={idx} className="hover:bg-slate-800/30">
                          <td className="px-4 py-2.5">{pt.x.toFixed(6)}</td>
                          <td className="px-4 py-2.5">{pt.fx.toFixed(6)}</td>
                          <td className="px-4 py-2.5">{pt.fSecond.toFixed(6)}</td>
                          <td className="px-4 py-2.5">
                            <span
                              className={
                                pt.classification === 'Local Minimum'
                                  ? 'text-emerald-400'
                                  : pt.classification === 'Local Maximum'
                                  ? 'text-rose-400'
                                  : 'text-amber-400'
                              }
                            >
                              {pt.classification}
                            </span>
                          </td>
                          <td className="px-4 py-2.5">
                            <button
                              type="button"
                              onClick={() => setProbeX(Number(pt.x.toFixed(2)))}
                              className="text-cyan-400 hover:text-cyan-300 underline"
                            >
                              Move Probe
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

import { useMemo, useState } from 'react';
import { MathTex } from '../components/MathTex';
import { MetricCard } from '../components/MetricCard';
import { PlotlyChart } from '../components/PlotlyChart';
import { analyzeExpression, computeFiniteDifferences1D } from '../core/mathematics';

const DIFF_PRESETS = [
  { label: 'sin(x) + x³', expr: 'sin(x) + x**3', x: 1.2 },
  { label: 'exp(x) * cos(x)', expr: 'exp(x) * cos(x)', x: 0.7 },
  { label: 'log(x² + 1)', expr: 'log(x**2 + 1)', x: 1.5 },
  { label: 'x⁴ - 2x²', expr: 'x**4 - 2*x**2', x: 0.8 },
];

export function NumericalDiffLabView() {
  const [exprInput, setExprInput] = useState('sin(x) + x**3');
  const [probeX, setProbeX] = useState(1.2);
  const [stepHExp, setStepHExp] = useState(-3); // h = 10^stepHExp

  const analysisState = useMemo(() => {
    try {
      const data = analyzeExpression(exprInput, { allowedDimensions: [1] });
      return { data, error: null as string | null };
    } catch (err) {
      return {
        data: null,
        error: err instanceof Error ? err.message : 'Enter a valid 1D expression f(x).',
      };
    }
  }, [exprInput]);

  const currentH = Math.pow(10, stepHExp);

  const singleComparison = useMemo(() => {
    if (!analysisState.data) return null;
    return computeFiniteDifferences1D(analysisState.data, probeX, currentH);
  }, [analysisState.data, probeX, currentH]);

  const sweepData = useMemo(() => {
    if (!analysisState.data) return [];
    const exps = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12];
    return exps.map((e) => {
      const h = Math.pow(10, e);
      return computeFiniteDifferences1D(analysisState.data!, probeX, h);
    });
  }, [analysisState.data, probeX]);

  const errorChartSpec = useMemo((): {
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
  } => {
    if (sweepData.length === 0) return { data: [], layout: {} };
    const hs = sweepData.map((d) => d.h);
    return {
      data: [
        {
          x: hs,
          y: sweepData.map((d) => Math.max(d.forwardError, 1e-16)),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Forward O(h)',
          line: { color: '#F59E0B', width: 2 },
        },
        {
          x: hs,
          y: sweepData.map((d) => Math.max(d.backwardError, 1e-16)),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Backward O(h)',
          line: { color: '#A78BFA', width: 2, dash: 'dot' },
        },
        {
          x: hs,
          y: sweepData.map((d) => Math.max(d.centralError, 1e-16)),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Central O(h²)',
          line: { color: '#06B6D4', width: 2.5 },
        },
      ],
      layout: {
        title: {
          text: 'Truncation vs IEEE-754 Round-Off Error Across Step Size h',
          font: { size: 13, color: '#E2E8F0' },
        },
        xaxis: {
          title: { text: 'Step Size h (log scale)' },
          type: 'log',
          autorange: 'reversed',
        },
        yaxis: {
          title: { text: 'Absolute Error |D_h f(x) - f\'(x)|' },
          type: 'log',
        },
      },
    };
  }, [sweepData]);

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      {/* Left Configuration */}
      <div className="space-y-5 xl:col-span-4">
        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5">
          <h2 className="font-display text-base font-semibold text-slate-100">
            Numerical Differentiation Laboratory
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Compare exact symbolic differentiation against forward, backward, and central finite-difference approximations.
          </p>

          <div className="mt-4 grid grid-cols-2 gap-1.5">
            {DIFF_PRESETS.map((p) => (
              <button
                key={p.expr}
                type="button"
                onClick={() => {
                  setExprInput(p.expr);
                  setProbeX(p.x);
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
              <label htmlFor="nd-expr" className="block text-xs font-medium text-slate-300">
                Single-Variable Function f(x)
              </label>
              <input
                id="nd-expr"
                type="text"
                value={exprInput}
                onChange={(e) => setExprInput(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <label htmlFor="nd-x" className="block text-xs font-medium text-slate-300">
                Evaluation Point x
              </label>
              <input
                id="nd-x"
                type="number"
                step="0.1"
                value={probeX}
                onChange={(e) => setProbeX(Number(e.target.value))}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-sm text-slate-100 focus:border-cyan-500 focus:outline-none"
              />
            </div>

            <div>
              <div className="flex items-center justify-between text-xs">
                <label htmlFor="nd-h" className="font-medium text-slate-300">
                  Step Size h = 10^{stepHExp}
                </label>
                <span className="font-mono text-cyan-300 tabular-nums">
                  {currentH.toExponential(0)}
                </span>
              </div>
              <input
                id="nd-h"
                type="range"
                min={-12}
                max={-1}
                step={1}
                value={stepHExp}
                onChange={(e) => setStepHExp(Number(e.target.value))}
                className="mt-1.5 w-full accent-cyan-500"
              />
            </div>
          </div>
        </div>

        {/* Formulas & Precision Explanation */}
        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 space-y-3">
          <h3 className="font-display text-sm font-semibold text-slate-100">
            Finite-Difference Formulas
          </h3>
          {analysisState.data && (
            <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
              <MathTex block tex={`f'(x) = ${analysisState.data.dfDxLatex}`} />
            </div>
          )}
          <div className="space-y-1.5 rounded border border-slate-800 bg-[#0B0F17] p-3 text-xs">
            <MathTex block tex="D^+_h f(x) = \frac{f(x+h)-f(x)}{h} = f'(x) + \mathcal{O}(h)" />
            <MathTex block tex="D^-_h f(x) = \frac{f(x)-f(x-h)}{h} = f'(x) + \mathcal{O}(h)" />
            <MathTex block tex="D^c_h f(x) = \frac{f(x+h)-f(x-h)}{2h} = f'(x) + \mathcal{O}(h^2)" />
          </div>
          <p className="text-xs leading-relaxed text-slate-400">
            As h decreases initially, Taylor truncation error shrinks (quadratically for central differences). Below h ≈ 10⁻⁶, subtracting nearly equal floating-point numbers causes catastrophic cancellation and round-off error grows as O(ε_mach / h).
          </p>
        </div>
      </div>

      {/* Right Stage */}
      <div className="space-y-5 xl:col-span-8">
        {analysisState.error && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 p-4 text-sm text-rose-200">
            <span className="font-mono font-semibold">✖ Input Error: </span>
            {analysisState.error}
          </div>
        )}

        {singleComparison && (
          <>
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <MetricCard
                label="Symbolic Exact f'(x)"
                value={singleComparison.exactSymbolic.toFixed(8)}
                detail={`at x = ${probeX}`}
                tone="emerald"
              />
              <MetricCard
                label="Central Difference"
                value={singleComparison.central.toFixed(8)}
                detail={`Err = ${singleComparison.centralError.toExponential(2)}`}
                tone="cyan"
              />
              <MetricCard
                label="Forward Difference"
                value={singleComparison.forward.toFixed(8)}
                detail={`Err = ${singleComparison.forwardError.toExponential(2)}`}
                tone="amber"
              />
              <MetricCard
                label="Backward Difference"
                value={singleComparison.backward.toFixed(8)}
                detail={`Err = ${singleComparison.backwardError.toExponential(2)}`}
              />
            </div>

            <PlotlyChart
              data={errorChartSpec.data}
              layout={errorChartSpec.layout}
              height={390}
            />

            <div className="rounded-lg border border-slate-800 bg-[#0F141D]">
              <div className="border-b border-slate-800 px-4 py-3">
                <h3 className="font-display text-sm font-semibold text-slate-100">
                  Step-Size Error Comparison Table
                </h3>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs tabular-nums">
                  <thead className="bg-[#0B0F17] text-slate-400">
                    <tr>
                      <th className="px-4 py-2.5 font-medium">h</th>
                      <th className="px-4 py-2.5 font-medium">Forward Err</th>
                      <th className="px-4 py-2.5 font-medium">Backward Err</th>
                      <th className="px-4 py-2.5 font-medium">Central Err</th>
                      <th className="px-4 py-2.5 font-medium">Central Value</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/80 text-slate-200">
                    {sweepData.map((row) => (
                      <tr
                        key={row.h}
                        className={
                          Math.abs(row.h - currentH) < currentH * 0.1
                            ? 'bg-cyan-500/15 text-cyan-200'
                            : 'hover:bg-slate-800/30'
                        }
                      >
                        <td className="px-4 py-2">{row.h.toExponential(0)}</td>
                        <td className="px-4 py-2">{row.forwardError.toExponential(3)}</td>
                        <td className="px-4 py-2">{row.backwardError.toExponential(3)}</td>
                        <td className="px-4 py-2 text-cyan-300">
                          {row.centralError.toExponential(3)}
                        </td>
                        <td className="px-4 py-2">{row.central.toFixed(10)}</td>
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

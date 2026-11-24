import { useMemo, useState } from 'react';
import { MathTex } from '../components/MathTex';
import { MetricCard } from '../components/MetricCard';
import { PlotlyChart } from '../components/PlotlyChart';

type TheoryTopicId =
  | 'sequences'
  | 'continuity'
  | 'gradients'
  | 'convexity'
  | 'taylor';

const TOPICS: { id: TheoryTopicId; title: string }[] = [
  { id: 'sequences', title: '1. Sequences, Limits & Contraction' },
  { id: 'continuity', title: '2. Continuity & Differentiability' },
  { id: 'gradients', title: '3. Gradients & Directional Derivatives' },
  { id: 'convexity', title: '4. Convexity & Global Optimality' },
  { id: 'taylor', title: '5. Taylor Approximation & Descent Lemma' },
];

export function TheoryLaboratoryView() {
  const [activeTopic, setActiveTopic] = useState<TheoryTopicId>('sequences');

  // Interactive controls for Sequences & Contraction
  const [seqAlpha, setSeqAlpha] = useState(0.15);
  const [seqX0, setSeqX0] = useState(8);
  const [seqEps, setSeqEps] = useState(0.05);

  // Interactive controls for Continuity / Kink
  const [kinkAlpha, setKinkAlpha] = useState(0.6);

  // Interactive controls for Taylor Approximation on sin(x) + 0.2*x^2
  const [taylorX0, setTaylorX0] = useState(1.0);

  const sequenceDemo = useMemo(() => {
    const r = 1 - 2 * seqAlpha;
    const nSteps = 30;
    const xs: number[] = [];
    const ys: number[] = [];
    let firstNWithinEps: number | null = null;
    for (let n = 0; n <= nSteps; n++) {
      const val = Math.pow(r, n) * seqX0;
      xs.push(n);
      ys.push(val);
      if (firstNWithinEps === null && Math.abs(val) < seqEps) {
        firstNWithinEps = n;
      }
    }
    const analyticalN =
      Math.abs(r) > 0 && Math.abs(r) < 1
        ? Math.max(0, Math.ceil(Math.log(seqEps / Math.abs(seqX0)) / Math.log(Math.abs(r))))
        : null;
    return { r, xs, ys, firstNWithinEps, analyticalN };
  }, [seqAlpha, seqX0, seqEps]);

  const kinkDemo = useMemo(() => {
    // Subgradient / GD on f(x) = |x| from x0 = 0.5: f'(x) = sign(x)
    const xs: number[] = [];
    const ys: number[] = [];
    let x = 0.5;
    for (let n = 0; n <= 20; n++) {
      xs.push(n);
      ys.push(x);
      const grad = x > 0 ? 1 : x < 0 ? -1 : 0;
      x = x - kinkAlpha * grad;
    }
    return { xs, ys };
  }, [kinkAlpha]);

  const taylorPlot = useMemo((): {
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
  } => {
    // f(x) = sin(x) + 0.2*x^2
    const f = (x: number) => Math.sin(x) + 0.2 * x * x;
    const fp = (x: number) => Math.cos(x) + 0.4 * x;
    const fpp = (x: number) => -Math.sin(x) + 0.4;

    const f0 = f(taylorX0);
    const fp0 = fp(taylorX0);
    const fpp0 = fpp(taylorX0);

    const xs: number[] = [];
    const yExact: number[] = [];
    const yLinear: number[] = [];
    const yQuad: number[] = [];

    for (let i = 0; i <= 160; i++) {
      const x = -2.5 + (i / 160) * 6.0;
      const dx = x - taylorX0;
      xs.push(x);
      yExact.push(f(x));
      yLinear.push(f0 + fp0 * dx);
      yQuad.push(f0 + fp0 * dx + 0.5 * fpp0 * dx * dx);
    }

    return {
      data: [
        {
          x: xs,
          y: yExact,
          type: 'scatter',
          mode: 'lines',
          name: 'Exact f(x) = sin(x) + 0.2x²',
          line: { color: '#38BDF8', width: 2.5 },
        },
        {
          x: xs,
          y: yLinear,
          type: 'scatter',
          mode: 'lines',
          name: '1st-Order Tangent',
          line: { color: '#F59E0B', width: 1.8, dash: 'dash' },
        },
        {
          x: xs,
          y: yQuad,
          type: 'scatter',
          mode: 'lines',
          name: '2nd-Order Quadratic Taylor',
          line: { color: '#10B981', width: 2, dash: 'dot' },
        },
        {
          x: [taylorX0],
          y: [f0],
          type: 'scatter',
          mode: 'markers',
          name: `Expansion Point x₀ = ${taylorX0.toFixed(1)}`,
          marker: { size: 9, color: '#FFFFFF', line: { color: '#07090E', width: 1.5 } },
        },
      ],
      layout: {
        title: {
          text: `Taylor Approximations Around x₀ = ${taylorX0.toFixed(1)}`,
          font: { size: 13, color: '#E2E8F0' },
        },
        xaxis: { title: { text: 'x' } },
        yaxis: { title: { text: 'y' }, range: [-2, 6] },
      },
    };
  }, [taylorX0]);

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      {/* Left Topic Index */}
      <div className="space-y-4 xl:col-span-4">
        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5">
          <h2 className="font-display text-base font-semibold text-slate-100">
            Real Analysis Theory Laboratory
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Formal MATH 303 foundations connecting sequences, limits, continuity, directional derivatives, convexity, and Taylor bounds to Gradient Descent.
          </p>

          <div className="mt-4 space-y-1.5">
            {TOPICS.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setActiveTopic(t.id)}
                className={`w-full rounded-md border px-3 py-2.5 text-left text-xs font-medium transition-colors ${
                  activeTopic === t.id
                    ? 'border-cyan-500/60 bg-cyan-500/12 text-cyan-300'
                    : 'border-slate-800 bg-[#0B0F17] text-slate-300 hover:border-slate-700 hover:text-slate-100'
                }`}
              >
                {t.title}
              </button>
            ))}
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 text-xs leading-relaxed text-slate-300 space-y-2">
          <h3 className="font-display text-sm font-semibold text-slate-100">
            Proofs vs. Numerical Observations
          </h3>
          <p>
            In Real Analysis, a <span className="text-cyan-300 font-medium">theorem</span> establishes a universal guarantee under explicit hypotheses (e.g., convexity + L-Lipschitz gradient).
          </p>
          <p className="text-slate-400">
            A finite computer simulation only provides a <span className="text-amber-300 font-medium">numerical observation</span> over finitely many floating-point steps. Stopping when |f&apos;(x_n)| &lt; ε does not prove a global minimum was reached unless convexity hypotheses also hold.
          </p>
        </div>
      </div>

      {/* Right Interactive Theory Stage */}
      <div className="space-y-5 xl:col-span-8">
        {activeTopic === 'sequences' && (
          <div className="space-y-5">
            <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 space-y-3">
              <h3 className="font-display text-base font-semibold text-slate-100">
                1. Sequences, ε–N Limits & Geometric Contraction
              </h3>
              <p className="text-xs leading-relaxed text-slate-300">
                Repeated Gradient Descent updates produce a sequence of real numbers (or vectors){' '}
                <span className="font-mono">(x_n)</span>. By definition,{' '}
                <span className="font-mono">lim x_n = L</span> iff for every ε &gt; 0 there exists N ∈ ℕ such that n ≥ N implies |x_n − L| &lt; ε.
              </p>
              <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
                <MathTex
                  block
                  tex="f(x)=x^2 \implies x_{n+1} = x_n - \alpha(2x_n) = (1-2\alpha)x_n \implies x_n = (1-2\alpha)^n x_0"
                />
              </div>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3 pt-2">
                <div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300">Learning Rate α</span>
                    <span className="font-mono text-cyan-300">{seqAlpha.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0.02}
                    max={1.15}
                    step={0.01}
                    value={seqAlpha}
                    onChange={(e) => setSeqAlpha(Number(e.target.value))}
                    className="mt-1.5 w-full accent-cyan-500"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300">Initial x₀</span>
                    <span className="font-mono text-cyan-300">{seqX0}</span>
                  </div>
                  <input
                    type="range"
                    min={-10}
                    max={10}
                    step={1}
                    value={seqX0}
                    onChange={(e) => setSeqX0(Number(e.target.value))}
                    className="mt-1.5 w-full accent-cyan-500"
                  />
                </div>
                <div>
                  <div className="flex justify-between text-xs">
                    <span className="text-slate-300">ε-Band Radius</span>
                    <span className="font-mono text-amber-300">{seqEps}</span>
                  </div>
                  <input
                    type="range"
                    min={0.01}
                    max={1.0}
                    step={0.01}
                    value={seqEps}
                    onChange={(e) => setSeqEps(Number(e.target.value))}
                    className="mt-1.5 w-full accent-amber-500"
                  />
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <MetricCard
                label="Contraction Ratio r = 1 - 2α"
                value={sequenceDemo.r.toFixed(3)}
                detail={Math.abs(sequenceDemo.r) < 1 ? '|r| < 1 (Contractive)' : '|r| ≥ 1 (Non-Contractive)'}
                tone={Math.abs(sequenceDemo.r) < 1 ? 'emerald' : 'rose'}
              />
              <MetricCard
                label="Formal Cutoff N(ε)"
                value={
                  sequenceDemo.analyticalN !== null
                    ? String(sequenceDemo.analyticalN)
                    : '∞ (Does not converge)'
                }
                detail={`For |x_n - 0| < ${seqEps}`}
                tone="cyan"
              />
              <MetricCard
                label="Limit Behavior"
                value={
                  Math.abs(sequenceDemo.r) < 1
                    ? sequenceDemo.r < 0
                      ? 'Oscillating → 0'
                      : 'Monotone → 0'
                    : Math.abs(sequenceDemo.r) === 1
                    ? 'Bounded Cycle'
                    : 'Diverges'
                }
                tone={Math.abs(sequenceDemo.r) < 1 ? 'emerald' : 'rose'}
              />
            </div>

            <PlotlyChart
              data={[
                {
                  x: sequenceDemo.xs,
                  y: sequenceDemo.ys,
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'Iterate x_n',
                  line: { color: '#06B6D4', width: 2 },
                },
                {
                  x: [0, 30],
                  y: [seqEps, seqEps],
                  type: 'scatter',
                  mode: 'lines',
                  name: '+ε bound',
                  line: { color: '#F59E0B', width: 1.2, dash: 'dot' },
                },
                {
                  x: [0, 30],
                  y: [-seqEps, -seqEps],
                  type: 'scatter',
                  mode: 'lines',
                  name: '-ε bound',
                  line: { color: '#F59E0B', width: 1.2, dash: 'dot' },
                },
              ]}
              layout={{
                title: {
                  text: 'Interactive ε–N Sequence Convergence Tube',
                  font: { size: 13, color: '#E2E8F0' },
                },
                xaxis: { title: { text: 'Index n' } },
                yaxis: { title: { text: 'x_n' } },
              }}
              height={360}
            />
          </div>
        )}

        {activeTopic === 'continuity' && (
          <div className="space-y-5">
            <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 space-y-3">
              <h3 className="font-display text-base font-semibold text-slate-100">
                2. Why Continuity & Differentiability Assumptions Matter
              </h3>
              <p className="text-xs leading-relaxed text-slate-300">
                Consider <span className="font-mono text-cyan-300">f(x) = |x|</span>. It is continuous everywhere on ℝ and convex, with global minimum at <span className="font-mono">x* = 0</span>. However, it is not differentiable at <span className="font-mono">x = 0</span>, and its slope magnitude |f&apos;(x)| = 1 never shrinks near zero! With any constant learning rate α, Gradient Descent oscillates permanently around 0 without converging.
              </p>
              <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
                <MathTex block tex="f(x) = |x| \implies f'(x) = \operatorname{sgn}(x) \quad (x \ne 0), \qquad x_{n+1} = x_n - \alpha \operatorname{sgn}(x_n)" />
              </div>
              <div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300">Constant Step α on f(x) = |x| (from x₀ = 0.5)</span>
                  <span className="font-mono text-amber-300">α = {kinkAlpha.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min={0.1}
                  max={1.2}
                  step={0.05}
                  value={kinkAlpha}
                  onChange={(e) => setKinkAlpha(Number(e.target.value))}
                  className="mt-1.5 w-full accent-amber-500"
                />
              </div>
            </div>

            <PlotlyChart
              data={[
                {
                  x: kinkDemo.xs,
                  y: kinkDemo.ys,
                  type: 'scatter',
                  mode: 'lines+markers',
                  name: 'x_n on f(x) = |x|',
                  line: { color: '#F59E0B', width: 2 },
                },
              ]}
              layout={{
                title: {
                  text: 'Non-Differentiable Kink Oscillation on f(x) = |x|',
                  font: { size: 13, color: '#E2E8F0' },
                },
                xaxis: { title: { text: 'Iteration n' } },
                yaxis: { title: { text: 'x_n' } },
              }}
              height={360}
            />
          </div>
        )}

        {activeTopic === 'gradients' && (
          <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-6 space-y-4">
            <h3 className="font-display text-base font-semibold text-slate-100">
              3. Partial Derivatives, Gradients & Steepest Descent Proof
            </h3>
            <p className="text-xs leading-relaxed text-slate-300">
              For a differentiable function f : ℝⁿ → ℝ and unit direction vector u (‖u‖ = 1), the directional derivative measures the instantaneous rate of change along u:
            </p>
            <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
              <MathTex block tex="D_u f(X) = \lim_{t \to 0} \frac{f(X + tu) - f(X)}{t} = \nabla f(X) \cdot u = \|\nabla f(X)\|\,\|u\|\,\cos\theta" />
            </div>
            <p className="text-xs leading-relaxed text-slate-300">
              By the Cauchy–Schwarz inequality, <span className="font-mono">-‖∇f(X)‖ ≤ D_u f(X) ≤ ‖∇f(X)‖</span>. The minimum (most negative rate of change, i.e., steepest descent) occurs uniquely when <span className="font-mono">cos θ = -1</span>, meaning:
            </p>
            <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
              <MathTex block tex="u^* = -\frac{\nabla f(X)}{\|\nabla f(X)\|} \implies X_{n+1} = X_n - \alpha \nabla f(X_n)" />
            </div>
            <p className="text-xs text-slate-400">
              Note: Steepest descent is a local property. On narrow curved valleys (high Hessian condition number κ), the negative gradient points mostly across the canyon walls rather than along the valley floor, motivating Momentum and Adam.
            </p>
          </div>
        )}

        {activeTopic === 'convexity' && (
          <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-6 space-y-4">
            <h3 className="font-display text-base font-semibold text-slate-100">
              4. Convexity, Second-Order Conditions & Global Guarantees
            </h3>
            <p className="text-xs leading-relaxed text-slate-300">
              A function f : ℝⁿ → ℝ on a convex set is convex if for all x, y and t ∈ [0, 1]:
            </p>
            <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
              <MathTex block tex="f(tx + (1-t)y) \le t f(x) + (1-t) f(y)" />
            </div>
            <p className="text-xs leading-relaxed text-slate-300">
              If f is twice continuously differentiable (C²), convexity is equivalent to its Hessian matrix being positive semidefinite everywhere:
            </p>
            <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
              <MathTex block tex="H(X) = \nabla^2 f(X) \succeq 0 \iff \lambda_i(H(X)) \ge 0 \;\; \forall i" />
            </div>
            <div className="rounded border border-slate-800 bg-[#0B0F17] p-4 text-xs leading-relaxed text-slate-300 space-y-1.5">
              <div className="font-semibold text-emerald-400">
                ● Mathematical Guarantee (Convex Case):
              </div>
              <p>
                Every stationary point (∇f(X*) = 0) of a convex function is automatically a global minimum. If f is strictly convex (λ_min &gt; 0) and L-smooth (λ_max ≤ L), Gradient Descent with 0 &lt; α &lt; 2/L converges globally.
              </p>
              <div className="pt-2 font-semibold text-amber-400">
                ▲ Non-Convex Reality (Neural Networks / Quartics):
              </div>
              <p>
                When H(X) has negative eigenvalues at some regions, ∇f(X*) = 0 only identifies a candidate stationary point—it may be a local minimum, local maximum, or saddle point.
              </p>
            </div>
          </div>
        )}

        {activeTopic === 'taylor' && (
          <div className="space-y-5">
            <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 space-y-3">
              <h3 className="font-display text-base font-semibold text-slate-100">
                5. Taylor Approximation & The Descent Lemma
              </h3>
              <p className="text-xs leading-relaxed text-slate-300">
                Gradient Descent works for small α because the first-order Taylor expansion dominates locally:
              </p>
              <div className="rounded border border-slate-800 bg-[#0B0F17] p-3">
                <MathTex block tex="f(x - \alpha f'(x)) = f(x) - \alpha |f'(x)|^2 + \frac{\alpha^2}{2} f''(\xi) |f'(x)|^2" />
              </div>
              <p className="text-xs text-slate-400">
                If f&apos;&apos;(ξ) ≤ L (L-smoothness), then choosing α &lt; 2/L guarantees the linear descent term exceeds the quadratic curvature term, ensuring f(x_{'{n+1}'}) &lt; f(x_n).
              </p>
              <div>
                <div className="flex justify-between text-xs">
                  <span className="text-slate-300">Taylor Expansion Point x₀</span>
                  <span className="font-mono text-cyan-300">x₀ = {taylorX0.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min={-2.0}
                  max={2.5}
                  step={0.1}
                  value={taylorX0}
                  onChange={(e) => setTaylorX0(Number(e.target.value))}
                  className="mt-1.5 w-full accent-cyan-500"
                />
              </div>
            </div>

            <PlotlyChart data={taylorPlot.data} layout={taylorPlot.layout} height={380} />
          </div>
        )}
      </div>
    </div>
  );
}

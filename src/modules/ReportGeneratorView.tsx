import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import { Download, FileText } from 'lucide-react';
import Plotly from 'plotly.js-dist-min';
import { useMemo, useState } from 'react';
import { MathTex } from '../components/MathTex';
import { MetricCard } from '../components/MetricCard';
import { PlotlyChart } from '../components/PlotlyChart';
import { analyzeExpression } from '../core/mathematics';
import {
  OptimizerAlgorithm,
  runGradientDescent1D,
  runGradientDescent2D,
} from '../core/optimizers';

export function ReportGeneratorView() {
  const [projectName, setProjectName] = useState(
    'GradAI — MATH 303 Gradient Descent Analysis'
  );
  const [authorName, setAuthorName] = useState('BS Mathematics Student');
  const [exprInput, setExprInput] = useState('x**2');
  const [x0Input, setX0Input] = useState('10');
  const [y0Input, setY0Input] = useState('0');
  const [alphaInput, setAlphaInput] = useState('0.1');
  const [maxIterInput, setMaxIterInput] = useState('150');
  const [tolInput, setTolInput] = useState('1e-6');
  const [algorithm, setAlgorithm] = useState<OptimizerAlgorithm>('gd');
  const [observations, setObservations] = useState(
    'Iterates follow x(n+1) = x(n) - alpha * f\'(x(n)). Convergence depends on smoothness, local curvature, and selecting a step size within the stability bound.'
  );
  const [limitations, setLimitations] = useState(
    'Floating-point IEEE-754 precision limits accuracy near 1e-16. On non-convex objectives, vanishing gradients may indicate a saddle point or local minimum rather than a global optimum.'
  );

  const reportRun = useMemo(() => {
    try {
      const analysis = analyzeExpression(exprInput);
      const x0 = Number(x0Input);
      const y0 = Number(y0Input);
      const alpha = Number(alphaInput);
      const maxIter = Number(maxIterInput);
      const tol = Number(tolInput);

      if (analysis.dimension === 1) {
        const res1D = runGradientDescent1D(exprInput, x0, alpha, maxIter, tol);
        return { analysis, dim: 1 as const, res1D, res2D: null, error: null as string | null };
      }
      const res2D = runGradientDescent2D(
        exprInput,
        x0,
        y0,
        alpha,
        maxIter,
        tol,
        algorithm
      );
      return { analysis, dim: 2 as const, res1D: null, res2D, error: null as string | null };
    } catch (err) {
      return {
        analysis: null,
        dim: 1 as const,
        res1D: null,
        res2D: null,
        error: err instanceof Error ? err.message : 'Invalid report configuration.',
      };
    }
  }, [exprInput, x0Input, y0Input, alphaInput, maxIterInput, tolInput, algorithm]);

  const activeResult = reportRun.res1D ?? reportRun.res2D;

  const previewChartSpec = useMemo((): {
    data: Plotly.Data[];
    layout: Partial<Plotly.Layout>;
  } => {
    if (!activeResult) return { data: [], layout: {} };
    if (reportRun.dim === 1 && reportRun.res1D) {
      return {
        data: [
          {
            x: reportRun.res1D.history.map((h) => h.iteration),
            y: reportRun.res1D.history.map((h) => h.fx),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'f(x_n)',
            line: { color: '#06B6D4', width: 2 },
          },
          {
            x: reportRun.res1D.history.map((h) => h.iteration),
            y: reportRun.res1D.history.map((h) => h.x),
            type: 'scatter',
            mode: 'lines',
            name: 'x_n',
            yaxis: 'y2',
            line: { color: '#F59E0B', width: 1.8, dash: 'dot' },
          },
        ],
        layout: {
          title: {
            text: 'Report Convergence Preview: f(x_n) & x_n',
            font: { size: 13, color: '#E2E8F0' },
          },
          xaxis: { title: { text: 'Iteration n' } },
          yaxis: { title: { text: 'f(x_n)' } },
          yaxis2: {
            title: { text: 'x_n' },
            overlaying: 'y',
            side: 'right',
            gridcolor: 'rgba(0,0,0,0)',
          },
        },
      };
    }

    if (reportRun.dim === 2 && reportRun.res2D) {
      return {
        data: [
          {
            x: reportRun.res2D.history.map((h) => h.iteration),
            y: reportRun.res2D.history.map((h) => h.fxy),
            type: 'scatter',
            mode: 'lines+markers',
            name: 'f(x_n, y_n)',
            line: { color: '#06B6D4', width: 2 },
          },
          {
            x: reportRun.res2D.history.map((h) => h.iteration),
            y: reportRun.res2D.history.map((h) => Math.max(h.gradNorm, 1e-16)),
            type: 'scatter',
            mode: 'lines',
            name: '||∇f||',
            yaxis: 'y2',
            line: { color: '#F59E0B', width: 1.8, dash: 'dot' },
          },
        ],
        layout: {
          title: {
            text: `Report Convergence Preview (${reportRun.res2D.algorithm.toUpperCase()})`,
            font: { size: 13, color: '#E2E8F0' },
          },
          xaxis: { title: { text: 'Iteration n' } },
          yaxis: { title: { text: 'f(X_n)' } },
          yaxis2: {
            title: { text: '||∇f||' },
            overlaying: 'y',
            side: 'right',
            type: 'log',
            gridcolor: 'rgba(0,0,0,0)',
          },
        },
      };
    }

    return { data: [], layout: {} };
  }, [activeResult, reportRun.dim, reportRun.res1D, reportRun.res2D]);

  function handleExportCsv() {
    if (!activeResult) return;
    let csv = '';
    if (reportRun.dim === 1 && reportRun.res1D) {
      csv = [
        'iteration,x,fx,grad,abs_grad,step_delta',
        ...reportRun.res1D.history.map((h) =>
          [h.iteration, h.x, h.fx, h.grad, h.absGrad, h.stepDelta].join(',')
        ),
      ].join('\n');
    } else if (reportRun.dim === 2 && reportRun.res2D) {
      csv = [
        'iteration,x,y,fxy,grad_x,grad_y,grad_norm,step_delta',
        ...reportRun.res2D.history.map((h) =>
          [
            h.iteration,
            h.x,
            h.y,
            h.fxy,
            h.gradX,
            h.gradY,
            h.gradNorm,
            h.stepDelta,
          ].join(',')
        ),
      ].join('\n');
    }
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'gradai_iteration_history.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  async function handleExportPdf() {
    if (!activeResult || !reportRun.analysis) return;

    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const pageW = doc.internal.pageSize.getWidth();

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(16);
    doc.text(projectName || 'GradAI Mathematical Report', 14, 18);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(10);
    doc.setTextColor(90);
    doc.text(
      `Author: ${authorName}  |  Generated: ${new Date().toISOString().slice(0, 19).replace('T', ' ')}`,
      14,
      25
    );

    doc.setDrawColor(200);
    doc.line(14, 28, pageW - 14, 28);

    doc.setTextColor(20);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.text('1. Mathematical Formulation & Parameters', 14, 36);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9.5);
    const derivLine =
      reportRun.dim === 1
        ? `f'(x) = ${reportRun.analysis.dfDxStr}`
        : `grad f(x, y) = [${reportRun.analysis.dfDxStr}, ${reportRun.analysis.dfDyStr}]`;
    const initLine =
      reportRun.dim === 1
        ? `x0 = ${x0Input}`
        : `(x0, y0) = (${x0Input}, ${y0Input}), Optimizer = ${algorithm.toUpperCase()}`;

    const summaryLines = [
      `Objective Function: f = ${reportRun.analysis.normalized}`,
      `Symbolic Derivative / Gradient: ${derivLine}`,
      `Configuration: ${initLine}, alpha = ${alphaInput}, maxIter = ${maxIterInput}, tol = ${tolInput}`,
      `Status: ${activeResult.statusLabel} (${activeResult.totalIterations} iterations)`,
      reportRun.dim === 1 && reportRun.res1D
        ? `Final Solution: x* = ${reportRun.res1D.finalX.toExponential(6)}, f(x*) = ${reportRun.res1D.finalFx.toExponential(6)}, |f'(x*)| = ${Math.abs(reportRun.res1D.finalGrad).toExponential(4)}`
        : reportRun.res2D
        ? `Final Solution: (x*, y*) = (${reportRun.res2D.finalX.toExponential(5)}, ${reportRun.res2D.finalY.toExponential(5)}), f(X*) = ${reportRun.res2D.finalFxy.toExponential(6)}, ||grad|| = ${reportRun.res2D.finalGradNorm.toExponential(4)}`
        : '',
    ];
    doc.text(summaryLines, 14, 43, { maxWidth: pageW - 28 });

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.text('2. Mathematical Interpretation & Observations', 14, 72);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9.5);
    const notesText = `Automated Diagnosis: ${activeResult.explanation}\n\nObservations: ${observations}\n\nLimitations: ${limitations}`;
    const wrappedNotes = doc.splitTextToSize(notesText, pageW - 28) as string[];
    doc.text(wrappedNotes, 14, 78);

    const pageH = doc.internal.pageSize.getHeight();
    let yCursor = 78 + wrappedNotes.length * 4.4 + 8;
    let tableStartY = yCursor + 6;
    const chartEl = document.getElementById('gradai-report-chart');
    if (chartEl) {
      try {
        const imgUrl = await Plotly.toImage(chartEl, {
          format: 'png',
          width: 900,
          height: 380,
        });
        if (yCursor + 86 > pageH - 20) {
          doc.addPage();
          yCursor = 18;
        }
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(11);
        doc.text('3. Convergence Visualization', 14, yCursor);
        doc.addImage(imgUrl, 'PNG', 14, yCursor + 4, pageW - 28, 72);
        yCursor += 84;
        if (yCursor + 20 > pageH - 20) {
          doc.addPage();
          yCursor = 18;
        }
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(11);
        doc.text('4. Iteration History (Sampled up to 40 Rows)', 14, yCursor);
        tableStartY = yCursor + 4;
      } catch {
        if (yCursor + 20 > pageH - 20) {
          doc.addPage();
          yCursor = 18;
        }
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(11);
        doc.text('3. Iteration History (Sampled up to 40 Rows)', 14, yCursor);
        tableStartY = yCursor + 4;
      }
    } else {
      if (yCursor + 20 > pageH - 20) {
        doc.addPage();
        yCursor = 18;
      }
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(11);
      doc.text('3. Iteration History (Sampled up to 40 Rows)', 14, yCursor);
      tableStartY = yCursor + 4;
    }

    if (reportRun.dim === 1 && reportRun.res1D) {
      const hist = reportRun.res1D.history;
      const sampled = hist.filter(
        (_, i) => i < 15 || i === hist.length - 1 || i % Math.max(1, Math.floor(hist.length / 25)) === 0
      );
      autoTable(doc, {
        startY: tableStartY,
        head: [['n', 'x_n', 'f(x_n)', "f'(x_n)", 'Step Delta']],
        body: sampled.slice(0, 40).map((r) => [
          r.iteration,
          r.x.toFixed(6),
          r.fx.toExponential(4),
          r.grad.toExponential(4),
          r.stepDelta.toExponential(3),
        ]),
        styles: { fontSize: 8, font: 'courier' },
        headStyles: { fillColor: [15, 23, 42] },
      });
    } else if (reportRun.res2D) {
      const hist = reportRun.res2D.history;
      const sampled = hist.filter(
        (_, i) => i < 15 || i === hist.length - 1 || i % Math.max(1, Math.floor(hist.length / 25)) === 0
      );
      autoTable(doc, {
        startY: tableStartY,
        head: [['n', 'x_n', 'y_n', 'f(X_n)', '||grad||', 'Step Delta']],
        body: sampled.slice(0, 40).map((r) => [
          r.iteration,
          r.x.toFixed(5),
          r.y.toFixed(5),
          r.fxy.toExponential(4),
          r.gradNorm.toExponential(4),
          r.stepDelta.toExponential(3),
        ]),
        styles: { fontSize: 8, font: 'courier' },
        headStyles: { fillColor: [15, 23, 42] },
      });
    }

    doc.save('gradai_mathematical_report.pdf');
  }

  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-12">
      {/* Left Report Builder */}
      <div className="space-y-5 xl:col-span-5">
        <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 space-y-3">
          <h2 className="font-display text-base font-semibold text-slate-100">
            Module 6 · Mathematical Report Generator
          </h2>
          <p className="text-xs text-slate-400">
            Compile a downloadable PDF laboratory report and CSV iteration log containing symbolic derivatives, optimization metrics, interpretation, and limitations.
          </p>

          <div>
            <label htmlFor="rep-proj" className="block text-xs font-medium text-slate-300">
              Project / Report Title
            </label>
            <input
              id="rep-proj"
              type="text"
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 text-xs text-slate-100"
            />
          </div>

          <div>
            <label htmlFor="rep-author" className="block text-xs font-medium text-slate-300">
              Author / Course Context
            </label>
            <input
              id="rep-author"
              type="text"
              value={authorName}
              onChange={(e) => setAuthorName(e.target.value)}
              className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 text-xs text-slate-100"
            />
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="col-span-2">
              <label htmlFor="rep-expr" className="block text-xs font-medium text-slate-300">
                Objective Function
              </label>
              <input
                id="rep-expr"
                type="text"
                value={exprInput}
                onChange={(e) => setExprInput(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 font-mono text-xs text-slate-100"
              />
            </div>

            <div>
              <label htmlFor="rep-x0" className="block text-xs font-medium text-slate-300">
                Initial x₀
              </label>
              <input
                id="rep-x0"
                type="number"
                step="any"
                value={x0Input}
                onChange={(e) => setX0Input(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-1.5 font-mono text-xs text-slate-100"
              />
            </div>

            {reportRun.dim === 2 && (
              <div>
                <label htmlFor="rep-y0" className="block text-xs font-medium text-slate-300">
                  Initial y₀
                </label>
                <input
                  id="rep-y0"
                  type="number"
                  step="any"
                  value={y0Input}
                  onChange={(e) => setY0Input(e.target.value)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-1.5 font-mono text-xs text-slate-100"
                />
              </div>
            )}

            <div>
              <label htmlFor="rep-alpha" className="block text-xs font-medium text-slate-300">
                Learning Rate α
              </label>
              <input
                id="rep-alpha"
                type="number"
                step="0.01"
                value={alphaInput}
                onChange={(e) => setAlphaInput(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-1.5 font-mono text-xs text-slate-100"
              />
            </div>

            <div>
              <label htmlFor="rep-iter" className="block text-xs font-medium text-slate-300">
                Max Iterations
              </label>
              <input
                id="rep-iter"
                type="number"
                value={maxIterInput}
                onChange={(e) => setMaxIterInput(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-1.5 font-mono text-xs text-slate-100"
              />
            </div>

            <div>
              <label htmlFor="rep-tol" className="block text-xs font-medium text-slate-300">
                Tolerance ε
              </label>
              <input
                id="rep-tol"
                type="text"
                value={tolInput}
                onChange={(e) => setTolInput(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-1.5 font-mono text-xs text-slate-100"
              />
            </div>

            {reportRun.dim === 2 && (
              <div>
                <label htmlFor="rep-algo" className="block text-xs font-medium text-slate-300">
                  Optimizer
                </label>
                <select
                  id="rep-algo"
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value as OptimizerAlgorithm)}
                  className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-2.5 py-1.5 font-mono text-xs text-slate-100"
                >
                  <option value="gd">Batch GD</option>
                  <option value="momentum">Momentum</option>
                  <option value="rmsprop">RMSProp</option>
                  <option value="adam">Adam</option>
                </select>
              </div>
            )}
          </div>

          <div>
            <label htmlFor="rep-obs" className="block text-xs font-medium text-slate-300">
              Mathematical Observations
            </label>
            <textarea
              id="rep-obs"
              rows={3}
              value={observations}
              onChange={(e) => setObservations(e.target.value)}
              className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 text-xs text-slate-100"
            />
          </div>

          <div>
            <label htmlFor="rep-lim" className="block text-xs font-medium text-slate-300">
              Numerical & Theoretical Limitations
            </label>
            <textarea
              id="rep-lim"
              rows={2}
              value={limitations}
              onChange={(e) => setLimitations(e.target.value)}
              className="mt-1 w-full rounded border border-slate-700 bg-[#07090E] px-3 py-2 text-xs text-slate-100"
            />
          </div>

          <div className="flex flex-wrap gap-2 pt-2">
            <button
              type="button"
              onClick={handleExportPdf}
              disabled={!activeResult}
              className="flex flex-1 items-center justify-center gap-2 rounded-md bg-cyan-500 px-4 py-2.5 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-40 whitespace-nowrap"
            >
              <FileText className="h-4 w-4" />
              Download PDF Report
            </button>
            <button
              type="button"
              onClick={handleExportCsv}
              disabled={!activeResult}
              className="flex items-center gap-2 rounded-md border border-slate-700 bg-[#0B0F17] px-4 py-2.5 text-xs font-medium text-slate-200 hover:border-cyan-500/50 disabled:opacity-40 whitespace-nowrap"
            >
              <Download className="h-4 w-4" />
              Export CSV
            </button>
          </div>
        </div>
      </div>

      {/* Right Preview */}
      <div className="space-y-5 xl:col-span-7">
        {reportRun.error && (
          <div className="rounded-lg border border-rose-500/40 bg-rose-950/30 p-4 text-sm text-rose-200">
            <span className="font-mono font-semibold">✖ Error: </span>
            {reportRun.error}
          </div>
        )}

        {activeResult && reportRun.analysis && (
          <>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
              <MetricCard
                label="Convergence Status"
                value={activeResult.statusLabel}
                tone={
                  activeResult.status.startsWith('converged')
                    ? 'emerald'
                    : activeResult.status === 'diverged'
                    ? 'rose'
                    : 'amber'
                }
              />
              <MetricCard
                label="Iterations Logged"
                value={String(activeResult.totalIterations)}
                detail={`α = ${activeResult.alpha}`}
                tone="cyan"
              />
              <MetricCard
                label="Final Objective"
                value={
                  reportRun.dim === 1 && reportRun.res1D
                    ? reportRun.res1D.finalFx.toExponential(3)
                    : reportRun.res2D
                    ? reportRun.res2D.finalFxy.toExponential(3)
                    : '—'
                }
              />
            </div>

            <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 space-y-2">
              <div className="text-xs text-slate-400">Symbolic Expressions Included in Report</div>
              <MathTex
                block
                tex={
                  reportRun.dim === 1
                    ? `f(x) = ${reportRun.analysis.latexF}, \\quad f'(x) = ${reportRun.analysis.dfDxLatex}`
                    : `f(x,y) = ${reportRun.analysis.latexF}, \\quad \\nabla f = \\begin{bmatrix} ${reportRun.analysis.dfDxLatex} \\\\ ${reportRun.analysis.dfDyLatex} \\end{bmatrix}`
                }
              />
            </div>

            <PlotlyChart
              chartId="gradai-report-chart"
              data={previewChartSpec.data}
              layout={previewChartSpec.layout}
              height={360}
            />
          </>
        )}
      </div>
    </div>
  );
}

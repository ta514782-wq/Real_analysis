interface StageItem {
  stage: string;
  title: string;
  status: 'Completed' | 'Queued';
  modules: string[];
  deliverables: string;
}

const STAGES: StageItem[] = [
  {
    stage: 'Stage 1',
    title: 'Core Mathematical Engine, Optimizers & Unit Tests',
    status: 'Completed',
    modules: ['core/validation.ts', 'core/mathematics.ts', 'core/optimizers.ts', 'tests/test_optimizers.ts'],
    deliverables:
      'Safe AST expression parser (no eval), symbolic derivatives & Hessian eigenvalues, finite differences, 1D/2D Gradient Descent, Momentum, RMSProp, Adam, and Test A/B/C verification suite.',
  },
  {
    stage: 'Stage 2',
    title: 'Module 1 (Function Explorer) & Module 2 (Gradient Descent Simulator)',
    status: 'Completed',
    modules: ['FunctionExplorerView.tsx', 'GradientDescentSimulatorView.tsx'],
    deliverables:
      'Interactive 2D Plotly derivative/tangent exploration, stationary point classification, step-by-step playback controls (start/play/step/reset/rerun), convergence curves, and iterate trajectories.',
  },
  {
    stage: 'Stage 3',
    title: 'Module 3 (3D Multivariable Lab) & Hessian / Convexity Analyzer',
    status: 'Completed',
    modules: ['Multivariable3DLabView.tsx', 'HessianAnalyzerView.tsx'],
    deliverables:
      'Interactive 3D surface + 2D contour plots, gradient vector fields, multi-path optimizer comparison (GD vs Momentum vs RMSProp vs Adam), and Hessian eigenvalue convexity classification.',
  },
  {
    stage: 'Stage 4',
    title: 'Module 5 (Convergence & Learning Rate Analyzer), Numerical Diff Lab & Experiment Manager',
    status: 'Completed',
    modules: ['ConvergenceAnalyzerView.tsx', 'NumericalDiffLabView.tsx', 'ExperimentManagerView.tsx'],
    deliverables:
      'Side-by-side alpha regimes (convergence, slow convergence, oscillation, divergence), finite-difference h-step error plots, and local experiment storage with CSV export.',
  },
  {
    stage: 'Stage 5',
    title: 'Module 4 (Machine Learning Laboratory & Advanced Training Analysis)',
    status: 'Completed',
    modules: ['core/mlEngine.ts', 'MachineLearningLabView.tsx'],
    deliverables:
      'Linear & Polynomial Regression from scratch (Batch GD, SGD, Mini-Batch, Momentum, RMSProp, Adam), CSV dataset upload, train/validation loss curves, overfitting/underfitting demos, MAE/MSE/R^2, and OLS benchmark.',
  },
  {
    stage: 'Stage 6',
    title: 'Real Analysis Theory Lab, Assessment, Module 6 (PDF/CSV Report) & Final QA',
    status: 'Completed',
    modules: ['TheoryLaboratoryView.tsx', 'InteractiveAssessmentView.tsx', 'ReportGeneratorView.tsx'],
    deliverables:
      'Formal MATH 303 proofs vs numerical observations, topic-wise quiz with hints & scoring, downloadable PDF reports with iteration tables/observations, CSV export, and 18/18 verification tests passing.',
  },
];

export function StageRoadmapView() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="font-display text-lg font-semibold text-slate-100">
          GradAI Staged Implementation Plan
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          Complete architectural roadmap mapping all core and advanced modules from the MATH 303 project brief.
        </p>
      </div>

      <div className="space-y-3">
        {STAGES.map((item) => {
          const done = item.status === 'Completed';
          return (
            <div
              key={item.stage}
              className="rounded-lg border border-slate-800 bg-[#0F141D] p-5"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-3">
                  <span className="font-mono text-xs font-semibold text-cyan-400">
                    {item.stage}
                  </span>
                  <span className="text-slate-600" aria-hidden="true">
                    ·
                  </span>
                  <h3 className="font-display text-sm font-semibold text-slate-100">
                    {item.title}
                  </h3>
                </div>
                <span
                  className={`font-mono text-xs ${
                    done ? 'text-emerald-400' : 'text-slate-400'
                  }`}
                >
                  {done ? '● Completed & Verified' : '○ Queued'}
                </span>
              </div>
              <p className="mt-2 text-xs leading-relaxed text-slate-300">
                {item.deliverables}
              </p>
              <div className="mt-2 font-mono text-xs text-slate-500">
                Files: {item.modules.join(' · ')}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

# GradAI — Interactive Gradient Descent & Real Analysis Laboratory

**Academic Context:** MATH 303 – Real Analysis I (BS Mathematics)

GradAI is an interactive mathematical and machine-learning optimization laboratory connecting Real Analysis concepts—derivatives, sequences, limits, convergence/divergence, continuity, multivariable gradients, convexity, and Taylor approximations—directly to numerical optimization and regression training from scratch.

## Features
- **Module 1 — Function Explorer:** Safe AST expression validation (no `eval()`), symbolic derivatives and Hessians, tangent line probing, stationary point detection, and 1D/2D Plotly visualization.
- **Module 2 — Gradient Descent Simulator:** Step-by-step playback controls (start, play/pause, step forward/back, rerun, reset), configurable stopping criteria, convergence curves, and iteration history tables.
- **Module 3 — 3D Multivariable Lab & Hessian Analyzer:** Interactive 3D surfaces, 2D contours with descent vector fields ($-\nabla f$), multi-optimizer trajectory comparison (Batch GD, Momentum, RMSProp, Adam), $2\times 2$ Hessian eigenvalue definiteness classification, and grid convexity scans.
- **Module 4 — Machine Learning Laboratory:** Linear and polynomial regression implemented from scratch using typed arrays (Batch GD, SGD, Mini-Batch GD, Momentum, RMSProp, Adam), CSV dataset upload, Z-score feature standardization (computed on training split only), train/validation loss curves, MAE/MSE/$R^2$ metrics, underfitting/overfitting demonstrations, and exact OLS benchmark comparison.
- **Module 5 — Convergence & Numerical Differentiation Labs:** Multi-$\alpha$ stability regime comparison (stable, slow, oscillatory, divergent), tolerance sensitivity analysis, finite-difference truncation vs. IEEE-754 round-off error curves, and local Experiment Manager with CSV export.
- **Module 6 — Real Analysis Theory Lab, Assessment & PDF/CSV Reports:** Interactive proofs vs. numerical observations, topic-wise assessment with hints, downloadable PDF reports with iteration tables, and a live 11-test Mathematical Verification Center.

## Project Structure
- `src/core/validation.ts` — AST whitelist expression & parameter validation
- `src/core/mathematics.ts` — Symbolic differentiation, finite differences, stationary/critical point solvers, Hessian & convexity scans
- `src/core/optimizers.ts` — 1D/2D Gradient Descent, Momentum, RMSProp, Adam from scratch
- `src/core/mlEngine.ts` — Regression training from scratch, CSV parser, OLS Gaussian elimination benchmark
- `src/modules/` — Modular interactive laboratory views
- `src/tests/test_optimizers.ts` — Live automated mathematical verification suite

## Installation, Testing & Running
```bash
npm install
npm run lint
npm run build
npm run dev
```

Run the mathematical verification suite from the CLI:
```bash
npx tsx -e "import { runAllVerificationTests } from './src/tests/test_optimizers'; console.log(JSON.stringify(runAllVerificationTests(), null, 2));"
```

## Known Limitations
- Symbolic calculus and safe expression parsing currently support 1D ($x$) and 2D ($x, y$) real-valued elementary expressions.
- Floating-point arithmetic uses IEEE-754 double precision (`number`), so finite differences below $h \approx 10^{-8}$ exhibit subtractive cancellation round-off error.
- Experiment persistence uses browser `localStorage` rather than an external cloud database.

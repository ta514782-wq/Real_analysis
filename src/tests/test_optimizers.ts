import { jsPDF } from 'jspdf';
import autoTable from 'jspdf-autotable';
import {
  analyzeExpression,
  analyzeHessian2D,
  computeFiniteDifferences1D,
  findCriticalPoints2D,
  findStationaryPoints1D,
  scanGridConvexity2D,
} from '../core/mathematics';
import {
  getBuiltinDatasets,
  parseCsvDataset,
  trainLinearRegressionFromScratch,
} from '../core/mlEngine';
import { runGradientDescent1D, runGradientDescent2D } from '../core/optimizers';
import { validateExpression, validateOptimizerParams } from '../core/validation';

export interface VerificationTestCase {
  id: string;
  name: string;
  category:
    | 'Convergence'
    | 'Divergence'
    | 'Multivariable'
    | 'Validation'
    | 'Differentiation'
    | 'Hessian'
    | 'Machine Learning';
  passed: boolean;
  expected: string;
  actualOutput: string;
  durationMs: number;
}

/**
 * Executes genuine mathematical unit tests and returns real computed outputs.
 * Never fabricates test results.
 */
export function runAllVerificationTests(): VerificationTestCase[] {
  const results: VerificationTestCase[] = [];

  // Test A: f(x) = x^2, x0 = 10, alpha = 0.1 -> rapid convergence to 0
  {
    const t0 = performance.now();
    const res = runGradientDescent1D('x^2', 10, 0.1, 200, 1e-6);
    const passed =
      (res.status === 'converged_grad' || res.status === 'converged_step') &&
      Math.abs(res.finalX) < 1e-5 &&
      res.finalFx < 1e-10;
    results.push({
      id: 'TEST-A',
      name: 'Test A: f(x) = x^2, x0 = 10, alpha = 0.1',
      category: 'Convergence',
      passed,
      expected: 'Converge toward x* = 0 with |x| < 1e-5',
      actualOutput: `status=${res.status}, iters=${res.totalIterations}, x_final=${res.finalX.toExponential(4)}, f(x_final)=${res.finalFx.toExponential(4)}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test B: f(x) = x^2, x0 = 10, alpha = 0.01 -> slower convergence than Test A
  {
    const t0 = performance.now();
    const resA = runGradientDescent1D('x^2', 10, 0.1, 1000, 1e-6);
    const resB = runGradientDescent1D('x^2', 10, 0.01, 1500, 1e-6);
    const passed =
      (resB.status === 'converged_grad' || resB.status === 'converged_step') &&
      resB.totalIterations > resA.totalIterations &&
      Math.abs(resB.finalX) < 1e-5;
    results.push({
      id: 'TEST-B',
      name: 'Test B: f(x) = x^2, x0 = 10, alpha = 0.01',
      category: 'Convergence',
      passed,
      expected: 'Slower convergence toward 0 (more iterations than Test A)',
      actualOutput: `status=${resB.status}, iters=${resB.totalIterations} (vs Test A: ${resA.totalIterations}), x_final=${resB.finalX.toExponential(4)}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test C: f(x) = x^2, x0 = 10, alpha = 1.1 -> oscillatory divergence
  {
    const t0 = performance.now();
    const resC = runGradientDescent1D('x^2', 10, 1.1, 100, 1e-6);
    const step1 = resC.history[1]?.x;
    const passed =
      resC.status === 'diverged' &&
      step1 !== undefined &&
      Math.abs(step1 - -12) < 1e-9;
    results.push({
      id: 'TEST-C',
      name: 'Test C: f(x) = x^2, x0 = 10, alpha = 1.1',
      category: 'Divergence',
      passed,
      expected: 'Oscillatory divergence: x_1 = -12, x_n -> +-infinity',
      actualOutput: `status=${resC.status}, x_1=${step1?.toFixed(2)}, x_last=${resC.finalX.toExponential(3)} after ${resC.totalIterations} steps`,
      durationMs: performance.now() - t0,
    });
  }

  // Test D: Multivariable f(x, y) = x^2 + 2y^2
  {
    const t0 = performance.now();
    const res = runGradientDescent2D('x**2 + 2*y**2', 4, -3, 0.1, 200, 1e-6, 'gd');
    const passed =
      res.status === 'converged_grad' &&
      Math.hypot(res.finalX, res.finalY) < 1e-5;
    results.push({
      id: 'TEST-MV',
      name: 'Multivariable GD: f(x,y) = x^2 + 2y^2 from (4, -3)',
      category: 'Multivariable',
      passed,
      expected: 'Converge to (0, 0) with ||grad|| <= 1e-6',
      actualOutput: `status=${res.status}, iters=${res.totalIterations}, final=(${res.finalX.toExponential(3)}, ${res.finalY.toExponential(3)}), ||grad||=${res.finalGradNorm.toExponential(3)}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test E: Optimizer comparison (Adam vs GD on anisotropic valley x^2 + 10*y^2)
  {
    const t0 = performance.now();
    const resGD = runGradientDescent2D('x^2 + 10*y^2', 5, 2, 0.05, 300, 1e-5, 'gd');
    const resAdam = runGradientDescent2D('x^2 + 10*y^2', 5, 2, 0.2, 300, 1e-5, 'adam');
    const passed =
      resGD.status === 'converged_grad' &&
      resAdam.status === 'converged_grad' &&
      Math.hypot(resAdam.finalX, resAdam.finalY) < 1e-4;
    results.push({
      id: 'TEST-OPT',
      name: 'Adam & GD on Anisotropic Valley x^2 + 10y^2',
      category: 'Multivariable',
      passed,
      expected: 'Both GD and Adam converge to (0, 0)',
      actualOutput: `GD iters=${resGD.totalIterations} (f=${resGD.finalFxy.toExponential(2)}), Adam iters=${resAdam.totalIterations} (f=${resAdam.finalFxy.toExponential(2)})`,
      durationMs: performance.now() - t0,
    });
  }

  // Test F: Validation safeguards (invalid expressions, alpha <= 0)
  {
    const t0 = performance.now();
    const emptyCheck = validateExpression('');
    const unsafeCheck = validateExpression('alert(1)');
    const unknownVarCheck = validateExpression('z^2');
    const zeroAlpha = validateOptimizerParams({
      alpha: 0,
      maxIterations: 50,
      tolerance: 1e-6,
      x0: 1,
    });
    const negAlpha = validateOptimizerParams({
      alpha: -0.25,
      maxIterations: 50,
      tolerance: 1e-6,
      x0: 1,
    });
    const passed =
      !emptyCheck.valid &&
      !unsafeCheck.valid &&
      !unknownVarCheck.valid &&
      !zeroAlpha.valid &&
      !negAlpha.valid;
    results.push({
      id: 'TEST-VAL',
      name: 'Input Validation: Malformed Expressions & Non-Positive Alpha',
      category: 'Validation',
      passed,
      expected: 'Reject empty/unsafe expressions and alpha <= 0',
      actualOutput: `empty=${!emptyCheck.valid}, unsafe=${!unsafeCheck.valid}, unknownSym=${!unknownVarCheck.valid}, alpha=0 rejected=${!zeroAlpha.valid}, alpha<0 rejected=${!negAlpha.valid}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test G: Symbolic vs Central Finite Difference Accuracy
  {
    const t0 = performance.now();
    const analysis = analyzeExpression('sin(x) + x^3');
    const diff = computeFiniteDifferences1D(analysis, 1.2, 1e-5);
    const exactExpected = Math.cos(1.2) + 3 * 1.2 * 1.2;
    const symErr = Math.abs(diff.exactSymbolic - exactExpected);
    const passed = symErr < 1e-12 && diff.centralError < 1e-9;
    results.push({
      id: 'TEST-DIFF',
      name: 'Symbolic & Central Finite Difference of sin(x) + x^3 at x=1.2',
      category: 'Differentiation',
      passed,
      expected: 'Symbolic exact (< 1e-12 error), Central FD error < 1e-9 (h=1e-5)',
      actualOutput: `symbolic=${diff.exactSymbolic.toFixed(10)}, central=${diff.central.toFixed(10)}, centralErr=${diff.centralError.toExponential(3)}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test H: Hessian Eigenvalues & Saddle Point Classification
  {
    const t0 = performance.now();
    const saddleAnalysis = analyzeExpression('x^2 - y^2');
    const hessian = analyzeHessian2D(saddleAnalysis, 0, 0);
    const passed =
      Math.abs(hessian.eigenvalues[0] - 2) < 1e-9 &&
      Math.abs(hessian.eigenvalues[1] - -2) < 1e-9 &&
      hessian.definiteness === 'Indefinite (Saddle)';
    results.push({
      id: 'TEST-HESS',
      name: 'Hessian Eigenvalues & Saddle Classification for f(x,y) = x^2 - y^2',
      category: 'Hessian',
      passed,
      expected: 'Eigenvalues [+2, -2], Indefinite (Saddle) at (0, 0)',
      actualOutput: `eigenvalues=[${hessian.eigenvalues[0].toFixed(2)}, ${hessian.eigenvalues[1].toFixed(2)}], class="${hessian.definiteness}"`,
      durationMs: performance.now() - t0,
    });
  }

  // Test I: Critical Point Finder on Non-Convex Quartic
  {
    const t0 = performance.now();
    const quartic = analyzeExpression('x**4 - 2*x**2 + y**2');
    const cpts = findCriticalPoints2D(quartic, 3);
    const hasLeftMin = cpts.some(
      (c) => Math.abs(c.x + 1) < 1e-2 && c.hessian.classification === 'Strict Local Minimum'
    );
    const hasRightMin = cpts.some(
      (c) => Math.abs(c.x - 1) < 1e-2 && c.hessian.classification === 'Strict Local Minimum'
    );
    const hasSaddle = cpts.some(
      (c) => Math.abs(c.x) < 1e-2 && c.hessian.classification === 'Saddle Point'
    );
    const passed = hasLeftMin && hasRightMin && hasSaddle;
    results.push({
      id: 'TEST-CRIT',
      name: '2D Critical Points of Non-Convex x^4 - 2x^2 + y^2',
      category: 'Hessian',
      passed,
      expected: 'Locate minima at (-1, 0), (1, 0) and saddle at (0, 0)',
      actualOutput: `found ${cpts.length} points: ${cpts.map((c) => `(${c.x.toFixed(2)},${c.y.toFixed(2)}):${c.hessian.classification}`).join('; ')}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test J: Linear Regression from Scratch & CSV Upload Parsing
  {
    const t0 = performance.now();
    const csv = parseCsvDataset('x,y\n1,3\n2,5\n3,7\n4,9\n5,11\n6,13', 'unit.csv');
    const mlRes = trainLinearRegressionFromScratch({
      dataset: csv,
      inputCols: ['x'],
      targetCol: 'y',
      optimizer: 'batch_gd',
      learningRate: 0.1,
      epochs: 250,
    });
    const slopeErr = Math.abs(mlRes.rawWeights[0] - 2);
    const biasErr = Math.abs(mlRes.rawBias - 1);
    const passed = slopeErr < 1e-3 && biasErr < 1e-3 && mlRes.valMetrics.r2 > 0.999;
    results.push({
      id: 'TEST-ML',
      name: 'Linear Regression from Scratch & CSV Parser (y = 2x + 1)',
      category: 'Machine Learning',
      passed,
      expected: 'Recover slope=2.0, intercept=1.0, Val R^2 > 0.999',
      actualOutput: `slope=${mlRes.rawWeights[0].toFixed(5)}, intercept=${mlRes.rawBias.toFixed(5)}, val_R2=${mlRes.valMetrics.r2.toFixed(6)}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test K: Iteration Limit Enforcement
  {
    const t0 = performance.now();
    const res = runGradientDescent1D('x^2', 10, 0.01, 15, 1e-6, 'fixed');
    const passed = res.totalIterations === 15 && res.status === 'max_iterations';
    results.push({
      id: 'TEST-LIMIT',
      name: 'Iteration Limit Enforcement (maxIter = 15, fixed stopping)',
      category: 'Validation',
      passed,
      expected: 'Halt cleanly at exactly 15 iterations with max_iterations status',
      actualOutput: `totalIterations=${res.totalIterations}, status=${res.status}, x_15=${res.finalX.toFixed(4)}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test L: Empirical Linear Contraction Ratio & O(h^2) Central FD Scaling
  {
    const t0 = performance.now();
    const res = runGradientDescent1D('x^2', 10, 0.1, 15, 1e-12, 'fixed');
    let maxRatioErr = 0;
    for (let n = 0; n < 10; n++) {
      const ratio = res.history[n + 1].x / res.history[n].x;
      maxRatioErr = Math.max(maxRatioErr, Math.abs(ratio - 0.8));
    }
    const analysis = analyzeExpression('sin(x) + x^3');
    const fd1 = computeFiniteDifferences1D(analysis, 1.2, 1e-2);
    const fd2 = computeFiniteDifferences1D(analysis, 1.2, 1e-3);
    const fdScalingRatio = fd1.centralError / fd2.centralError;
    const passed = maxRatioErr < 1e-12 && fdScalingRatio >= 95 && fdScalingRatio <= 105;
    results.push({
      id: 'TEST-RATE',
      name: 'Linear Contraction Factor (1 - 2α = 0.8) & O(h^2) FD Scaling',
      category: 'Convergence',
      passed,
      expected: 'max |x_{n+1}/x_n - 0.8| < 1e-12 and E(1e-2)/E(1e-3) ≈ 100',
      actualOutput: `maxContractionErr=${maxRatioErr.toExponential(2)}, centralFDScalingRatio=${fdScalingRatio.toFixed(2)}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test M: Non-Convex Curved Valley — Rosenbrock Optimization via Adam
  {
    const t0 = performance.now();
    const res = runGradientDescent2D(
      '(1 - x)**2 + 5*(y - x**2)**2',
      -0.5,
      1.0,
      0.05,
      800,
      1e-4,
      'adam'
    );
    const dist = Math.hypot(res.finalX - 1, res.finalY - 1);
    const passed = res.status === 'converged_grad' && dist < 1e-2 && res.finalFxy < 1e-6;
    results.push({
      id: 'TEST-ROSEN',
      name: 'Rosenbrock Curved Valley (1-x)^2 + 5(y-x^2)^2 via Adam',
      category: 'Multivariable',
      passed,
      expected: 'Converge to global min (1, 1) with ||X - (1,1)|| < 1e-2',
      actualOutput: `status=${res.status}, iters=${res.totalIterations}, final=(${res.finalX.toFixed(4)}, ${res.finalY.toFixed(4)}), f=${res.finalFxy.toExponential(2)}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test N: All ML Optimizer Variants (SGD, Mini-Batch, Momentum, RMSProp, Adam)
  {
    const t0 = performance.now();
    const csv = parseCsvDataset(
      'x,y\n1,3\n2,5\n3,7\n4,9\n5,11\n6,13\n7,15\n8,17',
      'variants.csv'
    );
    const runs = [
      { opt: 'sgd' as const, lr: 0.1, epochs: 250 },
      { opt: 'minibatch_gd' as const, lr: 0.1, epochs: 250 },
      { opt: 'momentum' as const, lr: 0.1, epochs: 250 },
      { opt: 'rmsprop' as const, lr: 0.25, epochs: 350 },
      { opt: 'adam' as const, lr: 0.25, epochs: 350 },
    ].map((cfg) => {
      const r = trainLinearRegressionFromScratch({
        dataset: csv,
        inputCols: ['x'],
        targetCol: 'y',
        optimizer: cfg.opt,
        learningRate: cfg.lr,
        epochs: cfg.epochs,
        batchSize: 2,
      });
      return { opt: cfg.opt, r2: r.valMetrics.r2 };
    });
    const passed = runs.every((r) => r.r2 > 0.99);
    results.push({
      id: 'TEST-ML-VARIANTS',
      name: 'ML Optimizer Variants (SGD, Mini-Batch, Momentum, RMSProp, Adam)',
      category: 'Machine Learning',
      passed,
      expected: 'All 5 ML optimizer variants achieve Val R^2 > 0.99 on y = 2x + 1',
      actualOutput: runs.map((r) => `${r.opt}:${r.r2.toFixed(4)}`).join(', '),
      durationMs: performance.now() - t0,
    });
  }

  // Test O: L2 Ridge Regularization Mitigating Degree-9 Overfitting
  {
    const t0 = performance.now();
    const waveDataset = getBuiltinDatasets()[2];
    const unreg = trainLinearRegressionFromScratch({
      dataset: waveDataset,
      inputCols: ['x'],
      targetCol: 'y',
      optimizer: 'adam',
      learningRate: 0.15,
      epochs: 1200,
      degree: 9,
      l2Lambda: 0,
    });
    const ridge = trainLinearRegressionFromScratch({
      dataset: waveDataset,
      inputCols: ['x'],
      targetCol: 'y',
      optimizer: 'adam',
      learningRate: 0.15,
      epochs: 1200,
      degree: 9,
      l2Lambda: 0.05,
    });
    const passed = ridge.valMetrics.mse < unreg.valMetrics.mse && ridge.valMetrics.mse < 0.2;
    results.push({
      id: 'TEST-RIDGE',
      name: 'L2 Ridge Penalty Reducing Degree-9 Polynomial Overfitting',
      category: 'Machine Learning',
      passed,
      expected: 'Ridge (λ=0.05) Val MSE strictly lower than unregularized (λ=0)',
      actualOutput: `unregValMSE=${unreg.valMetrics.mse.toFixed(4)}, ridgeValMSE=${ridge.valMetrics.mse.toFixed(4)}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test P: Non-Differentiable Kink Behavior on f(x) = |x|
  {
    const t0 = performance.now();
    const osc = runGradientDescent1D('abs(x)', 0.5, 0.6, 10, 1e-6);
    const kinkHit = runGradientDescent1D('abs(x)', 0.5, 0.5, 10, 1e-6);
    const passed =
      osc.status === 'max_iterations' &&
      Math.abs((osc.history[1]?.x ?? NaN) + 0.1) < 1e-12 &&
      Math.abs(osc.finalX - 0.5) < 1e-12 &&
      kinkHit.status === 'diverged';
    results.push({
      id: 'TEST-ABS',
      name: 'Non-Differentiable Kink Behavior: f(x) = |x| Oscillation & Kink Guard',
      category: 'Divergence',
      passed,
      expected: '2-cycle oscillation {0.5, -0.1} on α=0.6; safe halt at x=0 kink on α=0.5',
      actualOutput: `oscStatus=${osc.status}, x_1=${osc.history[1]?.x.toFixed(1)}, x_10=${osc.finalX.toFixed(1)}; kinkStatus=${kinkHit.status}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test Q: Audit Regression — Domain Guard, Tangent Root (x-1)^3 & Saddle Roll
  {
    const t0 = performance.now();
    const outHess = analyzeHessian2D(analyzeExpression('sqrt(x) + y^2'), -1, 0);
    const outGrid = scanGridConvexity2D(analyzeExpression('log(x - 100) + y^2'), 3, 5);
    const tangentPts = findStationaryPoints1D(analyzeExpression('(x - 1)^3'), -3, 3, 400);
    const saddleRoll = runGradientDescent2D('x^4 - 2*x^2 + y^2', 0.01, 0, 0.05, 20, 1e-6, 'gd');
    const passed =
      outHess.definiteness === 'Undefined / Out of Domain' &&
      outGrid.globalClassification === 'Undefined on Grid' &&
      tangentPts.some((p) => Math.abs(p.x - 1) < 1e-3) &&
      saddleRoll.status === 'max_iterations' &&
      saddleRoll.finalFxy < saddleRoll.history[0].fxy;
    results.push({
      id: 'TEST-AUDIT-DOMAIN',
      name: 'Audit Regression: Out-of-Domain Hessian, Tangent Root & Saddle Descent',
      category: 'Hessian',
      passed,
      expected: 'Traps NaN domain cleanly; finds x*=1 of (x-1)^3; saddle downhill not diverged',
      actualOutput: `hess="${outHess.definiteness}", grid="${outGrid.globalClassification}", rootX=${tangentPts[0]?.x.toFixed(4) ?? 'none'}, saddleStatus=${saddleRoll.status}`,
      durationMs: performance.now() - t0,
    });
  }

  // Test R: Audit Regression — ML Final gradNorm, Negative Lambda Guard & PDF Export
  {
    const t0 = performance.now();
    const ml = trainLinearRegressionFromScratch({
      dataset: getBuiltinDatasets()[0],
      inputCols: ['study_hours'],
      targetCol: 'exam_score',
      optimizer: 'batch_gd',
      learningRate: 0.1,
      epochs: 20,
    });
    let negLambdaRejected = false;
    try {
      trainLinearRegressionFromScratch({
        dataset: getBuiltinDatasets()[0],
        inputCols: ['study_hours'],
        targetCol: 'exam_score',
        optimizer: 'batch_gd',
        learningRate: 0.1,
        epochs: 20,
        l2Lambda: -0.5,
      });
    } catch {
      negLambdaRejected = true;
    }
    const doc = new jsPDF();
    doc.text('GradAI Audit Verification', 14, 20);
    autoTable(doc, { startY: 28, head: [['n', 'x']], body: [[0, '10.0']] });
    const pdfBytes = doc.output('arraybuffer');
    const pdfHeader = String.fromCharCode(...new Uint8Array(pdfBytes.slice(0, 5)));
    const finalEpochGrad = ml.history[ml.history.length - 1]?.gradNorm ?? 0;
    const passed =
      finalEpochGrad > 0 &&
      negLambdaRejected &&
      pdfHeader === '%PDF-' &&
      pdfBytes.byteLength > 500;
    results.push({
      id: 'TEST-AUDIT-ML-PDF',
      name: 'Audit Regression: ML Epoch GradNorm, λ<0 Guard & Binary PDF Export',
      category: 'Machine Learning',
      passed,
      expected: 'Non-zero final gradNorm on unconverged run, λ<0 rejected, valid %PDF- binary',
      actualOutput: `finalGrad=${finalEpochGrad.toFixed(3)}, negLambdaRejected=${negLambdaRejected}, pdf=${pdfHeader} (${pdfBytes.byteLength}B)`,
      durationMs: performance.now() - t0,
    });
  }

  return results;
}

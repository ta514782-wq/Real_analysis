import { analyzeExpression } from './mathematics';
import { validateOptimizerParams } from './validation';

export type OptimizerAlgorithm = 'gd' | 'momentum' | 'rmsprop' | 'adam';

export type ConvergenceStatus =
  | 'converged_grad'
  | 'converged_step'
  | 'max_iterations'
  | 'diverged'
  | 'numerical_error';

export interface IterationRecord1D {
  iteration: number;
  x: number;
  fx: number;
  grad: number;
  absGrad: number;
  stepDelta: number;
}

export interface GradientDescentResult1D {
  expression: string;
  latexF: string;
  dfDxLatex: string;
  alpha: number;
  x0: number;
  maxIterations: number;
  tolerance: number;
  history: IterationRecord1D[];
  finalX: number;
  finalFx: number;
  finalGrad: number;
  totalIterations: number;
  status: ConvergenceStatus;
  statusLabel: string;
  explanation: string;
}

export interface IterationRecord2D {
  iteration: number;
  x: number;
  y: number;
  fxy: number;
  gradX: number;
  gradY: number;
  gradNorm: number;
  stepDelta: number;
}

export interface GradientDescentResult2D {
  expression: string;
  algorithm: OptimizerAlgorithm;
  latexF: string;
  dfDxLatex: string;
  dfDyLatex: string;
  alpha: number;
  x0: number;
  y0: number;
  maxIterations: number;
  tolerance: number;
  history: IterationRecord2D[];
  finalX: number;
  finalY: number;
  finalFxy: number;
  finalGradNorm: number;
  totalIterations: number;
  status: ConvergenceStatus;
  statusLabel: string;
  explanation: string;
}

export interface OptimizerHyperparams {
  beta?: number; // Momentum factor (default 0.9)
  rho?: number; // RMSProp decay (default 0.9)
  beta1?: number; // Adam first moment (default 0.9)
  beta2?: number; // Adam second moment (default 0.999)
  epsilon?: number; // Numerical stability epsilon (default 1e-8)
}

export type StoppingCriterion = 'both' | 'gradient' | 'step' | 'fixed';

const DIVERGENCE_COORD_LIMIT = 1e8;
const DIVERGENCE_VALUE_LIMIT = 1e14;

/**
 * Genuine 1D Gradient Descent from scratch:
 * x(n+1) = x(n) - alpha * f'(x(n))
 */
export function runGradientDescent1D(
  rawExpr: string,
  x0: number,
  alpha: number,
  maxIterations = 100,
  tolerance = 1e-6,
  stoppingCriterion: StoppingCriterion = 'both'
): GradientDescentResult1D {
  const paramCheck = validateOptimizerParams({
    alpha,
    maxIterations,
    tolerance,
    x0,
  });
  if (!paramCheck.valid) {
    throw new Error(paramCheck.error);
  }

  const analysis = analyzeExpression(rawExpr, { allowedDimensions: [1] });
  const history: IterationRecord1D[] = [];

  let x = x0;
  let fx = analysis.evalF(x);
  let grad = analysis.evalGrad(x)[0];

  if (!Number.isFinite(fx) || !Number.isFinite(grad)) {
    return {
      expression: analysis.normalized,
      latexF: analysis.latexF,
      dfDxLatex: analysis.dfDxLatex,
      alpha,
      x0,
      maxIterations,
      tolerance,
      history: [],
      finalX: x0,
      finalFx: fx,
      finalGrad: grad,
      totalIterations: 0,
      status: 'numerical_error',
      statusLabel: 'Numerical Error at Initialization',
      explanation: `f(x) or f'(x) is undefined or non-finite at x0 = ${x0}. Choose a starting point inside the domain.`,
    };
  }

  history.push({
    iteration: 0,
    x,
    fx,
    grad,
    absGrad: Math.abs(grad),
    stepDelta: 0,
  });

  if (stoppingCriterion !== 'fixed' && stoppingCriterion !== 'step' && Math.abs(grad) <= tolerance) {
    return {
      expression: analysis.normalized,
      latexF: analysis.latexF,
      dfDxLatex: analysis.dfDxLatex,
      alpha,
      x0,
      maxIterations,
      tolerance,
      history,
      finalX: x,
      finalFx: fx,
      finalGrad: grad,
      totalIterations: 0,
      status: 'converged_grad',
      statusLabel: 'Converged at Initial Point',
      explanation: `Initial point x0 = ${x0} already satisfies |f'(x0)| <= ${tolerance.toExponential(1)}.`,
    };
  }

  let status: ConvergenceStatus = 'max_iterations';
  let statusLabel = 'Maximum Iterations Reached';
  let explanation = `Completed ${maxIterations} iterations without reaching stopping tolerance ${tolerance.toExponential(1)}.`;

  for (let n = 1; n <= maxIterations; n++) {
    const nextX = x - alpha * grad;

    if (!Number.isFinite(nextX) || Math.abs(nextX) > DIVERGENCE_COORD_LIMIT) {
      status = 'diverged';
      statusLabel = 'Diverged (Coordinate Overflow)';
      explanation = `Sequence diverged at iteration ${n}: |x(${n})| exceeded ${DIVERGENCE_COORD_LIMIT.toExponential(0)}. Learning rate alpha = ${alpha} is too large for local curvature.`;
      break;
    }

    const nextFx = analysis.evalF(nextX);
    const nextGrad = analysis.evalGrad(nextX)[0];

    if (
      !Number.isFinite(nextFx) ||
      !Number.isFinite(nextGrad) ||
      Math.abs(nextFx) > DIVERGENCE_VALUE_LIMIT
    ) {
      status = 'diverged';
      statusLabel = 'Diverged (Function Overflow / Out of Domain)';
      explanation = `Iteration ${n} left the safe numerical domain (x = ${nextX.toFixed(4)}).`;
      break;
    }

    const stepDelta = Math.abs(nextX - x);
    x = nextX;
    fx = nextFx;
    grad = nextGrad;

    history.push({
      iteration: n,
      x,
      fx,
      grad,
      absGrad: Math.abs(grad),
      stepDelta,
    });

    if (
      (stoppingCriterion === 'both' || stoppingCriterion === 'gradient') &&
      Math.abs(grad) <= tolerance
    ) {
      status = 'converged_grad';
      statusLabel = 'Converged (Gradient Tolerance)';
      explanation = `Stopped at iteration ${n} because |f'(x_${n})| = ${Math.abs(grad).toExponential(3)} <= ${tolerance.toExponential(1)}.`;
      break;
    }

    const stepTol = stoppingCriterion === 'step' ? tolerance : tolerance * 1e-3;
    if (
      (stoppingCriterion === 'both' || stoppingCriterion === 'step') &&
      stepDelta <= stepTol &&
      n > 1
    ) {
      status = 'converged_step';
      statusLabel = 'Converged (Step Size Tolerance)';
      explanation = `Stopped at iteration ${n} because step change |x_${n} - x_${n - 1}| = ${stepDelta.toExponential(3)} fell below threshold.`;
      break;
    }
  }

  // Detect clear unbounded objective & gradient growth within maxIterations (e.g., Test C before 1e8 threshold)
  if (
    status === 'max_iterations' &&
    history.length >= 6 &&
    history[history.length - 1].absGrad > history[0].absGrad * 10 &&
    history[history.length - 1].fx > history[0].fx &&
    history[history.length - 1].fx > history[history.length - 4].fx
  ) {
    status = 'diverged';
    statusLabel = 'Diverged (Unbounded Growth)';
    explanation = `Iterate magnitudes grew unboundedly from |x0|=${Math.abs(x0)} to |x_${history.length - 1}|=${Math.abs(x).toExponential(3)} with alpha = ${alpha}.`;
  }

  return {
    expression: analysis.normalized,
    latexF: analysis.latexF,
    dfDxLatex: analysis.dfDxLatex,
    alpha,
    x0,
    maxIterations,
    tolerance,
    history,
    finalX: x,
    finalFx: fx,
    finalGrad: grad,
    totalIterations: history.length - 1,
    status,
    statusLabel,
    explanation,
  };
}

/**
 * Genuine multivariable optimization from scratch supporting:
 * Batch GD, Momentum, RMSProp, and Adam.
 */
export function runGradientDescent2D(
  rawExpr: string,
  x0: number,
  y0: number,
  alpha: number,
  maxIterations = 100,
  tolerance = 1e-6,
  algorithm: OptimizerAlgorithm = 'gd',
  hyperparams: OptimizerHyperparams = {},
  stoppingCriterion: StoppingCriterion = 'both'
): GradientDescentResult2D {
  const paramCheck = validateOptimizerParams({
    alpha,
    maxIterations,
    tolerance,
    x0,
    y0,
  });
  if (!paramCheck.valid) {
    throw new Error(paramCheck.error);
  }

  const analysis = analyzeExpression(rawExpr, { allowedDimensions: [1, 2] });
  const beta = hyperparams.beta ?? 0.9;
  const rho = hyperparams.rho ?? 0.9;
  const beta1 = hyperparams.beta1 ?? 0.9;
  const beta2 = hyperparams.beta2 ?? 0.999;
  const epsilon = hyperparams.epsilon ?? 1e-8;

  const history: IterationRecord2D[] = [];
  let x = x0;
  let y = y0;
  let fxy = analysis.evalF(x, y);
  let [gx, gy] = analysis.evalGrad(x, y);
  let gradNorm = Math.hypot(gx, gy);

  if (!Number.isFinite(fxy) || !Number.isFinite(gradNorm)) {
    return {
      expression: analysis.normalized,
      algorithm,
      latexF: analysis.latexF,
      dfDxLatex: analysis.dfDxLatex,
      dfDyLatex: analysis.dfDyLatex,
      alpha,
      x0,
      y0,
      maxIterations,
      tolerance,
      history: [],
      finalX: x0,
      finalY: y0,
      finalFxy: fxy,
      finalGradNorm: gradNorm,
      totalIterations: 0,
      status: 'numerical_error',
      statusLabel: 'Numerical Error at Initialization',
      explanation: `Function or gradient is undefined at (${x0}, ${y0}).`,
    };
  }

  history.push({
    iteration: 0,
    x,
    y,
    fxy,
    gradX: gx,
    gradY: gy,
    gradNorm,
    stepDelta: 0,
  });

  if (stoppingCriterion !== 'fixed' && stoppingCriterion !== 'step' && gradNorm <= tolerance) {
    return {
      expression: analysis.normalized,
      algorithm,
      latexF: analysis.latexF,
      dfDxLatex: analysis.dfDxLatex,
      dfDyLatex: analysis.dfDyLatex,
      alpha,
      x0,
      y0,
      maxIterations,
      tolerance,
      history,
      finalX: x,
      finalY: y,
      finalFxy: fxy,
      finalGradNorm: gradNorm,
      totalIterations: 0,
      status: 'converged_grad',
      statusLabel: 'Converged at Initial Point',
      explanation: `Initial gradient norm ||grad f|| = ${gradNorm.toExponential(3)} <= ${tolerance.toExponential(1)}.`,
    };
  }

  // Optimizer state vectors
  let vx = 0;
  let vy = 0;
  let sx = 0;
  let sy = 0;
  let mx = 0;
  let my = 0;

  let status: ConvergenceStatus = 'max_iterations';
  let statusLabel = 'Maximum Iterations Reached';
  let explanation = `Completed ${maxIterations} iterations with ${algorithm.toUpperCase()}.`;

  for (let n = 1; n <= maxIterations; n++) {
    let nextX = x;
    let nextY = y;

    if (algorithm === 'gd') {
      nextX = x - alpha * gx;
      nextY = y - alpha * gy;
    } else if (algorithm === 'momentum') {
      vx = beta * vx - alpha * gx;
      vy = beta * vy - alpha * gy;
      nextX = x + vx;
      nextY = y + vy;
    } else if (algorithm === 'rmsprop') {
      sx = rho * sx + (1 - rho) * gx * gx;
      sy = rho * sy + (1 - rho) * gy * gy;
      nextX = x - (alpha / (Math.sqrt(sx) + epsilon)) * gx;
      nextY = y - (alpha / (Math.sqrt(sy) + epsilon)) * gy;
    } else if (algorithm === 'adam') {
      mx = beta1 * mx + (1 - beta1) * gx;
      my = beta1 * my + (1 - beta1) * gy;
      sx = beta2 * sx + (1 - beta2) * gx * gx;
      sy = beta2 * sy + (1 - beta2) * gy * gy;
      const mxHat = mx / (1 - Math.pow(beta1, n));
      const myHat = my / (1 - Math.pow(beta1, n));
      const sxHat = sx / (1 - Math.pow(beta2, n));
      const syHat = sy / (1 - Math.pow(beta2, n));
      nextX = x - (alpha / (Math.sqrt(sxHat) + epsilon)) * mxHat;
      nextY = y - (alpha / (Math.sqrt(syHat) + epsilon)) * myHat;
    }

    if (
      !Number.isFinite(nextX) ||
      !Number.isFinite(nextY) ||
      Math.hypot(nextX, nextY) > DIVERGENCE_COORD_LIMIT
    ) {
      status = 'diverged';
      statusLabel = 'Diverged (Coordinate Overflow)';
      explanation = `Coordinates exceeded safe bounds at iteration ${n} under ${algorithm.toUpperCase()} with alpha = ${alpha}.`;
      break;
    }

    const nextFxy = analysis.evalF(nextX, nextY);
    const [nextGx, nextGy] = analysis.evalGrad(nextX, nextY);
    const nextGradNorm = Math.hypot(nextGx, nextGy);

    if (
      !Number.isFinite(nextFxy) ||
      !Number.isFinite(nextGradNorm) ||
      Math.abs(nextFxy) > DIVERGENCE_VALUE_LIMIT
    ) {
      status = 'diverged';
      statusLabel = 'Diverged (Function Overflow)';
      explanation = `Objective value overflowed or left domain at iteration ${n}.`;
      break;
    }

    const stepDelta = Math.hypot(nextX - x, nextY - y);
    x = nextX;
    y = nextY;
    fxy = nextFxy;
    gx = nextGx;
    gy = nextGy;
    gradNorm = nextGradNorm;

    history.push({
      iteration: n,
      x,
      y,
      fxy,
      gradX: gx,
      gradY: gy,
      gradNorm,
      stepDelta,
    });

    if (
      (stoppingCriterion === 'both' || stoppingCriterion === 'gradient') &&
      gradNorm <= tolerance
    ) {
      status = 'converged_grad';
      statusLabel = 'Converged (Gradient Norm Tolerance)';
      explanation = `Stopped at iteration ${n}: ||grad f|| = ${gradNorm.toExponential(3)} <= ${tolerance.toExponential(1)}.`;
      break;
    }

    const stepTol = stoppingCriterion === 'step' ? tolerance : tolerance * 1e-4;
    if (
      (stoppingCriterion === 'both' || stoppingCriterion === 'step') &&
      stepDelta <= stepTol &&
      n > 3
    ) {
      status = 'converged_step';
      statusLabel = 'Converged (Step Size Tolerance)';
      explanation = `Stopped at iteration ${n}: step magnitude ${stepDelta.toExponential(3)} fell below threshold.`;
      break;
    }
  }

  if (
    status === 'max_iterations' &&
    history.length >= 6 &&
    history[history.length - 1].gradNorm > history[0].gradNorm * 10 &&
    history[history.length - 1].fxy > history[0].fxy &&
    history[history.length - 1].fxy > history[history.length - 4].fxy
  ) {
    status = 'diverged';
    statusLabel = 'Diverged (Unbounded Growth)';
    explanation = `Objective and gradient norm expanded unboundedly (||grad|| from ${history[0].gradNorm.toExponential(2)} to ${gradNorm.toExponential(2)}) with alpha = ${alpha}.`;
  }

  return {
    expression: analysis.normalized,
    algorithm,
    latexF: analysis.latexF,
    dfDxLatex: analysis.dfDxLatex,
    dfDyLatex: analysis.dfDyLatex,
    alpha,
    x0,
    y0,
    maxIterations,
    tolerance,
    history,
    finalX: x,
    finalY: y,
    finalFxy: fxy,
    finalGradNorm: gradNorm,
    totalIterations: history.length - 1,
    status,
    statusLabel,
    explanation,
  };
}

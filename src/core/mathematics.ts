import { compile, derivative, EvalFunction, parse, simplify } from 'mathjs';
import { validateExpression } from './validation';

export interface ExpressionAnalysis {
  raw: string;
  normalized: string;
  dimension: 1 | 2;
  variables: ('x' | 'y')[];
  latexF: string;
  dfDxStr: string;
  dfDxLatex: string;
  dfDyStr: string;
  dfDyLatex: string;
  d2fDx2Str: string;
  d2fDx2Latex: string;
  d2fDy2Str: string;
  d2fDy2Latex: string;
  d2fDxDyStr: string;
  d2fDxDyLatex: string;
  evalF: (x: number, y?: number) => number;
  evalGrad: (x: number, y?: number) => [number, number];
  evalSecondPartials: (
    x: number,
    y?: number
  ) => { fxx: number; fyy: number; fxy: number };
}

export interface StationaryPoint1D {
  x: number;
  fx: number;
  fSecond: number;
  classification: 'Local Minimum' | 'Local Maximum' | 'Inflection / Degenerate';
}

export interface FiniteDifferenceComparison {
  x: number;
  h: number;
  exactSymbolic: number;
  forward: number;
  backward: number;
  central: number;
  forwardError: number;
  backwardError: number;
  centralError: number;
}

export type HessianDefiniteness =
  | 'Positive Definite'
  | 'Negative Definite'
  | 'Indefinite (Saddle)'
  | 'Positive Semidefinite'
  | 'Negative Semidefinite'
  | 'Zero / Degenerate'
  | 'Undefined / Out of Domain';

export interface HessianAnalysis2D {
  x: number;
  y: number;
  matrix: [[number, number], [number, number]];
  trace: number;
  determinant: number;
  eigenvalues: [number, number];
  definiteness: HessianDefiniteness;
  classification: string;
  convexityNote: string;
  gradNorm: number;
}

function safeEvalCompiled(fn: EvalFunction, x: number, y = 0): number {
  try {
    const val = fn.evaluate({ x, y });
    if (typeof val !== 'number' || !Number.isFinite(val)) {
      return Number.NaN;
    }
    return val;
  } catch {
    return Number.NaN;
  }
}

function safeToLatex(exprStr: string): string {
  try {
    return parse(exprStr).toTex({ parenthesis: 'auto' });
  } catch {
    return exprStr;
  }
}

/**
 * Performs symbolic differentiation and builds safe compiled evaluators
 * for f, gradient, and second-order Hessian partials.
 */
export function analyzeExpression(
  rawExpr: string,
  options: { requireVariable?: boolean; allowedDimensions?: (1 | 2)[] } = {}
): ExpressionAnalysis {
  const validation = validateExpression(rawExpr, options);
  if (!validation.valid) {
    throw new Error(validation.error ?? 'Invalid mathematical expression.');
  }

  const { normalized, dimension, variables } = validation;
  const fNode = parse(normalized);

  const dfDxNode = simplify(derivative(fNode, 'x'));
  const dfDyNode = simplify(derivative(fNode, 'y'));

  const d2fDx2Node = simplify(derivative(dfDxNode, 'x'));
  const d2fDy2Node = simplify(derivative(dfDyNode, 'y'));
  const d2fDxDyNode = simplify(derivative(dfDxNode, 'y'));

  const fCompiled = compile(normalized);
  const dfDxCompiled = dfDxNode.compile();
  const dfDyCompiled = dfDyNode.compile();
  const d2fDx2Compiled = d2fDx2Node.compile();
  const d2fDy2Compiled = d2fDy2Node.compile();
  const d2fDxDyCompiled = d2fDxDyNode.compile();

  return {
    raw: rawExpr,
    normalized,
    dimension,
    variables,
    latexF: fNode.toTex({ parenthesis: 'auto' }),
    dfDxStr: dfDxNode.toString(),
    dfDxLatex: safeToLatex(dfDxNode.toString()),
    dfDyStr: dfDyNode.toString(),
    dfDyLatex: safeToLatex(dfDyNode.toString()),
    d2fDx2Str: d2fDx2Node.toString(),
    d2fDx2Latex: safeToLatex(d2fDx2Node.toString()),
    d2fDy2Str: d2fDy2Node.toString(),
    d2fDy2Latex: safeToLatex(d2fDy2Node.toString()),
    d2fDxDyStr: d2fDxDyNode.toString(),
    d2fDxDyLatex: safeToLatex(d2fDxDyNode.toString()),
    evalF: (x: number, y = 0) => safeEvalCompiled(fCompiled, x, y),
    evalGrad: (x: number, y = 0) => [
      safeEvalCompiled(dfDxCompiled, x, y),
      safeEvalCompiled(dfDyCompiled, x, y),
    ],
    evalSecondPartials: (x: number, y = 0) => ({
      fxx: safeEvalCompiled(d2fDx2Compiled, x, y),
      fyy: safeEvalCompiled(d2fDy2Compiled, x, y),
      fxy: safeEvalCompiled(d2fDxDyCompiled, x, y),
    }),
  };
}

/**
 * Compares symbolic f'(x) with forward, backward, and central finite differences.
 */
export function computeFiniteDifferences1D(
  analysis: ExpressionAnalysis,
  x: number,
  h: number
): FiniteDifferenceComparison {
  if (!Number.isFinite(h) || h <= 0) {
    throw new Error('Step size h must be strictly positive.');
  }
  const fx = analysis.evalF(x);
  const fxPlus = analysis.evalF(x + h);
  const fxMinus = analysis.evalF(x - h);
  const [exactSymbolic] = analysis.evalGrad(x);

  const forward = (fxPlus - fx) / h;
  const backward = (fx - fxMinus) / h;
  const central = (fxPlus - fxMinus) / (2 * h);

  return {
    x,
    h,
    exactSymbolic,
    forward,
    backward,
    central,
    forwardError: Math.abs(forward - exactSymbolic),
    backwardError: Math.abs(backward - exactSymbolic),
    centralError: Math.abs(central - exactSymbolic),
  };
}

/**
 * Computes exact 2x2 Hessian eigenvalues, definiteness, and critical-point classification.
 */
export function analyzeHessian2D(
  analysis: ExpressionAnalysis,
  x: number,
  y: number
): HessianAnalysis2D {
  const fxy = analysis.evalF(x, y);
  const { fxx, fyy, fxy: fxyMix } = analysis.evalSecondPartials(x, y);
  const [gx, gy] = analysis.evalGrad(x, y);
  const gradNorm = Math.hypot(gx, gy);

  if (
    !Number.isFinite(fxy) ||
    !Number.isFinite(fxx) ||
    !Number.isFinite(fyy) ||
    !Number.isFinite(fxyMix) ||
    !Number.isFinite(gradNorm)
  ) {
    return {
      x,
      y,
      matrix: [
        [fxx, fxyMix],
        [fxyMix, fyy],
      ],
      trace: Number.NaN,
      determinant: Number.NaN,
      eigenvalues: [Number.NaN, Number.NaN],
      definiteness: 'Undefined / Out of Domain',
      classification: 'Out of Real Domain / Non-Differentiable',
      convexityNote:
        'Function, gradient, or second partial derivatives are undefined or non-real at this point.',
      gradNorm,
    };
  }

  const trace = fxx + fyy;
  const determinant = fxx * fyy - fxyMix * fxyMix;
  const discriminant = Math.max(0, trace * trace - 4 * determinant);
  const sqrtDisc = Math.sqrt(discriminant);

  const lambda1 = 0.5 * (trace + sqrtDisc);
  const lambda2 = 0.5 * (trace - sqrtDisc);
  const eps = 1e-8;

  let definiteness: HessianDefiniteness;
  let classification: string;
  let convexityNote: string;

  if (Math.abs(lambda1) < eps && Math.abs(lambda2) < eps) {
    definiteness = 'Zero / Degenerate';
    classification = 'Flat / Higher-Order Test Needed';
    convexityNote = 'Both Hessian eigenvalues are zero at this point; curvature is degenerate.';
  } else if (lambda1 > eps && lambda2 > eps) {
    definiteness = 'Positive Definite';
    classification =
      gradNorm < 1e-4 ? 'Strict Local Minimum' : 'Locally Strictly Convex Region';
    convexityNote =
      'Both eigenvalues are strictly positive (lambda_1 > 0, lambda_2 > 0), guaranteeing upward bowl curvature.';
  } else if (lambda1 < -eps && lambda2 < -eps) {
    definiteness = 'Negative Definite';
    classification =
      gradNorm < 1e-4 ? 'Strict Local Maximum' : 'Locally Strictly Concave Region';
    convexityNote =
      'Both eigenvalues are strictly negative (lambda_1 < 0, lambda_2 < 0), indicating downward dome curvature.';
  } else if (lambda1 > eps && lambda2 < -eps) {
    definiteness = 'Indefinite (Saddle)';
    classification =
      gradNorm < 1e-4 ? 'Saddle Point' : 'Indefinite Curvature (Non-Convex)';
    convexityNote =
      'Eigenvalues have opposite signs (lambda_1 > 0, lambda_2 < 0): upward curvature along one eigenvector and downward along the orthogonal eigenvector.';
  } else if (lambda1 >= -eps && lambda2 >= -eps) {
    definiteness = 'Positive Semidefinite';
    classification = 'Weakly Convex / Valley Ridge';
    convexityNote =
      'Eigenvalues are non-negative with one zero eigenvalue, indicating a flat trough direction.';
  } else {
    definiteness = 'Negative Semidefinite';
    classification = 'Weakly Concave / Crest Ridge';
    convexityNote =
      'Eigenvalues are non-positive with one zero eigenvalue, indicating a flat ridge direction.';
  }

  return {
    x,
    y,
    matrix: [
      [fxx, fxyMix],
      [fxyMix, fyy],
    ],
    trace,
    determinant,
    eigenvalues: [lambda1, lambda2],
    definiteness,
    classification,
    convexityNote,
    gradNorm,
  };
}

/**
 * Locates and classifies stationary points of a 1D function f(x) in [minX, maxX]
 * via sign-change detection on f'(x) followed by bisection refinement.
 */
export function findStationaryPoints1D(
  analysis: ExpressionAnalysis,
  minX = -6,
  maxX = 6,
  samples = 400
): StationaryPoint1D[] {
  const roots: number[] = [];
  const step = (maxX - minX) / samples;

  const evalG = (x: number) => analysis.evalGrad(x)[0];

  const gridG: number[] = [];
  for (let i = 0; i <= samples; i++) {
    gridG.push(evalG(minX + i * step));
  }

  let prevX = minX;
  let prevG = gridG[0];

  for (let i = 1; i <= samples; i++) {
    const currX = minX + i * step;
    const currG = gridG[i];

    if (!Number.isFinite(prevG) || !Number.isFinite(currG)) {
      prevX = currX;
      prevG = currG;
      continue;
    }

    if (Math.abs(currG) < 1e-9) {
      roots.push(currX);
    } else if (prevG * currG < 0) {
      let lo = prevX;
      let hi = currX;
      let gLo = prevG;
      for (let b = 0; b < 45; b++) {
        const mid = 0.5 * (lo + hi);
        const gMid = evalG(mid);
        if (!Number.isFinite(gMid)) break;
        if (Math.abs(gMid) < 1e-12) {
          lo = mid;
          hi = mid;
          break;
        }
        if (gLo * gMid <= 0) {
          hi = mid;
        } else {
          lo = mid;
          gLo = gMid;
        }
      }
      roots.push(0.5 * (lo + hi));
    } else if (
      i < samples &&
      Number.isFinite(gridG[i + 1]) &&
      Math.abs(currG) < 0.2 &&
      Math.abs(currG) <= Math.abs(prevG) &&
      Math.abs(currG) <= Math.abs(gridG[i + 1])
    ) {
      // Golden-section search on |f'(x)| to catch even-multiplicity tangent roots (e.g., (x-1)^3)
      let a = prevX;
      let b = minX + (i + 1) * step;
      const phi = (Math.sqrt(5) - 1) / 2;
      let x1 = b - phi * (b - a);
      let x2 = a + phi * (b - a);
      let f1 = Math.abs(evalG(x1));
      let f2 = Math.abs(evalG(x2));
      for (let k = 0; k < 45; k++) {
        if (!Number.isFinite(f1) || !Number.isFinite(f2)) break;
        if (f1 < f2) {
          b = x2;
          x2 = x1;
          f2 = f1;
          x1 = b - phi * (b - a);
          f1 = Math.abs(evalG(x1));
        } else {
          a = x1;
          x1 = x2;
          f1 = f2;
          x2 = a + phi * (b - a);
          f2 = Math.abs(evalG(x2));
        }
      }
      const cand = 0.5 * (a + b);
      if (Math.abs(evalG(cand)) < 1e-5) {
        roots.push(cand);
      }
    }

    prevX = currX;
    prevG = currG;
  }

  // Deduplicate close roots
  const uniqueRoots: number[] = [];
  for (const r of roots) {
    if (!uniqueRoots.some((u) => Math.abs(u - r) < 1e-3)) {
      uniqueRoots.push(r);
    }
  }

  return uniqueRoots
    .map((x) => {
      const cleanX = Math.abs(x) < 1e-9 ? 0 : x;
      const fx = analysis.evalF(cleanX);
      const { fxx } = analysis.evalSecondPartials(cleanX);
      let classification: StationaryPoint1D['classification'] = 'Inflection / Degenerate';
      if (fxx > 1e-6) classification = 'Local Minimum';
      else if (fxx < -1e-6) classification = 'Local Maximum';
      return {
        x: cleanX,
        fx,
        fSecond: fxx,
        classification,
      };
    })
    .filter((pt) => Number.isFinite(pt.fx) && Math.abs(evalG(pt.x)) < 1e-3);
}

export interface GridConvexitySummary {
  xs: number[];
  ys: number[];
  minEigenGrid: number[][];
  globalClassification:
    | 'Strictly Convex on Grid'
    | 'Convex on Grid'
    | 'Strictly Concave on Grid'
    | 'Concave on Grid'
    | 'Non-Convex (Mixed Curvature)'
    | 'Undefined on Grid';
  minEigenvalue: number;
  maxEigenvalue: number;
}

/**
 * Evaluates Hessian eigenvalues across a 2D grid [-bound, bound]^2 to classify
 * empirical convexity/concavity on the sampled domain.
 */
export function scanGridConvexity2D(
  analysis: ExpressionAnalysis,
  bound = 4,
  gridN = 21
): GridConvexitySummary {
  const xs: number[] = [];
  const ys: number[] = [];
  const minEigenGrid: number[][] = [];
  let globalMinEig = Infinity;
  let globalMaxEig = -Infinity;

  for (let i = 0; i < gridN; i++) {
    const v = -bound + (i / (gridN - 1)) * (2 * bound);
    xs.push(v);
    ys.push(v);
  }

  for (let j = 0; j < gridN; j++) {
    const row: number[] = [];
    for (let i = 0; i < gridN; i++) {
      const h = analyzeHessian2D(analysis, xs[i], ys[j]);
      const lMin = Math.min(h.eigenvalues[0], h.eigenvalues[1]);
      const lMax = Math.max(h.eigenvalues[0], h.eigenvalues[1]);
      if (Number.isFinite(lMin) && Number.isFinite(lMax)) {
        globalMinEig = Math.min(globalMinEig, lMin);
        globalMaxEig = Math.max(globalMaxEig, lMax);
        row.push(lMin);
      } else {
        row.push(Number.NaN);
      }
    }
    minEigenGrid.push(row);
  }

  const eps = 1e-7;
  let globalClassification: GridConvexitySummary['globalClassification'] =
    'Non-Convex (Mixed Curvature)';

  if (!Number.isFinite(globalMinEig) || !Number.isFinite(globalMaxEig)) {
    globalClassification = 'Undefined on Grid';
  } else if (globalMinEig > eps) {
    globalClassification = 'Strictly Convex on Grid';
  } else if (globalMinEig >= -eps && globalMaxEig >= -eps) {
    globalClassification = 'Convex on Grid';
  } else if (globalMaxEig < -eps) {
    globalClassification = 'Strictly Concave on Grid';
  } else if (globalMaxEig <= eps) {
    globalClassification = 'Concave on Grid';
  }

  return {
    xs,
    ys,
    minEigenGrid,
    globalClassification,
    minEigenvalue: globalMinEig,
    maxEigenvalue: globalMaxEig,
  };
}

export interface CriticalPoint2D {
  x: number;
  y: number;
  fxy: number;
  hessian: HessianAnalysis2D;
}

/**
 * Locates critical points (where grad f = 0) in [-bound, bound]^2 using
 * multi-seed Newton-Raphson iterations on the gradient vector.
 */
export function findCriticalPoints2D(
  analysis: ExpressionAnalysis,
  bound = 4
): CriticalPoint2D[] {
  const found: CriticalPoint2D[] = [];
  const seeds = 7;

  for (let i = 0; i < seeds; i++) {
    for (let j = 0; j < seeds; j++) {
      let x = -bound + (i / (seeds - 1)) * (2 * bound);
      let y = -bound + (j / (seeds - 1)) * (2 * bound);

      for (let iter = 0; iter < 25; iter++) {
        const [gx, gy] = analysis.evalGrad(x, y);
        if (!Number.isFinite(gx) || !Number.isFinite(gy)) break;
        if (Math.hypot(gx, gy) < 1e-9) break;

        const { fxx, fyy, fxy } = analysis.evalSecondPartials(x, y);
        const det = fxx * fyy - fxy * fxy;
        if (!Number.isFinite(det) || Math.abs(det) < 1e-10) break;

        const dx = (fyy * gx - fxy * gy) / det;
        const dy = (-fxy * gx + fxx * gy) / det;
        x -= dx;
        y -= dy;
        if (!Number.isFinite(x) || !Number.isFinite(y) || Math.hypot(x, y) > bound * 2) {
          break;
        }
      }

      if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
      if (Math.abs(x) > bound || Math.abs(y) > bound) continue;

      const [gx, gy] = analysis.evalGrad(x, y);
      if (Math.hypot(gx, gy) > 1e-5) continue;

      const cleanX = Math.abs(x) < 1e-8 ? 0 : x;
      const cleanY = Math.abs(y) < 1e-8 ? 0 : y;

      if (!found.some((pt) => Math.hypot(pt.x - cleanX, pt.y - cleanY) < 1e-2)) {
        const fxyVal = analysis.evalF(cleanX, cleanY);
        if (Number.isFinite(fxyVal)) {
          found.push({
            x: cleanX,
            y: cleanY,
            fxy: fxyVal,
            hessian: analyzeHessian2D(analysis, cleanX, cleanY),
          });
        }
      }
    }
  }

  return found;
}


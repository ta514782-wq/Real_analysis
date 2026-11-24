import { parse, MathNode } from 'mathjs';

export const ALLOWED_VARIABLES = new Set(['x', 'y']);
export const ALLOWED_CONSTANTS = new Set(['e', 'pi', 'tau']);
export const ALLOWED_FUNCTIONS = new Set([
  'sin',
  'cos',
  'tan',
  'asin',
  'acos',
  'atan',
  'sinh',
  'cosh',
  'tanh',
  'exp',
  'log',
  'log10',
  'sqrt',
  'abs',
  'cbrt',
]);

export const ALLOWED_OPERATORS = new Set(['+', '-', '*', '/', '^']);

export interface ExpressionValidationResult {
  valid: boolean;
  normalized: string;
  variables: ('x' | 'y')[];
  dimension: 1 | 2;
  error?: string;
}

export interface OptimizerParamsInput {
  alpha: number;
  maxIterations: number;
  tolerance: number;
  x0: number;
  y0?: number;
}

export interface ParamsValidationResult {
  valid: boolean;
  error?: string;
}

/**
 * Normalizes user mathematical syntax safely without using eval().
 * Converts Python-style `**` to `^` and `ln(` to `log(`.
 */
export function normalizeExpressionInput(raw: string): string {
  return raw
    .trim()
    .replace(/\*\*/g, '^')
    .replace(/\bln\s*\(/g, 'log(');
}

/**
 * Validates a mathematical expression using mathjs AST traversal.
 * Strictly prohibits arbitrary code execution or disallowed symbols.
 */
export function validateExpression(
  raw: string,
  options: { requireVariable?: boolean; allowedDimensions?: (1 | 2)[] } = {}
): ExpressionValidationResult {
  const { requireVariable = true, allowedDimensions = [1, 2] } = options;

  if (!raw || typeof raw !== 'string' || raw.trim().length === 0) {
    return {
      valid: false,
      normalized: '',
      variables: [],
      dimension: 1,
      error: 'Expression cannot be empty.',
    };
  }

  if (raw.length > 240) {
    return {
      valid: false,
      normalized: '',
      variables: [],
      dimension: 1,
      error: 'Expression exceeds maximum length of 240 characters.',
    };
  }

  // Reject any code-like punctuation before parsing
  if (/[;={}[\]$`"'\\<>!?&|]/.test(raw)) {
    return {
      valid: false,
      normalized: '',
      variables: [],
      dimension: 1,
      error: 'Expression contains unsupported characters or assignment/code tokens.',
    };
  }

  const normalized = normalizeExpressionInput(raw);

  let ast: MathNode;
  try {
    ast = parse(normalized);
  } catch (err) {
    return {
      valid: false,
      normalized,
      variables: [],
      dimension: 1,
      error: `Syntax error: ${err instanceof Error ? err.message : 'Invalid mathematical expression.'}`,
    };
  }

  const foundVars = new Set<'x' | 'y'>();
  let validationError: string | undefined;

  ast.traverse((node: MathNode, path: string | null) => {
    if (validationError) return;

    switch (node.type) {
      case 'ConstantNode':
      case 'ParenthesisNode':
        break;

      case 'OperatorNode': {
        const opNode = node as unknown as { op: string };
        if (!ALLOWED_OPERATORS.has(opNode.op)) {
          validationError = `Unsupported operator "${opNode.op}". Allowed: +, -, *, /, ^.`;
        }
        break;
      }

      case 'FunctionNode': {
        const fnNode = node as unknown as { fn: { name?: string } };
        const fnName = fnNode.fn?.name ?? '';
        if (!ALLOWED_FUNCTIONS.has(fnName)) {
          validationError = `Unsupported function "${fnName}". Allowed: ${Array.from(ALLOWED_FUNCTIONS).join(', ')}.`;
        }
        break;
      }

      case 'SymbolNode': {
        // Function names are also SymbolNodes at path === 'fn'
        if (path === 'fn') break;
        const symNode = node as unknown as { name: string };
        const name = symNode.name;
        if (ALLOWED_VARIABLES.has(name)) {
          foundVars.add(name as 'x' | 'y');
        } else if (!ALLOWED_CONSTANTS.has(name)) {
          validationError = `Unknown symbol "${name}". Use variables x, y or constants e, pi, tau.`;
        }
        break;
      }

      default:
        validationError = `Disallowed expression construct "${node.type}".`;
    }
  });

  if (validationError) {
    return {
      valid: false,
      normalized,
      variables: Array.from(foundVars),
      dimension: foundVars.has('y') ? 2 : 1,
      error: validationError,
    };
  }

  const variables = Array.from(foundVars).sort() as ('x' | 'y')[];
  if (requireVariable && variables.length === 0) {
    return {
      valid: false,
      normalized,
      variables,
      dimension: 1,
      error: 'Expression must depend on at least one variable (x or y).',
    };
  }

  const dimension: 1 | 2 = variables.includes('y') ? 2 : 1;
  if (!allowedDimensions.includes(dimension)) {
    return {
      valid: false,
      normalized,
      variables,
      dimension,
      error:
        dimension === 2
          ? 'This module expects a single-variable function f(x), but variable "y" was found.'
          : 'This module expects a multivariable function f(x, y).',
    };
  }

  return {
    valid: true,
    normalized,
    variables,
    dimension,
  };
}

/**
 * Validates optimization hyperparameters and initial coordinates.
 */
export function validateOptimizerParams(params: OptimizerParamsInput): ParamsValidationResult {
  const { alpha, maxIterations, tolerance, x0, y0 } = params;

  if (!Number.isFinite(alpha) || alpha <= 0) {
    return {
      valid: false,
      error: 'Learning rate (alpha) must be a strictly positive finite number (alpha > 0).',
    };
  }
  if (alpha > 100) {
    return {
      valid: false,
      error: 'Learning rate (alpha) exceeds safe exploratory limit (alpha <= 100).',
    };
  }

  if (!Number.isInteger(maxIterations) || maxIterations < 1 || maxIterations > 10000) {
    return {
      valid: false,
      error: 'Maximum iterations must be an integer between 1 and 10,000.',
    };
  }

  if (!Number.isFinite(tolerance) || tolerance <= 0 || tolerance > 1) {
    return {
      valid: false,
      error: 'Stopping tolerance must be in the interval (0, 1].',
    };
  }

  if (!Number.isFinite(x0) || Math.abs(x0) > 1e6) {
    return {
      valid: false,
      error: 'Initial coordinate x0 must be a finite number in [-1e6, 1e6].',
    };
  }

  if (y0 !== undefined && (!Number.isFinite(y0) || Math.abs(y0) > 1e6)) {
    return {
      valid: false,
      error: 'Initial coordinate y0 must be a finite number in [-1e6, 1e6].',
    };
  }

  return { valid: true };
}

import { CheckCircle2, HelpCircle, RotateCcw, XCircle } from 'lucide-react';
import { useMemo, useState } from 'react';
import { MathTex } from '../components/MathTex';
import { MetricCard } from '../components/MetricCard';

interface QuizQuestion {
  id: string;
  topic: 'Real Analysis' | 'Gradient Descent' | 'Multivariable & Hessian' | 'Machine Learning';
  prompt: string;
  latex?: string;
  options: string[];
  correctIndex: number;
  hint: string;
  explanation: string;
}

const QUESTIONS: QuizQuestion[] = [
  {
    id: 'Q1',
    topic: 'Real Analysis',
    prompt: 'For f(x) = x² and Gradient Descent x_{n+1} = x_n - α f\'(x_n) with x₀ = 10, for which range of α does x_n converge to 0?',
    latex: 'x_{n+1} = (1 - 2\\alpha) x_n',
    options: ['0 < α < 0.5 only', '0 < α < 1', '0 < α ≤ 2', 'All α > 0'],
    correctIndex: 1,
    hint: 'A geometric sequence r^n x₀ converges to 0 iff |r| < 1, where r = 1 - 2α.',
    explanation: '|1 - 2α| < 1 simplifies to -1 < 1 - 2α < 1, which yields 0 < α < 1. For 0.5 < α < 1 it oscillates while converging; at α ≥ 1 it fails to contract.',
  },
  {
    id: 'Q2',
    topic: 'Gradient Descent',
    prompt: 'With f(x) = x², x₀ = 10, and α = 1.1, what is x₁ and what happens to the sequence?',
    options: [
      'x₁ = 8.9, converges monotonically to 0',
      'x₁ = -12, diverges with alternating signs',
      'x₁ = -10, remains in a bounded 2-cycle',
      'x₁ = 12, diverges monotonically to +∞',
    ],
    correctIndex: 1,
    hint: 'Compute f\'(10) = 20 and substitute into x₁ = 10 - 1.1 * 20.',
    explanation: 'x₁ = 10 - 1.1(20) = -12. Because r = 1 - 2(1.1) = -1.2 has |r| > 1 and r < 0, iterates grow in magnitude while flipping sign every step.',
  },
  {
    id: 'Q3',
    topic: 'Real Analysis',
    prompt: 'Why can constant-step Gradient Descent fail to converge on f(x) = |x| even though f is continuous and convex?',
    options: [
      'f(x) = |x| has no global minimum',
      'f is non-differentiable at 0 and |f\'(x)| = 1 does not approach 0 near the minimum',
      'Continuous functions cannot be minimized numerically',
      'Its Hessian is strictly positive definite everywhere',
    ],
    correctIndex: 1,
    hint: 'Look at the size of the update step α |f\'(x)| as x approaches 0.',
    explanation: 'Because f\'(x) = ±1 for all x ≠ 0, the step size remains constant (α) regardless of proximity to 0, causing permanent oscillation around x* = 0 unless α decays.',
  },
  {
    id: 'Q4',
    topic: 'Multivariable & Hessian',
    prompt: 'If the Hessian H(x*, y*) at a critical point (∇f = 0) has eigenvalues λ₁ = 4 and λ₂ = -2, how is (x*, y*) classified?',
    options: [
      'Strict Local Minimum',
      'Strict Local Maximum',
      'Saddle Point (Indefinite Hessian)',
      'Degenerate Inflection Point',
    ],
    correctIndex: 2,
    hint: 'Check the signs of λ₁ and λ₂: are they both positive, both negative, or opposite?',
    explanation: 'Opposite-sign eigenvalues indicate upward curvature along one eigendirection (λ₁ > 0) and downward curvature along the orthogonal direction (λ₂ < 0), defining a saddle point.',
  },
  {
    id: 'Q5',
    topic: 'Multivariable & Hessian',
    prompt: 'Which direction gives the fastest instantaneous decrease of a differentiable function f(x, y) at a point with ∇f ≠ 0?',
    options: ['+∇f(x, y)', '-∇f(x, y)', 'Orthogonal to ∇f(x, y)', 'The eigenvector of smallest eigenvalue'],
    correctIndex: 1,
    hint: 'Minimize the directional derivative D_u f = ∇f · u = ||∇f|| cos(θ).',
    explanation: 'By Cauchy–Schwarz, ∇f · u is minimized when cos(θ) = -1, i.e. u points directly opposite the gradient (-∇f).',
  },
  {
    id: 'Q6',
    topic: 'Real Analysis',
    prompt: 'In numerical differentiation, why does central finite difference have higher accuracy than forward difference for moderate step h?',
    latex: 'D^c_h f(x) = \\frac{f(x+h)-f(x-h)}{2h}',
    options: [
      'Odd-order Taylor error terms cancel out, leaving O(h²) truncation error instead of O(h)',
      'It eliminates IEEE-754 floating-point round-off completely',
      'It requires only one function evaluation',
      'It uses second derivatives directly',
    ],
    correctIndex: 0,
    hint: 'Expand f(x+h) and f(x-h) to third order and subtract them.',
    explanation: 'In f(x+h) - f(x-h), the f(x) and h²f\'\'(x)/2 terms cancel, leaving 2h f\'(x) + O(h³); dividing by 2h yields O(h²) truncation error.',
  },
  {
    id: 'Q7',
    topic: 'Machine Learning',
    prompt: 'When training Linear Regression with Gradient Descent, why should Z-score standardization statistics (mean and std) be computed ONLY on the training split?',
    options: [
      'To make training MSE artificially lower',
      'To prevent data leakage from validation samples into model preprocessing',
      'Because validation samples cannot have negative values',
      'Standardization only works on 1D features',
    ],
    correctIndex: 1,
    hint: 'Think about what information from the test/validation set should be unseen during training.',
    explanation: 'Computing mean and variance across the combined dataset leaks validation distribution information into training. Statistics must be learned solely from training data and applied unchanged to validation.',
  },
  {
    id: 'Q8',
    topic: 'Machine Learning',
    prompt: 'What pattern on train/validation loss curves indicates overfitting as polynomial degree increases?',
    options: [
      'Both training MSE and validation MSE remain very high',
      'Training MSE decreases toward zero while validation MSE begins increasing',
      'Validation MSE is always lower than training MSE',
      'Gradient norm diverges to infinity',
    ],
    correctIndex: 1,
    hint: 'An overparameterized model memorizes noise in the training set.',
    explanation: 'High-degree polynomials fit random noise in the training set (very low training MSE) but generalize poorly to unseen validation points (rising validation MSE).',
  },
];

export function InteractiveAssessmentView() {
  const [topicFilter, setTopicFilter] = useState<string>('All');
  const [selectedAnswers, setSelectedAnswers] = useState<Record<string, number>>({});
  const [revealedHints, setRevealedHints] = useState<Record<string, boolean>>({});

  const filteredQuestions = useMemo(() => {
    if (topicFilter === 'All') return QUESTIONS;
    return QUESTIONS.filter((q) => q.topic === topicFilter);
  }, [topicFilter]);

  const stats = useMemo(() => {
    let answered = 0;
    let correct = 0;
    for (const q of QUESTIONS) {
      if (selectedAnswers[q.id] !== undefined) {
        answered++;
        if (selectedAnswers[q.id] === q.correctIndex) correct++;
      }
    }
    return {
      answered,
      correct,
      total: QUESTIONS.length,
      pct: answered > 0 ? Math.round((correct / answered) * 100) : 0,
    };
  }, [selectedAnswers]);

  const topics = ['All', 'Real Analysis', 'Gradient Descent', 'Multivariable & Hessian', 'Machine Learning'];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="font-display text-lg font-semibold text-slate-100">
            Interactive Assessment · MATH 303 & Optimization
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            Test conceptual mastery with step-by-step hints, instant verification, and topic-wise score tracking.
          </p>
        </div>

        <button
          type="button"
          onClick={() => {
            setSelectedAnswers({});
            setRevealedHints({});
          }}
          className="flex items-center gap-1.5 rounded-md border border-slate-700 bg-[#0F141D] px-3 py-2 text-xs font-medium text-slate-300 hover:border-slate-600 hover:text-slate-100"
        >
          <RotateCcw className="h-3.5 w-3.5" />
          Reset Answers
        </button>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <MetricCard
          label="Questions Answered"
          value={`${stats.answered} / ${stats.total}`}
          tone="cyan"
        />
        <MetricCard
          label="Correct Answers"
          value={String(stats.correct)}
          detail={stats.answered > 0 ? `${stats.pct}% accuracy on attempted` : 'Select an option below'}
          tone={stats.correct > 0 ? 'emerald' : 'default'}
        />
        <MetricCard
          label="Topic Filter"
          value={topicFilter}
          detail={`${filteredQuestions.length} questions shown`}
        />
      </div>

      {/* Topic Segmented Filter */}
      <div className="flex flex-wrap gap-1.5">
        {topics.map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTopicFilter(t)}
            className={`rounded-md border px-3 py-1.5 text-xs font-medium transition-colors ${
              topicFilter === t
                ? 'border-cyan-500/60 bg-cyan-500/15 text-cyan-300'
                : 'border-slate-800 bg-[#0F141D] text-slate-400 hover:text-slate-200'
            }`}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="space-y-4">
        {filteredQuestions.map((q) => {
          const chosen = selectedAnswers[q.id];
          const isAnswered = chosen !== undefined;
          const isCorrect = chosen === q.correctIndex;
          const hintOpen = !!revealedHints[q.id];

          return (
            <div
              key={q.id}
              className="rounded-lg border border-slate-800 bg-[#0F141D] p-5 space-y-3"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="text-xs font-mono text-slate-400">
                  {q.id} · {q.topic}
                </div>
                <button
                  type="button"
                  onClick={() =>
                    setRevealedHints((prev) => ({ ...prev, [q.id]: !prev[q.id] }))
                  }
                  className="flex items-center gap-1 text-xs text-amber-400 hover:text-amber-300"
                >
                  <HelpCircle className="h-3.5 w-3.5" />
                  {hintOpen ? 'Hide Hint' : 'Show Hint'}
                </button>
              </div>

              <div className="text-sm font-medium text-slate-100">{q.prompt}</div>

              {q.latex && (
                <div className="rounded border border-slate-800 bg-[#0B0F17] px-3 py-2">
                  <MathTex block tex={q.latex} />
                </div>
              )}

              {hintOpen && (
                <div className="rounded border border-amber-500/30 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">
                  <span className="font-semibold">Hint: </span>
                  {q.hint}
                </div>
              )}

              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {q.options.map((opt, idx) => {
                  const isThisChosen = chosen === idx;
                  const isThisCorrect = idx === q.correctIndex;
                  let cls =
                    'border-slate-800 bg-[#0B0F17] text-slate-300 hover:border-slate-700 hover:text-slate-100';
                  if (isAnswered) {
                    if (isThisCorrect) {
                      cls = 'border-emerald-500/60 bg-emerald-950/30 text-emerald-200';
                    } else if (isThisChosen) {
                      cls = 'border-rose-500/60 bg-rose-950/30 text-rose-200';
                    } else {
                      cls = 'border-slate-800/60 bg-[#0B0F17]/50 text-slate-500';
                    }
                  }

                  return (
                    <button
                      key={opt}
                      type="button"
                      onClick={() =>
                        setSelectedAnswers((prev) => ({ ...prev, [q.id]: idx }))
                      }
                      className={`rounded border px-3 py-2.5 text-left text-xs transition-colors ${cls}`}
                    >
                      <span className="mr-2 font-mono text-slate-500">
                        {String.fromCharCode(65 + idx)}.
                      </span>
                      {opt}
                    </button>
                  );
                })}
              </div>

              {isAnswered && (
                <div
                  className={`flex items-start gap-2 rounded border px-3 py-2.5 text-xs ${
                    isCorrect
                      ? 'border-emerald-500/30 bg-emerald-950/20 text-emerald-200'
                      : 'border-rose-500/30 bg-rose-950/20 text-rose-200'
                  }`}
                >
                  {isCorrect ? (
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                  ) : (
                    <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" />
                  )}
                  <div>
                    <span className="font-semibold">
                      {isCorrect ? 'Correct: ' : 'Incorrect: '}
                    </span>
                    {q.explanation}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

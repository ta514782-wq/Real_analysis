import { Play } from 'lucide-react';
import { useMemo, useState } from 'react';
import { MetricCard } from '../components/MetricCard';
import { runAllVerificationTests } from '../tests/test_optimizers';

export function VerificationCenterView() {
  const [runCount, setRunCount] = useState(1);

  const testResults = useMemo(() => {
    void runCount;
    return runAllVerificationTests();
  }, [runCount]);

  const passedCount = testResults.filter((t) => t.passed).length;
  const totalCount = testResults.length;
  const totalMs = testResults.reduce((acc, t) => acc + t.durationMs, 0);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="font-display text-lg font-semibold text-slate-100">
            Stage 1 Mathematical Verification Center
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            Live execution of automated unit tests covering Test A/B/C convergence, divergence,
            multivariable optimization, input validation, symbolic/numerical derivatives, and Hessian classification.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setRunCount((c) => c + 1)}
          className="flex items-center gap-2 rounded-md bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 transition-colors hover:bg-cyan-400 whitespace-nowrap"
        >
          <Play className="h-3.5 w-3.5" />
          Re-Run Test Suite
        </button>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <MetricCard
          label="Suite Status"
          value={
            passedCount === totalCount
              ? `● ${passedCount}/${totalCount} Passed`
              : `✖ ${passedCount}/${totalCount} Passed`
          }
          tone={passedCount === totalCount ? 'emerald' : 'rose'}
          detail="Live computed results (no simulated outputs)"
        />
        <MetricCard
          label="Execution Time"
          value={totalMs.toFixed(1)}
          unit="ms"
          tone="cyan"
          detail={`Run #${runCount}`}
        />
        <MetricCard
          label="Covered Subsystems"
          value={String(new Set(testResults.map((t) => t.category)).size)}
          unit="domains"
          detail={Array.from(new Set(testResults.map((t) => t.category))).join(' · ')}
        />
      </div>

      <div className="overflow-hidden rounded-lg border border-slate-800 bg-[#0F141D]">
        <table className="w-full text-left text-xs">
          <thead className="bg-[#0B0F17] text-slate-400">
            <tr>
              <th className="px-4 py-3 font-medium">ID</th>
              <th className="px-4 py-3 font-medium">Verification Scenario</th>
              <th className="px-4 py-3 font-medium">Expected Behavior</th>
              <th className="px-4 py-3 font-medium">Actual Computed Output</th>
              <th className="px-4 py-3 font-medium">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/80">
            {testResults.map((t) => (
              <tr key={t.id} className="hover:bg-slate-800/30">
                <td className="px-4 py-3 font-mono text-slate-400 whitespace-nowrap">
                  {t.id}
                </td>
                <td className="px-4 py-3">
                  <div className="font-medium text-slate-100">{t.name}</div>
                  <div className="mt-0.5 text-slate-500">
                    {t.category} · {t.durationMs.toFixed(1)} ms
                  </div>
                </td>
                <td className="px-4 py-3 text-slate-300">{t.expected}</td>
                <td className="px-4 py-3 font-mono text-slate-200 tabular-nums">
                  {t.actualOutput}
                </td>
                <td className="px-4 py-3 font-mono whitespace-nowrap">
                  <span className={t.passed ? 'text-emerald-400' : 'text-rose-400'}>
                    {t.passed ? '● PASS' : '✖ FAIL'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

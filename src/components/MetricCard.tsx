interface MetricCardProps {
  label: string;
  value: string;
  unit?: string;
  detail?: string;
  tone?: 'default' | 'cyan' | 'emerald' | 'amber' | 'rose';
}

const TONE_VALUE_CLASS: Record<NonNullable<MetricCardProps['tone']>, string> = {
  default: 'text-slate-100',
  cyan: 'text-cyan-400',
  emerald: 'text-emerald-400',
  amber: 'text-amber-400',
  rose: 'text-rose-400',
};

export function MetricCard({
  label,
  value,
  unit,
  detail,
  tone = 'default',
}: MetricCardProps) {
  return (
    <div className="rounded-lg border border-slate-800 bg-[#0F141D] p-4">
      <div className="text-xs font-medium tracking-wide text-slate-400">{label}</div>
      <div className="mt-1.5 flex items-baseline gap-1.5">
        <span
          className={`font-mono text-2xl font-semibold tabular-nums tracking-tight ${TONE_VALUE_CLASS[tone]}`}
        >
          {value}
        </span>
        {unit && (
          <span className="font-mono text-xs text-slate-500">{unit}</span>
        )}
      </div>
      {detail && (
        <div className="mt-1.5 truncate font-mono text-xs text-slate-400">
          {detail}
        </div>
      )}
    </div>
  );
}

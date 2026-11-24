import Plotly from 'plotly.js-dist-min';
import { useEffect, useRef } from 'react';

interface PlotlyChartProps {
  data: Plotly.Data[];
  layout?: Partial<Plotly.Layout>;
  config?: Partial<Plotly.Config>;
  className?: string;
  height?: number;
  chartId?: string;
}

const BASE_DARK_LAYOUT: Partial<Plotly.Layout> = {
  paper_bgcolor: '#0F141D',
  plot_bgcolor: '#0B0F17',
  font: {
    family: 'IBM Plex Sans, system-ui, sans-serif',
    color: '#CBD5E1',
    size: 12,
  },
  margin: { t: 36, r: 24, b: 48, l: 56 },
  xaxis: {
    gridcolor: '#1E293B',
    zerolinecolor: '#334155',
    tickfont: { family: 'JetBrains Mono, monospace', size: 11, color: '#94A3B8' },
  },
  yaxis: {
    gridcolor: '#1E293B',
    zerolinecolor: '#334155',
    tickfont: { family: 'JetBrains Mono, monospace', size: 11, color: '#94A3B8' },
  },
  legend: {
    bgcolor: 'rgba(15, 20, 29, 0.85)',
    bordercolor: '#1E293B',
    borderwidth: 1,
    font: { size: 11, color: '#CBD5E1' },
  },
};

export function PlotlyChart({
  data,
  layout,
  config,
  className = '',
  height = 420,
  chartId,
}: PlotlyChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const mergedLayout: Partial<Plotly.Layout> = {
      ...BASE_DARK_LAYOUT,
      ...layout,
      xaxis: { ...BASE_DARK_LAYOUT.xaxis, ...(layout?.xaxis ?? {}) },
      yaxis: { ...BASE_DARK_LAYOUT.yaxis, ...(layout?.yaxis ?? {}) },
      margin: { ...BASE_DARK_LAYOUT.margin, ...(layout?.margin ?? {}) },
      autosize: true,
      height,
    };

    const mergedConfig: Partial<Plotly.Config> = {
      responsive: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      ...config,
    };

    Plotly.react(el, data, mergedLayout, mergedConfig);

    const observer = new ResizeObserver(() => {
      if (el) {
        Plotly.Plots.resize(el);
      }
    });
    observer.observe(el);

    return () => {
      observer.disconnect();
    };
  }, [data, layout, config, height]);

  useEffect(() => {
    const el = containerRef.current;
    return () => {
      if (el) {
        Plotly.purge(el);
      }
    };
  }, []);

  return (
    <div
      id={chartId}
      ref={containerRef}
      style={{ height }}
      className={`w-full overflow-hidden rounded-lg border border-slate-800 bg-[#0F141D] ${className}`}
    />
  );
}

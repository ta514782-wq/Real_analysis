import katex from 'katex';
import { useMemo } from 'react';

interface MathTexProps {
  tex: string;
  block?: boolean;
  className?: string;
}

export function MathTex({ tex, block = false, className = '' }: MathTexProps) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(tex, {
        displayMode: block,
        throwOnError: false,
        strict: false,
      });
    } catch {
      return tex;
    }
  }, [tex, block]);

  if (block) {
    return (
      <div
        className={`overflow-x-auto py-1 text-slate-100 ${className}`}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }

  return (
    <span
      className={`text-slate-100 ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

type SparkProps = {
  seriesA: number[];
  seriesB?: number[];
  height?: number;
  /** Accessible description, e.g. "Updates per round, latest 12". */
  label: string;
};

/** Dual-line sparkline with an accessible text alternative. */
export function Sparkline({ seriesA, seriesB, height = 120, label }: SparkProps) {
  const w = 320;
  const h = height;
  const pad = 8;
  const all = [...seriesA, ...(seriesB ?? [])];
  const max = Math.max(...all, 1);
  const min = 0;
  const span = max - min || 1;

  function path(data: number[]): string {
    if (data.length === 0) return "";
    return data
      .map((v, i) => {
        const x = pad + (i / Math.max(data.length - 1, 1)) * (w - pad * 2);
        const y = h - pad - ((v - min) / span) * (h - pad * 2);
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }

  const latest = seriesA.length ? seriesA[seriesA.length - 1] : 0;
  const description = `${label}. Latest value ${latest}, maximum ${max}, ${seriesA.length} points.`;

  if (seriesA.length === 0) {
    return <div className="spark-empty">No data yet</div>;
  }

  return (
    <svg
      className="spark"
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      role="img"
      aria-label={description}
    >
      <path
        d={path(seriesA)}
        fill="none"
        stroke="var(--chart-blue)"
        strokeWidth="2"
        vectorEffect="non-scaling-stroke"
      />
      {seriesB && (
        <path
          d={path(seriesB)}
          fill="none"
          stroke="var(--chart-orange)"
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
      )}
    </svg>
  );
}

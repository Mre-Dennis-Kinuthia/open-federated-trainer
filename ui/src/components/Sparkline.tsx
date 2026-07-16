type SparkProps = {
  seriesA: number[];
  seriesB?: number[];
  height?: number;
};

/** Simple dual-line sparkline matching Vercel observability cards. */
export function Sparkline({ seriesA, seriesB, height = 120 }: SparkProps) {
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

  return (
    <svg className="spark" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
      <path
        d={path(seriesA)}
        fill="none"
        stroke="#0070f3"
        strokeWidth="2"
        vectorEffect="non-scaling-stroke"
      />
      {seriesB && (
        <path
          d={path(seriesB)}
          fill="none"
          stroke="#f5a623"
          strokeWidth="2"
          vectorEffect="non-scaling-stroke"
        />
      )}
    </svg>
  );
}

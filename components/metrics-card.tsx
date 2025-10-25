interface MetricsCardProps {
  title: string
  value: string
  unit?: string
}

export function MetricsCard({ title, value, unit }: MetricsCardProps) {
  return (
    <div className="bg-card border border-border rounded-lg p-6">
      <p className="text-sm text-muted-foreground mb-2">{title}</p>
      <div className="flex items-baseline gap-2">
        <span className="text-3xl font-bold text-accent">{value}</span>
        {unit && <span className="text-sm text-muted-foreground">{unit}</span>}
      </div>
    </div>
  )
}

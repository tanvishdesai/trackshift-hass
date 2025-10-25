"use client"
import { ArrowLeft, Download, Save, RotateCcw, TrendingUp, Zap, Target } from "lucide-react"

interface ResultsDashboardProps {
  image1: string
  image2: string
  date1: string
  date2: string
  onBackClick: () => void
}

export function ResultsDashboard({ image1, image2, date1, date2, onBackClick }: ResultsDashboardProps) {
  return (
    <main className="px-6 py-8">
      <button
        onClick={onBackClick}
        className="flex items-center gap-2 text-accent hover:text-accent/80 transition-colors mb-8"
      >
        <ArrowLeft className="w-4 h-4" />
        Back to Upload
      </button>

      {/* Results Dashboard */}
      <div className="space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-4xl font-bold text-foreground mb-2">Analysis Results</h1>
          <p className="text-muted-foreground">Temporal change detection complete • High confidence analysis</p>
        </div>

        {/* Image Comparison Section */}
        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-foreground">Image Comparison</h2>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Past Image */}
            <div className="bg-card border border-border rounded-lg p-4 hover:border-accent/50 transition-all hover:shadow-lg hover:shadow-accent/10">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                <h3 className="font-semibold text-foreground">Past (Baseline)</h3>
              </div>
              <div className="bg-secondary rounded-lg overflow-hidden h-64 mb-3">
                {image1 && <img src={image1 || "/placeholder.svg"} alt="Past" className="w-full h-full object-cover" />}
              </div>
              <p className="text-sm text-muted-foreground">{date1 || "Date not specified"}</p>
            </div>

            {/* Present Image */}
            <div className="bg-card border border-border rounded-lg p-4 hover:border-accent/50 transition-all hover:shadow-lg hover:shadow-accent/10">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <h3 className="font-semibold text-foreground">Present (Current)</h3>
              </div>
              <div className="bg-secondary rounded-lg overflow-hidden h-64 mb-3">
                {image2 && (
                  <img src={image2 || "/placeholder.svg"} alt="Present" className="w-full h-full object-cover" />
                )}
              </div>
              <p className="text-sm text-muted-foreground">{date2 || "Date not specified"}</p>
            </div>

            {/* Heatmap */}
            <div className="bg-card border border-border rounded-lg p-4 hover:border-accent/50 transition-all hover:shadow-lg hover:shadow-accent/10">
              <div className="flex items-center gap-2 mb-3">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <h3 className="font-semibold text-foreground">Change Detection</h3>
              </div>
              <div className="bg-secondary rounded-lg overflow-hidden h-64 mb-3 relative">
                {image2 && (
                  <>
                    <img src={image2 || "/placeholder.svg"} alt="Heatmap" className="w-full h-full object-cover" />
                    <div className="absolute inset-0 bg-red-500/30 mix-blend-multiply"></div>
                  </>
                )}
              </div>
              <p className="text-sm text-muted-foreground">AI-detected changes highlighted in red</p>
            </div>
          </div>
        </div>

        {/* Key Metrics Section */}
        <div className="space-y-4">
          <h2 className="text-xl font-semibold text-foreground">Key Metrics</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Total Area Changed */}
            <div className="bg-card border border-border rounded-lg p-6 hover:border-accent/50 transition-all hover:shadow-lg hover:shadow-accent/10">
              <div className="flex items-center justify-between mb-4">
                <p className="text-sm font-medium text-muted-foreground">Total Area Changed</p>
                <TrendingUp className="w-5 h-5 text-accent" />
              </div>
              <p className="text-3xl font-bold text-accent mb-2">1.45 km²</p>
              <p className="text-xs text-muted-foreground">Detected change area</p>
            </div>

            {/* Percentage Change */}
            <div className="bg-card border border-border rounded-lg p-6 hover:border-accent/50 transition-all hover:shadow-lg hover:shadow-accent/10">
              <div className="flex items-center justify-between mb-4">
                <p className="text-sm font-medium text-muted-foreground">Percentage Change</p>
                <Zap className="w-5 h-5 text-accent" />
              </div>
              <p className="text-3xl font-bold text-accent mb-2">+5.8%</p>
              <p className="text-xs text-muted-foreground">Relative to baseline</p>
            </div>

            {/* Analysis Confidence */}
            <div className="bg-card border border-border rounded-lg p-6 hover:border-accent/50 transition-all hover:shadow-lg hover:shadow-accent/10">
              <div className="flex items-center justify-between mb-4">
                <p className="text-sm font-medium text-muted-foreground">Analysis Confidence</p>
                <Target className="w-5 h-5 text-accent" />
              </div>
              <p className="text-3xl font-bold text-accent mb-2">96.3%</p>
              <p className="text-xs text-muted-foreground">Model accuracy score</p>
            </div>
          </div>
        </div>

        {/* Detailed Analysis Section */}
        <div className="bg-card border border-border rounded-lg p-6">
          <h2 className="text-lg font-semibold text-foreground mb-4">Detailed Analysis</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="font-medium text-foreground mb-3">Change Categories</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg">
                  <span className="text-sm text-foreground">Vegetation Loss</span>
                  <span className="text-sm font-semibold text-accent">42%</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg">
                  <span className="text-sm text-foreground">Urban Development</span>
                  <span className="text-sm font-semibold text-accent">35%</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg">
                  <span className="text-sm text-foreground">Water Body Changes</span>
                  <span className="text-sm font-semibold text-accent">23%</span>
                </div>
              </div>
            </div>
            <div>
              <h3 className="font-medium text-foreground mb-3">Analysis Metadata</h3>
              <div className="space-y-2">
                <div className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg">
                  <span className="text-sm text-muted-foreground">Processing Time</span>
                  <span className="text-sm font-semibold text-foreground">2.4s</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg">
                  <span className="text-sm text-muted-foreground">Model Version</span>
                  <span className="text-sm font-semibold text-foreground">v2.1</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-secondary/50 rounded-lg">
                  <span className="text-sm text-muted-foreground">Resolution</span>
                  <span className="text-sm font-semibold text-foreground">30m/pixel</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 pt-4">
          <button className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-accent text-accent-foreground font-semibold rounded-lg hover:bg-accent/90 transition-all hover:shadow-lg hover:shadow-accent/20">
            <Download className="w-5 h-5" />
            Export Report (PDF)
          </button>
          <button className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-secondary text-foreground font-semibold rounded-lg hover:bg-secondary/80 transition-colors">
            <Save className="w-5 h-5" />
            Save Analysis
          </button>
          <button
            onClick={onBackClick}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 border border-border text-foreground font-semibold rounded-lg hover:bg-secondary/50 transition-colors"
          >
            <RotateCcw className="w-5 h-5" />
            Start New Analysis
          </button>
        </div>
      </div>
    </main>
  )
}

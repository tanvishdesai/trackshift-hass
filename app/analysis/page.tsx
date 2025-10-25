"use client"

import { useState } from "react"
import { Header } from "@/components/header"
import { UploadZone } from "@/components/upload-zone"
import { AnalysisModal } from "@/components/analysis-modal"
import { ResultsDashboard } from "@/components/results-dashboard"
import { Info } from "lucide-react"

export default function AnalysisPage() {
  const [image1, setImage1] = useState<string>()
  const [image2, setImage2] = useState<string>()
  const [date1, setDate1] = useState<string>()
  const [date2, setDate2] = useState<string>()
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [showResults, setShowResults] = useState(false)

  const handleImage1Select = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => setImage1(e.target?.result as string)
    reader.readAsDataURL(file)
  }

  const handleImage2Select = (file: File) => {
    const reader = new FileReader()
    reader.onload = (e) => setImage2(e.target?.result as string)
    reader.readAsDataURL(file)
  }

  const handleAnalyze = () => {
    if (image1 && image2) {
      setIsAnalyzing(true)
    }
  }

  const handleAnalysisComplete = () => {
    setIsAnalyzing(false)
    setShowResults(true)
  }

  if (showResults) {
    return (
      <div className="min-h-screen bg-background">
        <Header />
        <ResultsDashboard
          image1={image1 || ""}
          image2={image2 || ""}
          date1={date1 || ""}
          date2={date2 || ""}
          onBackClick={() => setShowResults(false)}
        />
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <Header />
      <AnalysisModal isOpen={isAnalyzing} onComplete={handleAnalysisComplete} />

      <main className="px-6 py-12">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Page Header */}
          <div>
            <h1 className="text-4xl font-bold text-foreground mb-2">New Analysis</h1>
            <p className="text-muted-foreground">
              Upload two satellite images of the same location from different times to detect environmental changes
            </p>
          </div>

          <div className="bg-accent/10 border border-accent/30 rounded-lg p-4 flex gap-3">
            <Info className="w-5 h-5 text-accent flex-shrink-0 mt-0.5" />
            <div className="text-sm text-foreground">
              <p className="font-semibold mb-1">Tips for best results:</p>
              <ul className="text-muted-foreground space-y-1 list-disc list-inside">
                <li>Use images of the same geographic location</li>
                <li>Ensure images have similar resolution and scale</li>
                <li>Supported formats: JPG, PNG, GeoTIFF</li>
              </ul>
            </div>
          </div>

          {/* Upload Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div className="bg-card border border-border rounded-lg p-8 hover:border-accent/50 transition-colors">
              <UploadZone
                title="Image 1: The Past (Baseline Image)"
                subtitle="Earlier satellite image"
                onImageSelect={handleImage1Select}
                onDateChange={setDate1}
                image={image1}
              />
            </div>

            <div className="bg-card border border-border rounded-lg p-8 hover:border-accent/50 transition-colors">
              <UploadZone
                title="Image 2: The Present (Current Image)"
                subtitle="Recent satellite image"
                onImageSelect={handleImage2Select}
                onDateChange={setDate2}
                image={image2}
              />
            </div>
          </div>

          {/* Analyze Button */}
          <div className="flex justify-center">
            <button
              onClick={handleAnalyze}
              disabled={!image1 || !image2}
              className={`px-12 py-4 font-semibold rounded-lg transition-all text-lg ${
                image1 && image2
                  ? "bg-accent text-accent-foreground hover:bg-accent/90 cursor-pointer shadow-lg shadow-accent/20 hover:shadow-accent/40"
                  : "bg-muted text-muted-foreground cursor-not-allowed opacity-50"
              }`}
            >
              Analyze Changes
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}

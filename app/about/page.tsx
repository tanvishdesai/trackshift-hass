"use client"

import { Header } from "@/components/header"
import { Zap, Eye, TrendingUp } from "lucide-react"

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="px-6 py-12">
        <div className="max-w-3xl mx-auto space-y-12">
          {/* Hero */}
          <div className="text-center space-y-4">
            <h1 className="text-4xl font-bold text-foreground">About EcoVision</h1>
            <p className="text-xl text-muted-foreground">Leveraging AI to understand our changing planet</p>
          </div>

          {/* Features */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-card border border-border rounded-lg p-6">
              <Eye className="w-8 h-8 text-accent mb-4" />
              <h3 className="text-lg font-semibold text-foreground mb-2">Advanced Vision</h3>
              <p className="text-muted-foreground">
                State-of-the-art deep learning models detect subtle changes in satellite imagery
              </p>
            </div>

            <div className="bg-card border border-border rounded-lg p-6">
              <Zap className="w-8 h-8 text-accent mb-4" />
              <h3 className="text-lg font-semibold text-foreground mb-2">Real-Time Analysis</h3>
              <p className="text-muted-foreground">
                Get instant results with our optimized neural networks and GPU acceleration
              </p>
            </div>

            <div className="bg-card border border-border rounded-lg p-6">
              <TrendingUp className="w-8 h-8 text-accent mb-4" />
              <h3 className="text-lg font-semibold text-foreground mb-2">Actionable Insights</h3>
              <p className="text-muted-foreground">
                Comprehensive metrics and visualizations to understand environmental changes
              </p>
            </div>
          </div>

          {/* Description */}
          <div className="bg-card border border-border rounded-lg p-8 space-y-4">
            <h2 className="text-2xl font-bold text-foreground">How It Works</h2>
            <p className="text-muted-foreground">
              EcoVision uses a Siamese ResNet34 architecture combined with transformer-based feature fusion to detect
              changes between satellite images. Our attention-gated decoder generates precise change masks, allowing you
              to visualize exactly where and how much has changed.
            </p>
            <p className="text-muted-foreground">
              Whether you're monitoring deforestation, urban sprawl, agricultural changes, or disaster recovery,
              EcoVision provides the precision and confidence metrics you need to make informed decisions.
            </p>
          </div>
        </div>
      </main>
    </div>
  )
}

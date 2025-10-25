"use client"

import { useEffect, useState } from "react"
import { Loader2, CheckCircle2 } from "lucide-react"

interface AnalysisModalProps {
  isOpen: boolean
  onComplete: () => void
}

export function AnalysisModal({ isOpen, onComplete }: AnalysisModalProps) {
  const [step, setStep] = useState(0)
  const [isComplete, setIsComplete] = useState(false)

  const steps = [
    {
      title: "Initializing Siamese ResNet34 Encoders",
      description: "Loading deep learning models for image feature extraction",
    },
    {
      title: "Fusing Deep Features with Transformer",
      description: "Applying attention mechanisms to identify temporal changes",
    },
    {
      title: "Generating Attention-Gated Decoder Map",
      description: "Creating detailed change detection heatmap",
    },
    {
      title: "Finalizing Change Mask & Calculating Metrics",
      description: "Computing area changes and confidence scores",
    },
  ]

  useEffect(() => {
    if (!isOpen) {
      setStep(0)
      setIsComplete(false)
      return
    }

    const interval = setInterval(() => {
      setStep((prev) => {
        if (prev >= steps.length - 1) {
          clearInterval(interval)
          setIsComplete(true)
          setTimeout(onComplete, 1000)
          return prev
        }
        return prev + 1
      })
    }, 1500)

    return () => clearInterval(interval)
  }, [isOpen, onComplete, steps.length])

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-card border border-border rounded-lg p-8 max-w-md w-full mx-4 shadow-2xl">
        {/* Animated Loading Spinner */}
        <div className="flex justify-center mb-8">
          <div className="relative w-28 h-28">
            {/* Outer rotating ring */}
            <div className="absolute inset-0 rounded-full border-4 border-accent/20"></div>
            <div className="absolute inset-0 rounded-full border-4 border-transparent border-t-accent border-r-accent animate-spin"></div>

            <div className="absolute inset-2 rounded-full border-2 border-accent/10 animate-pulse-glow"></div>

            {/* Center icon */}
            <Loader2 className="absolute inset-0 m-auto w-10 h-10 text-accent animate-spin" />
          </div>
        </div>

        {/* Title */}
        <h2 className="text-2xl font-bold text-center text-foreground mb-2">Analyzing Temporal Differences</h2>
        <p className="text-center text-muted-foreground text-sm mb-8">
          Processing satellite imagery with AI-powered change detection
        </p>

        {/* Status Updates */}
        <div className="space-y-3 mb-6">
          {steps.map((stepData, index) => (
            <div
              key={index}
              className={`flex items-start gap-3 p-4 rounded-lg transition-all duration-300 ${
                index < step
                  ? "bg-accent/10 border border-accent/30"
                  : index === step
                    ? "bg-accent/20 border border-accent/50 ring-1 ring-accent/30"
                    : "bg-secondary/30 border border-border"
              }`}
            >
              {/* Step Indicator */}
              <div
                className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold transition-all ${
                  index < step
                    ? "bg-accent text-accent-foreground"
                    : index === step
                      ? "bg-accent text-accent-foreground animate-pulse"
                      : "bg-muted text-muted-foreground"
                }`}
              >
                {index < step ? <CheckCircle2 className="w-5 h-5" /> : index + 1}
              </div>

              {/* Step Content */}
              <div className="flex-1 min-w-0">
                <p
                  className={`text-sm font-semibold transition-colors ${
                    index <= step ? "text-foreground" : "text-muted-foreground"
                  }`}
                >
                  {stepData.title}
                </p>
                <p
                  className={`text-xs mt-1 transition-colors ${
                    index <= step ? "text-muted-foreground" : "text-muted-foreground/60"
                  }`}
                >
                  {stepData.description}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-secondary rounded-full h-2 overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-accent to-accent/70 transition-all duration-300 rounded-full"
            style={{ width: `${((step + 1) / steps.length) * 100}%` }}
          ></div>
        </div>

        {/* Progress Text */}
        <p className="text-center text-xs text-muted-foreground mt-4">
          {isComplete ? "Analysis complete!" : `Step ${step + 1} of ${steps.length}`}
        </p>
      </div>
    </div>
  )
}

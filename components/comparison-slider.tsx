"use client"

import type React from "react"

import { useState, useRef } from "react"

interface ComparisonSliderProps {
  beforeImage: string
  afterImage: string
  beforeLabel: string
  afterLabel: string
}

export function ComparisonSlider({ beforeImage, afterImage, beforeLabel, afterLabel }: ComparisonSliderProps) {
  const [sliderPosition, setSliderPosition] = useState(50)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const newPosition = ((e.clientX - rect.left) / rect.width) * 100
    setSliderPosition(Math.max(0, Math.min(100, newPosition)))
  }

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const newPosition = ((e.touches[0].clientX - rect.left) / rect.width) * 100
    setSliderPosition(Math.max(0, Math.min(100, newPosition)))
  }

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onTouchMove={handleTouchMove}
      className="relative w-full h-96 rounded-lg overflow-hidden cursor-col-resize bg-secondary"
    >
      {/* After Image (Background) */}
      <img
        src={afterImage || "/placeholder.svg"}
        alt={afterLabel}
        className="absolute inset-0 w-full h-full object-cover"
      />

      {/* Before Image (Overlay) */}
      <div className="absolute inset-0 overflow-hidden" style={{ width: `${sliderPosition}%` }}>
        <img
          src={beforeImage || "/placeholder.svg"}
          alt={beforeLabel}
          className="absolute inset-0 w-full h-full object-cover"
          style={{ width: `${(100 / sliderPosition) * 100}%` }}
        />
      </div>

      {/* Slider Handle */}
      <div className="absolute top-0 bottom-0 w-1 bg-accent" style={{ left: `${sliderPosition}%` }}>
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-12 h-12 bg-accent rounded-full shadow-lg flex items-center justify-center">
          <div className="flex gap-1">
            <div className="w-0.5 h-4 bg-accent-foreground"></div>
            <div className="w-0.5 h-4 bg-accent-foreground"></div>
          </div>
        </div>
      </div>

      {/* Labels */}
      <div className="absolute top-4 left-4 bg-black/50 px-3 py-1 rounded text-sm text-white">{beforeLabel}</div>
      <div className="absolute top-4 right-4 bg-black/50 px-3 py-1 rounded text-sm text-white">{afterLabel}</div>
    </div>
  )
}

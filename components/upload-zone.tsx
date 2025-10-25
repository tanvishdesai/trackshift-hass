"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Upload } from "lucide-react"

interface UploadZoneProps {
  title: string
  subtitle: string
  onImageSelect: (file: File) => void
  onDateChange: (date: string) => void
  image?: string
}

export function UploadZone({ title, subtitle, onImageSelect, onDateChange, image }: UploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => {
    setIsDragging(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const files = e.dataTransfer.files
    if (files.length > 0) {
      onImageSelect(files[0])
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.currentTarget.files
    if (files && files.length > 0) {
      onImageSelect(files[0])
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div>
        <h3 className="text-lg font-semibold text-foreground mb-1">{title}</h3>
        <p className="text-sm text-muted-foreground">{subtitle}</p>
      </div>

      {/* Upload Area */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-all ${
          isDragging ? "border-accent bg-accent/10" : "border-border hover:border-accent/50 hover:bg-card/50"
        }`}
      >
        <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileSelect} className="hidden" />
        {image ? (
          <div className="flex flex-col items-center gap-2">
            <img src={image || "/placeholder.svg"} alt="Preview" className="w-32 h-32 object-cover rounded" />
            <p className="text-sm text-muted-foreground">Click to change image</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <Upload className="w-12 h-12 text-accent mx-auto" />
            <div>
              <p className="text-foreground font-medium">Drag & Drop Image or Click to Browse</p>
              <p className="text-sm text-muted-foreground mt-1">Supported formats: JPG, PNG, GeoTIFF</p>
            </div>
          </div>
        )}
      </div>

      {/* Date Input */}
      <div>
        <label className="block text-sm font-medium text-foreground mb-2">Date of Capture</label>
        <input
          type="date"
          onChange={(e) => onDateChange(e.target.value)}
          className="w-full px-4 py-2 bg-secondary border border-border rounded-lg text-foreground placeholder-muted-foreground focus:outline-none focus:ring-2 focus:ring-accent"
        />
      </div>
    </div>
  )
}

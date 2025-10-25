"use client"

import { Header } from "@/components/header"
import Link from "next/link"
import { ArrowRight, Zap } from "lucide-react"

export default function Home() {
  return (
    <div className="min-h-screen bg-background">
      <Header />

      <main className="flex flex-col items-center justify-center min-h-[calc(100vh-80px)] px-4 py-12">
        <div className="max-w-2xl text-center space-y-8">
          {/* Hero Section */}
          <div className="space-y-4">
            <div className="inline-block px-4 py-2 bg-accent/10 border border-accent/30 rounded-full">
              <span className="text-sm font-semibold text-accent">AI-Powered Analysis</span>
            </div>
            <h1 className="text-5xl md:text-6xl font-bold text-foreground leading-tight">
              Pinpoint Change with <span className="text-accent">AI Precision</span>
            </h1>
            <p className="text-xl text-muted-foreground">
              Upload two satellite images of the same location from different times to generate an exact map of the
              changes.
            </p>
          </div>

          {/* CTA Button */}
          <Link
            href="/analysis"
            className="inline-flex items-center gap-2 px-8 py-4 bg-accent text-accent-foreground font-semibold rounded-lg hover:bg-accent/90 transition-colors group"
          >
            Start Analysis
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Link>

          {/* Example Use Cases */}
          <div className="pt-8 border-t border-border">
            <p className="text-sm text-muted-foreground mb-6">Example Use Cases</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-card border border-border rounded-lg p-4 hover:border-accent/50 transition-colors">
                <div className="flex items-center gap-3 mb-2">
                  <Zap className="w-5 h-5 text-accent" />
                  <span className="font-semibold text-foreground">Deforestation in the Amazon</span>
                </div>
                <p className="text-sm text-muted-foreground">Track forest loss over time</p>
              </div>
              <div className="bg-card border border-border rounded-lg p-4 hover:border-accent/50 transition-colors">
                <div className="flex items-center gap-3 mb-2">
                  <Zap className="w-5 h-5 text-accent" />
                  <span className="font-semibold text-foreground">Urban Sprawl in Dubai</span>
                </div>
                <p className="text-sm text-muted-foreground">Monitor urban development</p>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}

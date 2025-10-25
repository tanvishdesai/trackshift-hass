"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { Eye, User, Menu, X } from "lucide-react"
import { useState } from "react"

export function Header() {
  const pathname = usePathname()
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  const isActive = (href: string) => {
    if (href === "/" && pathname === "/") return true
    if (href !== "/" && pathname.startsWith(href)) return true
    return false
  }

  const navLinks = [
    { href: "/", label: "Dashboard" },
    { href: "/analysis", label: "New Analysis" },
    { href: "/about", label: "About" },
  ]

  return (
    <header className="sticky top-0 z-50 border-b border-border bg-card/95 backdrop-blur-sm">
      <div className="flex items-center justify-between px-6 py-4 max-w-7xl mx-auto">
        {/* Logo and Brand */}
        <Link href="/" className="flex items-center gap-3 flex-shrink-0">
          <div className="flex items-center justify-center w-10 h-10 bg-accent rounded-lg hover:shadow-lg hover:shadow-accent/50 transition-all duration-300">
            <Eye className="w-6 h-6 text-accent-foreground" />
          </div>
          <span className="text-xl font-bold text-foreground hidden sm:inline">EcoVision</span>
        </Link>

        {/* Desktop Navigation Links */}
        <nav className="hidden md:flex items-center gap-1">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={`px-4 py-2 rounded-md transition-all duration-200 ${
                isActive(link.href)
                  ? "bg-accent text-accent-foreground font-semibold"
                  : "text-foreground hover:bg-secondary hover:text-accent"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>

        {/* Right Section */}
        <div className="flex items-center gap-4">
          {/* User Profile */}
          <button className="flex items-center justify-center w-10 h-10 rounded-full bg-muted hover:bg-secondary hover:text-accent transition-all duration-200">
            <User className="w-5 h-5 text-foreground" />
          </button>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden flex items-center justify-center w-10 h-10 rounded-lg bg-muted hover:bg-secondary transition-colors"
          >
            {mobileMenuOpen ? <X className="w-5 h-5 text-foreground" /> : <Menu className="w-5 h-5 text-foreground" />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation Menu */}
      {mobileMenuOpen && (
        <nav className="md:hidden border-t border-border bg-card px-6 py-4 space-y-2">
          {navLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              onClick={() => setMobileMenuOpen(false)}
              className={`block px-4 py-2 rounded-md transition-all duration-200 ${
                isActive(link.href)
                  ? "bg-accent text-accent-foreground font-semibold"
                  : "text-foreground hover:bg-secondary hover:text-accent"
              }`}
            >
              {link.label}
            </Link>
          ))}
        </nav>
      )}
    </header>
  )
}

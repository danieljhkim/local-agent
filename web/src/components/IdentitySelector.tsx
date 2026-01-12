/**
 * Identity selector dropdown component
 */

import { useState, useEffect } from 'react'
import { apiClient } from '../lib/api-client'
import type { IdentityInfo } from '../lib/types'

interface IdentitySelectorProps {
  onIdentityChange?: (identity: string) => void
}

export function IdentitySelector({ onIdentityChange }: IdentitySelectorProps) {
  const [identities, setIdentities] = useState<IdentityInfo[]>([])
  const [activeIdentity, setActiveIdentity] = useState<string>('')
  const [isOpen, setIsOpen] = useState(false)
  const [loading, setLoading] = useState(false)

  // Load identities on mount
  useEffect(() => {
    loadIdentities()
  }, [])

  const loadIdentities = async () => {
    try {
      const response = await apiClient.listIdentities()
      setIdentities(response.identities)
      setActiveIdentity(response.active)
    } catch (err) {
      console.error('Failed to load identities:', err)
    }
  }

  const handleSelectIdentity = async (name: string) => {
    if (name === activeIdentity) {
      setIsOpen(false)
      return
    }

    setLoading(true)
    try {
      await apiClient.setActiveIdentity(name)
      setActiveIdentity(name)
      setIsOpen(false)

      // Notify parent component
      if (onIdentityChange) {
        onIdentityChange(name)
      }
    } catch (err) {
      console.error('Failed to set active identity:', err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm transition-colors"
        disabled={loading}
      >
        <span className="font-medium">Identity:</span>
        <span className="text-gray-700">{activeIdentity || 'default'}</span>
        <span className="text-gray-400">{isOpen ? '▲' : '▼'}</span>
      </button>

      {isOpen && (
        <>
          {/* Backdrop to close dropdown */}
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />

          {/* Dropdown menu */}
          <div className="absolute top-full left-0 mt-1 w-64 bg-white border border-gray-300 rounded-lg shadow-lg z-20">
            <div className="max-h-80 overflow-y-auto">
              {identities.map((identity) => (
                <button
                  key={identity.name}
                  onClick={() => handleSelectIdentity(identity.name)}
                  className={`w-full px-4 py-2 text-left hover:bg-gray-100 flex items-center justify-between ${
                    identity.is_active ? 'bg-blue-50' : ''
                  }`}
                >
                  <div className="flex-1">
                    <div className="font-medium text-sm">{identity.name}</div>
                    {identity.is_builtin && (
                      <div className="text-xs text-gray-500">Built-in</div>
                    )}
                  </div>
                  {identity.is_active && (
                    <span className="text-blue-600 text-lg">✓</span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

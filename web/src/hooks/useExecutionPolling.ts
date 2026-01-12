/**
 * Hook for polling execution status
 */

import { useEffect, useState } from 'react'
import { apiClient } from '../lib/api-client'

export function useExecutionPolling(
  sessionId: string | null,
  isExecuting: boolean
) {
  const [status, setStatus] = useState<'idle' | 'executing' | 'completed' | 'error'>('idle')
  const [response, setResponse] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!sessionId || !isExecuting) return

    const intervalId = setInterval(async () => {
      try {
        const result = await apiClient.getStatus(sessionId)

        if (result.status === 'completed') {
          setStatus('completed')
          setResponse(result.response || null)
          clearInterval(intervalId)
        } else if (result.status === 'error') {
          setStatus('error')
          setError(result.error || 'Unknown error')
          clearInterval(intervalId)
        }
      } catch (err) {
        console.error('Failed to poll status:', err)
        setStatus('error')
        setError(err instanceof Error ? err.message : 'Polling failed')
        clearInterval(intervalId)
      }
    }, 500) // Poll every 500ms

    return () => clearInterval(intervalId)
  }, [sessionId, isExecuting])

  return { status, response, error }
}

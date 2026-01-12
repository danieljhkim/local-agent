/**
 * Hook for managing conversation threads
 */

import { useState, useEffect, useCallback } from 'react'
import { apiClient } from '../lib/api-client'
import type { ThreadInfo } from '../lib/types'

export function useThreads() {
  const [threads, setThreads] = useState<ThreadInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  /**
   * Load threads from API
   */
  const loadThreads = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const threadList = await apiClient.listThreads(50, 0)
      setThreads(threadList)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load threads')
    } finally {
      setLoading(false)
    }
  }, [])

  /**
   * Create a new thread
   */
  const createThread = useCallback(
    async (title?: string): Promise<string | null> => {
      try {
        const threadId = await apiClient.createThread(title)
        await loadThreads() // Refresh list
        return threadId
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to create thread')
        return null
      }
    },
    [loadThreads]
  )

  /**
   * Update thread title
   */
  const updateThread = useCallback(
    async (threadId: string, title: string): Promise<boolean> => {
      try {
        await apiClient.updateThread(threadId, title)
        await loadThreads() // Refresh list
        return true
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to update thread')
        return false
      }
    },
    [loadThreads]
  )

  /**
   * Delete a thread
   */
  const deleteThread = useCallback(
    async (threadId: string): Promise<boolean> => {
      try {
        await apiClient.deleteThread(threadId)
        await loadThreads() // Refresh list
        return true
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to delete thread')
        return false
      }
    },
    [loadThreads]
  )

  /**
   * Clear error state
   */
  const clearError = useCallback(() => {
    setError(null)
  }, [])

  // Load threads on mount
  useEffect(() => {
    loadThreads()
  }, [loadThreads])

  return {
    threads,
    loading,
    error,
    loadThreads,
    createThread,
    updateThread,
    deleteThread,
    clearError,
  }
}

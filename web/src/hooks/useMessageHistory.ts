/**
 * Hook for managing message history
 */

import { useCallback, useEffect, useState } from 'react'
import { apiClient } from '../lib/api-client'
import type { MessageInfo } from '../lib/types'

export function useMessageHistory(sessionId: string | null) {
  const [messages, setMessages] = useState<MessageInfo[]>([])
  const [loading, setLoading] = useState(false)

  const loadHistory = useCallback(async () => {
    if (!sessionId) return

    setLoading(true)
    try {
      const history = await apiClient.getHistory(sessionId)
      setMessages(history)
    } catch (error) {
      console.error('Failed to load history:', error)
    } finally {
      setLoading(false)
    }
  }, [sessionId])

  useEffect(() => {
    loadHistory()
  }, [loadHistory])

  const addMessage = useCallback((message: MessageInfo) => {
    setMessages((prev) => [...prev, message])
  }, [])

  return { messages, loading, addMessage, refreshHistory: loadHistory }
}

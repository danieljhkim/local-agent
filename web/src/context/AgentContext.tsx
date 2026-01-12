/**
 * Global agent state management with React Context
 */

import { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import { apiClient } from '../lib/api-client'
import type { ApprovalInfo, MessageInfo } from '../lib/types'
import { useExecutionPolling } from '../hooks/useExecutionPolling'
import { useApprovalPolling } from '../hooks/useApprovalPolling'
import { useMessageHistory } from '../hooks/useMessageHistory'

interface AgentContextType {
  // State
  sessionId: string | null
  executionStatus: 'idle' | 'executing' | 'completed' | 'error'
  messages: MessageInfo[]
  pendingApprovals: ApprovalInfo[]
  error: string | null

  // Actions
  sendMessage: (message: string) => Promise<void>
  approveRequest: (approvalId: string, reason?: string) => Promise<void>
  denyRequest: (approvalId: string, reason: string) => Promise<void>
  clearError: () => void
}

const AgentContext = createContext<AgentContextType | undefined>(undefined)

export function AgentProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [executionStatus, setExecutionStatus] = useState<
    'idle' | 'executing' | 'completed' | 'error'
  >('idle')
  const [error, setError] = useState<string | null>(null)

  // Use custom hooks
  const { messages, addMessage, refreshHistory } = useMessageHistory(sessionId)
  const polling = useExecutionPolling(sessionId, executionStatus === 'executing')
  const { approvals } = useApprovalPolling(sessionId, executionStatus === 'executing')

  // Create session on mount
  useEffect(() => {
    let currentSessionId: string | null = null

    const initSession = async () => {
      try {
        const id = await apiClient.createSession()
        currentSessionId = id
        setSessionId(id)
      } catch (err) {
        console.error('Failed to create session:', err)
        setError(err instanceof Error ? err.message : 'Failed to create session')
      }
    }

    initSession()

    // Cleanup session on unmount
    return () => {
      if (currentSessionId) {
        apiClient.deleteSession(currentSessionId).catch(console.error)
      }
    }
  }, [])

  // Handle execution polling results
  useEffect(() => {
    if (polling.status === 'completed' && polling.response) {
      // Add assistant response to messages
      addMessage({ role: 'assistant', content: polling.response })
      setExecutionStatus('completed')
      // Refresh history to ensure sync
      setTimeout(() => refreshHistory(), 100)
    } else if (polling.status === 'error') {
      setError(polling.error || 'Execution failed')
      setExecutionStatus('error')
    }
  }, [polling.status, polling.response, polling.error, addMessage, refreshHistory])

  const sendMessage = async (message: string) => {
    if (!sessionId) {
      setError('No active session')
      return
    }

    try {
      // Add user message to UI immediately
      addMessage({ role: 'user', content: message })

      // Start execution
      await apiClient.sendMessage(sessionId, message)
      setExecutionStatus('executing')
      setError(null)
    } catch (err) {
      console.error('Failed to send message:', err)
      setError(err instanceof Error ? err.message : 'Failed to send message')
      setExecutionStatus('error')
    }
  }

  const approveRequest = async (approvalId: string, reason?: string) => {
    if (!sessionId) {
      setError('No active session')
      return
    }

    try {
      await apiClient.approveRequest(sessionId, approvalId, reason)
      setError(null)
    } catch (err) {
      console.error('Failed to approve request:', err)
      setError(err instanceof Error ? err.message : 'Failed to approve request')
    }
  }

  const denyRequest = async (approvalId: string, reason: string) => {
    if (!sessionId) {
      setError('No active session')
      return
    }

    try {
      await apiClient.denyRequest(sessionId, approvalId, reason)
      setError(null)
    } catch (err) {
      console.error('Failed to deny request:', err)
      setError(err instanceof Error ? err.message : 'Failed to deny request')
    }
  }

  const clearError = () => setError(null)

  return (
    <AgentContext.Provider
      value={{
        sessionId,
        executionStatus,
        messages,
        pendingApprovals: approvals,
        error,
        sendMessage,
        approveRequest,
        denyRequest,
        clearError,
      }}
    >
      {children}
    </AgentContext.Provider>
  )
}

export function useAgentContext() {
  const context = useContext(AgentContext)
  if (!context) {
    throw new Error('useAgentContext must be used within AgentProvider')
  }
  return context
}

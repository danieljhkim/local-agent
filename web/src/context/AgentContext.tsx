/**
 * Global agent state management with React Context
 */

import { createContext, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { apiClient } from '../lib/api-client'
import type { ApprovalInfo, MessageInfo, ThreadInfo } from '../lib/types'
import { useExecutionPolling } from '../hooks/useExecutionPolling'
import { useApprovalPolling } from '../hooks/useApprovalPolling'
import { useMessageHistory } from '../hooks/useMessageHistory'
import { useThreads } from '../hooks/useThreads'

interface AgentContextType {
  // Session state
  sessionId: string | null
  executionStatus: 'idle' | 'executing' | 'completed' | 'error'
  messages: MessageInfo[]
  pendingApprovals: ApprovalInfo[]
  error: string | null

  // Thread state
  threads: ThreadInfo[]
  currentThreadId: string | null
  threadsLoading: boolean

  // Session actions
  sendMessage: (message: string) => Promise<void>
  approveRequest: (approvalId: string, reason?: string) => Promise<void>
  denyRequest: (approvalId: string, reason: string) => Promise<void>
  clearError: () => void

  // Thread actions
  createNewThread: () => Promise<void>
  selectThread: (threadId: string) => Promise<void>
  updateThreadTitle: (threadId: string, title: string) => Promise<void>
  deleteThread: (threadId: string) => Promise<void>
}

const AgentContext = createContext<AgentContextType | undefined>(undefined)

export function AgentProvider({ children }: { children: ReactNode }) {
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null)
  const [executionStatus, setExecutionStatus] = useState<
    'idle' | 'executing' | 'completed' | 'error'
  >('idle')
  const [error, setError] = useState<string | null>(null)

  // Use custom hooks
  const { messages, addMessage, refreshHistory } = useMessageHistory(sessionId)
  const polling = useExecutionPolling(sessionId, executionStatus === 'executing')
  const { approvals } = useApprovalPolling(sessionId, executionStatus === 'executing')
  const {
    threads,
    loading: threadsLoading,
    createThread,
    updateThread,
    deleteThread: deleteThreadApi,
    loadThreads,
  } = useThreads()

  // Create ephemeral session on mount (no thread)
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

  /**
   * Create a new thread and start a session for it
   */
  const createNewThread = async () => {
    try {
      const threadId = await createThread('New chat')
      if (threadId) {
        await selectThread(threadId)
      }
    } catch (err) {
      console.error('Failed to create new thread:', err)
      setError(err instanceof Error ? err.message : 'Failed to create thread')
    }
  }

  /**
   * Select a thread and create/switch to a session for it
   */
  const selectThread = async (threadId: string) => {
    try {
      // Clean up old session if exists
      if (sessionId) {
        await apiClient.deleteSession(sessionId).catch(console.error)
      }

      // Create new session linked to thread
      const newSessionId = await apiClient.createSession(undefined, undefined, threadId)
      setSessionId(newSessionId)
      setCurrentThreadId(threadId)
      setExecutionStatus('idle')
      setError(null)

      // Refresh history for the new thread
      setTimeout(() => refreshHistory(), 100)
    } catch (err) {
      console.error('Failed to select thread:', err)
      setError(err instanceof Error ? err.message : 'Failed to select thread')
    }
  }

  /**
   * Update thread title
   */
  const updateThreadTitle = async (threadId: string, title: string) => {
    try {
      await updateThread(threadId, title)
    } catch (err) {
      console.error('Failed to update thread:', err)
      setError(err instanceof Error ? err.message : 'Failed to update thread')
    }
  }

  /**
   * Delete a thread
   */
  const handleDeleteThread = async (threadId: string) => {
    try {
      // If deleting current thread, switch to ephemeral session
      if (threadId === currentThreadId) {
        const newSessionId = await apiClient.createSession()
        setSessionId(newSessionId)
        setCurrentThreadId(null)
        setExecutionStatus('idle')
      }

      await deleteThreadApi(threadId)
      await loadThreads()
    } catch (err) {
      console.error('Failed to delete thread:', err)
      setError(err instanceof Error ? err.message : 'Failed to delete thread')
    }
  }

  return (
    <AgentContext.Provider
      value={{
        sessionId,
        executionStatus,
        messages,
        pendingApprovals: approvals,
        error,
        threads,
        currentThreadId,
        threadsLoading,
        sendMessage,
        approveRequest,
        denyRequest,
        clearError,
        createNewThread,
        selectThread,
        updateThreadTitle,
        deleteThread: handleDeleteThread,
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

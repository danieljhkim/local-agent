/**
 * Hook for polling pending approvals
 */

import { useEffect, useState } from 'react'
import { apiClient } from '../lib/api-client'
import type { ApprovalInfo } from '../lib/types'

export function useApprovalPolling(
  sessionId: string | null,
  isExecuting: boolean
) {
  const [approvals, setApprovals] = useState<ApprovalInfo[]>([])

  useEffect(() => {
    if (!sessionId || !isExecuting) {
      setApprovals([])
      return
    }

    const intervalId = setInterval(async () => {
      try {
        const pending = await apiClient.getPendingApprovals(sessionId)
        setApprovals(pending)
      } catch (err) {
        console.error('Failed to poll approvals:', err)
      }
    }, 1000) // Poll every 1 second

    return () => {
      clearInterval(intervalId)
      setApprovals([])
    }
  }, [sessionId, isExecuting])

  return { approvals }
}

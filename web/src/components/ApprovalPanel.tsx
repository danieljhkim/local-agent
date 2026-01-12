/**
 * Floating approval panel displaying pending requests
 */

import { useState } from 'react'
import { ApprovalCard } from './ApprovalCard'
import type { ApprovalInfo } from '../lib/types'

interface ApprovalPanelProps {
  approvals: ApprovalInfo[]
  onApprove: (approvalId: string) => void
  onDeny: (approvalId: string, reason: string) => void
}

export function ApprovalPanel({ approvals, onApprove, onDeny }: ApprovalPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true)

  if (approvals.length === 0) return null

  return (
    <div className="fixed bottom-4 right-4 w-96 max-h-[600px] bg-white shadow-lg rounded-lg border border-gray-300">
      {/* Header with badge and toggle */}
      <div
        className="flex justify-between items-center p-4 bg-blue-600 text-white rounded-t-lg cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <h2 className="font-bold flex items-center gap-2">
          Pending Approvals
          <span className="bg-white text-blue-600 px-2 py-1 rounded-full text-sm">
            {approvals.length}
          </span>
        </h2>
        <button className="text-white text-xl font-bold">
          {isExpanded ? '−' : '+'}
        </button>
      </div>

      {/* Approval list */}
      {isExpanded && (
        <div className="p-4 space-y-3 overflow-y-auto max-h-[500px]">
          {approvals.map((approval) => (
            <ApprovalCard
              key={approval.approval_id}
              approval={approval}
              onApprove={onApprove}
              onDeny={onDeny}
            />
          ))}
        </div>
      )}
    </div>
  )
}

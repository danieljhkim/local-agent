/**
 * Individual approval request card with risk tier styling
 */

import { useState } from 'react'
import type { ApprovalInfo } from '../lib/types'

interface ApprovalCardProps {
  approval: ApprovalInfo
  onApprove: (approvalId: string) => void
  onDeny: (approvalId: string, reason: string) => void
}

export function ApprovalCard({ approval, onApprove, onDeny }: ApprovalCardProps) {
  const [showDenyReason, setShowDenyReason] = useState(false)
  const [reason, setReason] = useState('')

  const tierColors = {
    tier_0: 'border-green-500 bg-green-50',
    tier_1: 'border-yellow-500 bg-yellow-50',
    tier_2: 'border-red-500 bg-red-50',
  }

  const handleDeny = () => {
    if (!reason.trim()) {
      alert('Please provide a reason for denial')
      return
    }
    onDeny(approval.approval_id, reason)
    setShowDenyReason(false)
    setReason('')
  }

  return (
    <div
      className={`border-l-4 p-4 rounded-lg ${
        tierColors[approval.risk_tier as keyof typeof tierColors]
      }`}
    >
      <h3 className="font-bold text-lg">{approval.tool_name}</h3>
      <p className="text-sm text-gray-600 mb-2">{approval.description}</p>

      <div className="bg-gray-100 p-2 rounded text-xs mb-3">
        <pre className="overflow-x-auto">
          {JSON.stringify(approval.parameters, null, 2)}
        </pre>
      </div>

      {!showDenyReason ? (
        <div className="flex gap-2">
          <button
            onClick={() => onApprove(approval.approval_id)}
            className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700"
          >
            Approve
          </button>
          <button
            onClick={() => setShowDenyReason(true)}
            className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
          >
            Deny
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          <input
            type="text"
            placeholder="Reason for denial..."
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            className="w-full border px-3 py-2 rounded"
            autoFocus
          />
          <div className="flex gap-2">
            <button
              onClick={handleDeny}
              className="bg-red-600 text-white px-4 py-2 rounded"
            >
              Confirm Deny
            </button>
            <button
              onClick={() => setShowDenyReason(false)}
              className="bg-gray-300 px-4 py-2 rounded"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

/**
 * Chat header with session info and status indicator
 */

import { IdentitySelector } from './IdentitySelector'

interface ChatHeaderProps {
  sessionId: string | null
  status: 'idle' | 'executing' | 'completed' | 'error'
  approvalCount: number
  onIdentityChange?: (identity: string) => void
}

export function ChatHeader({
  sessionId,
  status,
  approvalCount,
  onIdentityChange,
}: ChatHeaderProps) {
  const statusConfig = {
    idle: { color: 'bg-gray-400', label: 'Idle' },
    executing: { color: 'bg-blue-500 animate-pulse', label: 'Executing' },
    completed: { color: 'bg-green-500', label: 'Completed' },
    error: { color: 'bg-red-500', label: 'Error' },
  }

  const { color, label } = statusConfig[status]

  return (
    <div className="bg-gray-100 border-b p-4 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <h1 className="text-xl font-bold">Local Agent</h1>
        <div className="flex items-center gap-2 text-sm text-gray-600">
          <div className={`w-3 h-3 rounded-full ${color}`} />
          <span>{label}</span>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <IdentitySelector onIdentityChange={onIdentityChange} />

        {approvalCount > 0 && (
          <div className="bg-orange-500 text-white px-3 py-1 rounded-full text-sm font-bold">
            {approvalCount} pending
          </div>
        )}
        {sessionId && (
          <div className="text-xs text-gray-500">
            Session: {sessionId.slice(0, 8)}...
          </div>
        )}
      </div>
    </div>
  )
}

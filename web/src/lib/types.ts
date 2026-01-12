/**
 * Type definitions matching backend Pydantic models
 */

export interface MessageInfo {
  role: string
  content: string
}

export interface ApprovalInfo {
  approval_id: string
  tool_name: string
  description: string
  risk_tier: string
  parameters: Record<string, any>
  created_at: string
}

export interface ExecutionStatusResponse {
  status: 'executing' | 'completed' | 'error'
  response?: string
  error?: string
}

export interface CreateSessionResponse {
  session_id: string
}

export interface ApprovalDecisionResponse {
  success: boolean
}

export interface MessageHistoryResponse {
  messages: MessageInfo[]
}

export interface PendingApprovalsResponse {
  approvals: ApprovalInfo[]
}

export interface ThreadInfo {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count?: number
}

export interface ListThreadsResponse {
  threads: ThreadInfo[]
}

export interface IdentityInfo {
  name: string
  is_builtin: boolean
  is_active: boolean
}

export interface ListIdentitiesResponse {
  identities: IdentityInfo[]
  active: string
}

export interface IdentityContentResponse {
  name: string
  content: string
  is_builtin: boolean
}

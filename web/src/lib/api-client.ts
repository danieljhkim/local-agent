/**
 * API client for Local Agent backend
 */

import type {
  ApprovalInfo,
  CreateSessionResponse,
  ExecutionStatusResponse,
  MessageInfo,
  MessageHistoryResponse,
  PendingApprovalsResponse,
  ThreadInfo,
  ListThreadsResponse,
  IdentityInfo,
  ListIdentitiesResponse,
  IdentityContentResponse,
} from './types'

export class AgentAPIClient {
  private baseURL: string

  constructor(baseURL: string = 'http://localhost:8000') {
    this.baseURL = baseURL
  }

  /**
   * Create a new agent session
   */
  async createSession(
    identity?: string,
    systemPrompt?: string,
    threadId?: string
  ): Promise<string> {
    const response = await fetch(`${this.baseURL}/api/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        identity: identity || null,
        system_prompt: systemPrompt || null,
        thread_id: threadId || null,
      }),
    })

    if (!response.ok) {
      throw new Error(`Failed to create session: ${response.statusText}`)
    }

    const data: CreateSessionResponse = await response.json()
    return data.session_id
  }

  /**
   * Delete an agent session
   */
  async deleteSession(sessionId: string): Promise<void> {
    const response = await fetch(`${this.baseURL}/api/sessions/${sessionId}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      throw new Error(`Failed to delete session: ${response.statusText}`)
    }
  }

  /**
   * Send a message and start execution
   */
  async sendMessage(
    sessionId: string,
    message: string,
    systemPrompt?: string
  ): Promise<void> {
    const response = await fetch(`${this.baseURL}/api/sessions/${sessionId}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        system_prompt: systemPrompt || null,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Failed to send message: ${response.statusText} - ${errorText}`)
    }
  }

  /**
   * Get execution status (for polling)
   */
  async getStatus(sessionId: string): Promise<ExecutionStatusResponse> {
    const response = await fetch(`${this.baseURL}/api/sessions/${sessionId}/status`)

    if (!response.ok) {
      throw new Error(`Failed to get status: ${response.statusText}`)
    }

    return await response.json()
  }

  /**
   * Get message history for a session
   */
  async getHistory(sessionId: string): Promise<MessageInfo[]> {
    const response = await fetch(`${this.baseURL}/api/sessions/${sessionId}/history`)

    if (!response.ok) {
      throw new Error(`Failed to get history: ${response.statusText}`)
    }

    const data: MessageHistoryResponse = await response.json()
    return data.messages
  }

  /**
   * Get pending approval requests
   */
  async getPendingApprovals(sessionId: string): Promise<ApprovalInfo[]> {
    const response = await fetch(
      `${this.baseURL}/api/sessions/${sessionId}/approvals/pending`
    )

    if (!response.ok) {
      throw new Error(`Failed to get pending approvals: ${response.statusText}`)
    }

    const data: PendingApprovalsResponse = await response.json()
    return data.approvals
  }

  /**
   * Approve a tool execution request
   */
  async approveRequest(
    sessionId: string,
    approvalId: string,
    reason?: string
  ): Promise<void> {
    const response = await fetch(
      `${this.baseURL}/api/sessions/${sessionId}/approvals/${approvalId}/approve`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reason: reason || null,
        }),
      }
    )

    if (!response.ok) {
      throw new Error(`Failed to approve request: ${response.statusText}`)
    }
  }

  /**
   * Deny a tool execution request
   */
  async denyRequest(
    sessionId: string,
    approvalId: string,
    reason: string
  ): Promise<void> {
    const response = await fetch(
      `${this.baseURL}/api/sessions/${sessionId}/approvals/${approvalId}/deny`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reason,
        }),
      }
    )

    if (!response.ok) {
      throw new Error(`Failed to deny request: ${response.statusText}`)
    }
  }

  // ========================================
  // Thread Management
  // ========================================

  /**
   * Create a new thread
   */
  async createThread(title?: string): Promise<string> {
    const response = await fetch(`${this.baseURL}/api/threads`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: title || null,
      }),
    })

    if (!response.ok) {
      throw new Error(`Failed to create thread: ${response.statusText}`)
    }

    const data: CreateSessionResponse = await response.json()
    return data.session_id
  }

  /**
   * List threads
   */
  async listThreads(limit: number = 50, offset: number = 0): Promise<ThreadInfo[]> {
    const response = await fetch(
      `${this.baseURL}/api/threads?limit=${limit}&offset=${offset}`
    )

    if (!response.ok) {
      throw new Error(`Failed to list threads: ${response.statusText}`)
    }

    const data: ListThreadsResponse = await response.json()
    return data.threads
  }

  /**
   * Get a single thread
   */
  async getThread(threadId: string): Promise<ThreadInfo> {
    const response = await fetch(`${this.baseURL}/api/threads/${threadId}`)

    if (!response.ok) {
      throw new Error(`Failed to get thread: ${response.statusText}`)
    }

    return await response.json()
  }

  /**
   * Update thread title
   */
  async updateThread(threadId: string, title: string): Promise<ThreadInfo> {
    const response = await fetch(`${this.baseURL}/api/threads/${threadId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title,
      }),
    })

    if (!response.ok) {
      throw new Error(`Failed to update thread: ${response.statusText}`)
    }

    return await response.json()
  }

  /**
   * Delete a thread
   */
  async deleteThread(threadId: string): Promise<void> {
    const response = await fetch(`${this.baseURL}/api/threads/${threadId}`, {
      method: 'DELETE',
    })

    if (!response.ok) {
      throw new Error(`Failed to delete thread: ${response.statusText}`)
    }
  }

  // ========================================
  // Identity Management
  // ========================================

  /**
   * List all identities
   */
  async listIdentities(): Promise<ListIdentitiesResponse> {
    const response = await fetch(`${this.baseURL}/api/identities`)

    if (!response.ok) {
      throw new Error(`Failed to list identities: ${response.statusText}`)
    }

    return await response.json()
  }

  /**
   * Get identity content
   */
  async getIdentity(name: string): Promise<IdentityContentResponse> {
    const response = await fetch(`${this.baseURL}/api/identities/${name}`)

    if (!response.ok) {
      throw new Error(`Failed to get identity: ${response.statusText}`)
    }

    return await response.json()
  }

  /**
   * Set active identity
   */
  async setActiveIdentity(name: string): Promise<void> {
    const response = await fetch(`${this.baseURL}/api/identities/active`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
      }),
    })

    if (!response.ok) {
      throw new Error(`Failed to set active identity: ${response.statusText}`)
    }
  }
}

// Export singleton instance
export const apiClient = new AgentAPIClient()

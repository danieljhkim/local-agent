/**
 * Main application component integrating chat UI and approval workflow
 */

import { AgentProvider, useAgentContext } from './context/AgentContext'
import { ChatHeader } from './components/ChatHeader'
import { MessageList } from './components/MessageList'
import { MessageInput } from './components/MessageInput'
import { ApprovalPanel } from './components/ApprovalPanel'

function ChatContainer() {
  const {
    sessionId,
    executionStatus,
    messages,
    pendingApprovals,
    sendMessage,
    approveRequest,
    denyRequest,
    error,
  } = useAgentContext()

  return (
    <div className="h-screen flex flex-col">
      <ChatHeader
        sessionId={sessionId}
        status={executionStatus}
        approvalCount={pendingApprovals.length}
      />

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 m-4 rounded">
          <strong>Error:</strong> {error}
        </div>
      )}

      <MessageList messages={messages} />

      <MessageInput onSend={sendMessage} disabled={executionStatus === 'executing'} />

      <ApprovalPanel
        approvals={pendingApprovals}
        onApprove={approveRequest}
        onDeny={denyRequest}
      />
    </div>
  )
}

export default function App() {
  return (
    <AgentProvider>
      <ChatContainer />
    </AgentProvider>
  )
}

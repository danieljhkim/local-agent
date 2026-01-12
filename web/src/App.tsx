/**
 * Main application component integrating chat UI and approval workflow
 */

import { AgentProvider, useAgentContext } from './context/AgentContext'
import { ChatHeader } from './components/ChatHeader'
import { MessageList } from './components/MessageList'
import { MessageInput } from './components/MessageInput'
import { ApprovalPanel } from './components/ApprovalPanel'
import { ThreadSidebar } from './components/ThreadSidebar'

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
    threads,
    currentThreadId,
    threadsLoading,
    createNewThread,
    selectThread,
    updateThreadTitle,
    deleteThread,
  } = useAgentContext()

  return (
    <div className="h-screen flex">
      {/* Thread Sidebar */}
      <ThreadSidebar
        threads={threads}
        currentThreadId={currentThreadId}
        onSelectThread={selectThread}
        onCreateThread={createNewThread}
        onUpdateThread={updateThreadTitle}
        onDeleteThread={deleteThread}
        loading={threadsLoading}
      />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        <ChatHeader
          sessionId={sessionId}
          status={executionStatus}
          approvalCount={pendingApprovals.length}
          onIdentityChange={(identity) => {
            console.log('Identity changed to:', identity)
            // Note: Identity change will take effect on next session creation
          }}
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

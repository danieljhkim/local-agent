/**
 * Sidebar showing conversation threads with create/edit/delete
 */

import { useState } from 'react'
import type { ThreadInfo } from '../lib/types'

interface ThreadSidebarProps {
  threads: ThreadInfo[]
  currentThreadId: string | null
  onSelectThread: (threadId: string) => void
  onCreateThread: () => void
  onUpdateThread: (threadId: string, title: string) => void
  onDeleteThread: (threadId: string) => void
  loading?: boolean
}

export function ThreadSidebar({
  threads,
  currentThreadId,
  onSelectThread,
  onCreateThread,
  onUpdateThread,
  onDeleteThread,
  loading,
}: ThreadSidebarProps) {
  const [editingThreadId, setEditingThreadId] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState('')

  const handleStartEdit = (thread: ThreadInfo) => {
    setEditingThreadId(thread.id)
    setEditTitle(thread.title)
  }

  const handleSaveEdit = (threadId: string) => {
    if (editTitle.trim()) {
      onUpdateThread(threadId, editTitle.trim())
    }
    setEditingThreadId(null)
    setEditTitle('')
  }

  const handleCancelEdit = () => {
    setEditingThreadId(null)
    setEditTitle('')
  }

  const handleDelete = (threadId: string) => {
    if (confirm('Delete this thread? This cannot be undone.')) {
      onDeleteThread(threadId)
    }
  }

  return (
    <div className="w-64 bg-gray-100 border-r border-gray-300 flex flex-col h-full">
      {/* Header with New Thread button */}
      <div className="p-4 border-b border-gray-300">
        <button
          onClick={onCreateThread}
          className="w-full bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 flex items-center justify-center gap-2"
        >
          <span className="text-xl font-bold">+</span>
          <span>New Thread</span>
        </button>
      </div>

      {/* Thread list */}
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="p-4 text-center text-gray-500">Loading threads...</div>
        )}

        {!loading && threads.length === 0 && (
          <div className="p-4 text-center text-gray-400">
            <p className="text-sm">No threads yet</p>
            <p className="text-xs mt-1">Create one to get started</p>
          </div>
        )}

        {!loading &&
          threads.map((thread) => (
            <div
              key={thread.id}
              className={`border-b border-gray-200 ${
                currentThreadId === thread.id ? 'bg-blue-50' : 'hover:bg-gray-200'
              }`}
            >
              {editingThreadId === thread.id ? (
                <div className="p-3">
                  <input
                    type="text"
                    value={editTitle}
                    onChange={(e) => setEditTitle(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') {
                        handleSaveEdit(thread.id)
                      } else if (e.key === 'Escape') {
                        handleCancelEdit()
                      }
                    }}
                    className="w-full border px-2 py-1 rounded text-sm"
                    autoFocus
                  />
                  <div className="flex gap-1 mt-2">
                    <button
                      onClick={() => handleSaveEdit(thread.id)}
                      className="flex-1 bg-green-600 text-white px-2 py-1 rounded text-xs hover:bg-green-700"
                    >
                      Save
                    </button>
                    <button
                      onClick={handleCancelEdit}
                      className="flex-1 bg-gray-300 px-2 py-1 rounded text-xs hover:bg-gray-400"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              ) : (
                <div
                  className="p-3 cursor-pointer group"
                  onClick={() => onSelectThread(thread.id)}
                >
                  <div className="flex justify-between items-start">
                    <div className="flex-1 min-w-0">
                      <h3 className="font-medium text-sm truncate">{thread.title}</h3>
                      <p className="text-xs text-gray-500 mt-1">
                        {thread.message_count || 0} messages
                      </p>
                    </div>
                    <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleStartEdit(thread)
                        }}
                        className="text-blue-600 hover:text-blue-800 px-1"
                        title="Rename"
                      >
                        ✎
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDelete(thread.id)
                        }}
                        className="text-red-600 hover:text-red-800 px-1"
                        title="Delete"
                      >
                        ×
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
      </div>
    </div>
  )
}

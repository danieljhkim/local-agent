/**
 * Scrollable message list container with auto-scroll
 */

import { useEffect, useRef } from 'react'
import { ChatMessage } from './ChatMessage'
import type { MessageInfo } from '../lib/types'

interface MessageListProps {
  messages: MessageInfo[]
  loading?: boolean
}

export function MessageList({ messages, loading }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-500">Loading messages...</div>
      </div>
    )
  }

  if (messages.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-gray-400 text-center">
          <p className="text-lg mb-2">No messages yet</p>
          <p className="text-sm">Start a conversation below</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-4">
      {messages.map((msg, index) => (
        <ChatMessage key={index} role={msg.role} content={msg.content} />
      ))}
      <div ref={messagesEndRef} />
    </div>
  )
}

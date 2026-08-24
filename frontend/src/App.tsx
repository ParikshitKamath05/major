import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, KeyboardEvent, MouseEvent } from 'react'
import axios from 'axios'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

type DocumentItem = {
  doc_id: string
  filename: string
  chunks: number
}

type SourceItem = {
  filename: string
  chunk_index: number
  text_preview: string
}

type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceItem[]
}

type UploadResponse = {
  doc_id: string
  filename: string
  chunks: number
}

type QueryResponse = {
  answer: string
  sources: SourceItem[]
  doc_id?: string | null
}

type DocumentsResponse = {
  documents: DocumentItem[]
}

function createMessageId() {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

/* ── ICONS ── */
const IconUpload = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" style={{ width: 14, height: 14 }}>
    <path d="M2.5 11.5v1.5a1 1 0 001 1h9a1 1 0 001-1v-1.5" />
    <path d="M8 10V2m0 0L5 5m3-3l3 3" />
  </svg>
)

const IconPDF = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ width: 13, height: 13 }}>
    <rect x="2" y="1" width="9" height="13" rx="1.5" />
    <path d="M11 1l3 3v10" />
    <path d="M5 6.5h4M5 9h3" />
  </svg>
)

const IconAll = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ width: 13, height: 13 }}>
    <rect x="1.5" y="1.5" width="5.5" height="5.5" rx="1" />
    <rect x="9" y="1.5" width="5.5" height="5.5" rx="1" />
    <rect x="1.5" y="9" width="5.5" height="5.5" rx="1" />
    <rect x="9" y="9" width="5.5" height="5.5" rx="1" />
  </svg>
)

const IconTrash = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2.5 4.5h11M6 4.5V3a1 1 0 011-1h2a1 1 0 011 1v1.5M12.5 4.5l-.7 9a1 1 0 01-1 .92H5.2a1 1 0 01-1-.92l-.7-9" />
  </svg>
)

const IconSources = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
    <path d="M2 4h12M2 8h9M2 12h6" />
  </svg>
)

const IconSend = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2L7 9" />
    <path d="M14 2L9.5 14 7 9 2 6.5 14 2z" />
  </svg>
)

const IconDoc = () => (
  <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" style={{ width: 24, height: 24, opacity: 0.35 }}>
    <rect x="2" y="1" width="12" height="14" rx="2" />
    <path d="M5 5.5h6M5 8.5h6M5 11.5h3" />
  </svg>
)

const IconBrain = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round">
    <path d="M9.5 2A2.5 2.5 0 017 4.5V5a3 3 0 00-3 3 3 3 0 001.5 2.6v.9a3.5 3.5 0 007 0V11a2.5 2.5 0 000-5V4.5A2.5 2.5 0 009.5 2z" />
    <path d="M14.5 2A2.5 2.5 0 0117 4.5V5a3 3 0 013 3 3 3 0 01-1.5 2.6v.9a3.5 3.5 0 01-7 0V11a2.5 2.5 0 000-5V4.5A2.5 2.5 0 0114.5 2z" />
    <path d="M12 11v8M9 16l3 3 3-3" />
  </svg>
)

function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [selectedDoc, setSelectedDoc] = useState('all')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [expandedSource, setExpandedSource] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const chatEndRef = useRef<HTMLDivElement | null>(null)

  const appendMessage = (message: Omit<Message, 'id'>) => {
    setMessages((prev) => [...prev, { ...message, id: createMessageId() }])
  }

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  useEffect(() => {
    void fetchDocuments()
  }, [])

  const fetchDocuments = async () => {
    try {
      const res = await axios.get<DocumentsResponse>(`${API_URL}/docs`)
      setDocuments(res.data.documents || [])
    } catch (err: unknown) {
      console.error('Failed to fetch documents:', err)
    }
  }

  const handleUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setUploading(true)
    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await axios.post<UploadResponse>(`${API_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })
      setDocuments((prev) => [
        ...prev,
        { doc_id: res.data.doc_id, filename: res.data.filename, chunks: res.data.chunks },
      ])
      setSelectedDoc(res.data.doc_id)
    } catch (err: unknown) {
      const message = axios.isAxiosError(err)
        ? err.response?.data?.detail || err.message
        : 'Unknown error'
      alert(`Upload failed: ${message}`)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleDelete = async (docId: string) => {
    try {
      await axios.delete(`${API_URL}/delete/${docId}`)
      setDocuments((prev) => prev.filter((doc) => doc.doc_id !== docId))
      if (selectedDoc === docId) setSelectedDoc('all')
    } catch (err: unknown) {
      const message = axios.isAxiosError(err)
        ? err.response?.data?.detail || err.message
        : 'Unknown error'
      alert(`Delete failed: ${message}`)
    }
  }

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const question = input.trim()
    setInput('')
    appendMessage({ role: 'user', content: question })
    setLoading(true)

    try {
      const res =
        selectedDoc === 'all'
          ? await axios.post<QueryResponse>(`${API_URL}/multi-query`, {
              question,
              doc_ids: [],
              top_k: 5,
            })
          : await axios.post<QueryResponse>(`${API_URL}/query`, {
              doc_id: selectedDoc,
              question,
              top_k: 5,
            })

      appendMessage({
        role: 'assistant',
        content: res.data.answer,
        sources: res.data.sources || [],
      })
    } catch (err: unknown) {
      const message = axios.isAxiosError(err)
        ? err.response?.data?.detail || err.message
        : 'Sorry, something went wrong. Please try again.'

      appendMessage({ role: 'assistant', content: message, sources: [] })
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void handleSend()
    }
  }

  const handleDeleteClick = (e: MouseEvent<HTMLButtonElement>, docId: string) => {
    e.stopPropagation()
    void handleDelete(docId)
  }

  const selectedDocument = documents.find((doc) => doc.doc_id === selectedDoc)

  return (
    <div className="app">
      {/* ── SIDEBAR ── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <h1>Docu<span>Mind</span></h1>
          <p>Ask your documents anything</p>
        </div>

        <div className="upload-section">
          <input
            type="file"
            accept=".pdf"
            onChange={handleUpload}
            ref={fileInputRef}
            style={{ display: 'none' }}
          />
          <button
            className="upload-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            type="button"
          >
            <IconUpload />
            {uploading ? 'Uploading…' : 'Upload PDF'}
          </button>
        </div>

        <div className="doc-list">
          <p className="doc-list-label">Library</p>

          <button
            className={`doc-item ${selectedDoc === 'all' ? 'active' : ''}`}
            onClick={() => setSelectedDoc('all')}
            type="button"
          >
            <span className="doc-icon-wrap">
              <IconAll />
            </span>
            <span className="doc-name">All Documents</span>
            <span className="doc-badge">{documents.length}</span>
          </button>

          {documents.map((doc) => (
            <div
              key={doc.doc_id}
              className={`doc-item ${selectedDoc === doc.doc_id ? 'active' : ''}`}
            >
              <button
                className="doc-select"
                onClick={() => setSelectedDoc(doc.doc_id)}
                type="button"
              >
                <span className="doc-icon-wrap">
                  <IconPDF />
                </span>
                <div className="doc-info">
                  <span className="doc-name">{doc.filename}</span>
                  <span className="doc-meta">{doc.chunks} chunks</span>
                </div>
              </button>
              <button
                className="delete-btn"
                onClick={(e) => handleDeleteClick(e, doc.doc_id)}
                title={`Delete ${doc.filename}`}
                type="button"
              >
                <IconTrash />
              </button>
            </div>
          ))}

          {documents.length === 0 && (
            <div className="empty-state">
              <IconDoc />
              <p>No documents yet.<br />Upload a PDF to get started.</p>
            </div>
          )}
        </div>
      </aside>

      {/* ── CHAT AREA ── */}
      <main className="chat-area">
        <div className="chat-header">
          <span className="chat-header-icon" />
          <span className="chat-header-text">
            {selectedDoc === 'all'
              ? 'Querying all documents'
              : selectedDocument?.filename || 'Select a document'}
          </span>
          {selectedDoc !== 'all' && selectedDocument && (
            <span className="chat-header-sub">{selectedDocument.chunks} chunks</span>
          )}
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome">
              <div className="welcome-icon">
                <IconBrain />
              </div>
              <h2>Welcome to DocuMind</h2>
              <p>Upload PDFs and ask questions about their contents.</p>
              <div className="suggestions">
                <p>Try asking</p>
                <button onClick={() => setInput('Summarize this document')} type="button">
                  Summarize this document
                </button>
                <button onClick={() => setInput('What topics are covered?')} type="button">
                  What topics are covered?
                </button>
                <button onClick={() => setInput('Which concepts are the hardest?')} type="button">
                  Which concepts are the hardest?
                </button>
                <button onClick={() => setInput('Compare the uploaded documents')} type="button">
                  Compare the uploaded documents
                </button>
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? 'YOU' : 'AI'}
              </div>
              <div className="message-content">
                <p>{msg.content}</p>

                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources">
                    <button
                      className="sources-toggle"
                      onClick={() => setExpandedSource(expandedSource === msg.id ? null : msg.id)}
                      type="button"
                    >
                      <IconSources />
                      {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''}
                    </button>
                    {expandedSource === msg.id && (
                      <div className="sources-list">
                        {msg.sources.map((source) => (
                          <div
                            key={`${msg.id}-${source.filename}-${source.chunk_index}`}
                            className="source-item"
                          >
                            <div className="source-header">
                              <span className="source-filename">{source.filename}</span>
                              <span className="source-chunk">chunk {source.chunk_index}</span>
                            </div>
                            <p className="source-text">{source.text_preview}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="message-avatar">AI</div>
              <div className="message-content">
                <div className="typing-indicator">
                  <span /><span /><span />
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        <div className="input-area">
          <div className="input-wrap">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                documents.length === 0
                  ? 'Upload a PDF to start asking questions…'
                  : 'Ask anything about your documents…'
              }
              rows={1}
              disabled={documents.length === 0}
            />
          </div>
          <button
            onClick={() => void handleSend()}
            disabled={!input.trim() || loading}
            className="send-btn"
            type="button"
            title="Send (Enter)"
          >
            {loading ? (
              <div className="send-btn-loading">
                <span /><span /><span />
              </div>
            ) : (
              <IconSend />
            )}
          </button>
        </div>
      </main>
    </div>
  )
}

export default App

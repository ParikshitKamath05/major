import { useEffect, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, KeyboardEvent, MouseEvent } from 'react'
import axios from 'axios'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

/* ── TYPES ── */
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

type ConfidenceSignals = {
  retrieval_similarity_mean: number
  retrieval_score: number
  top1_similarity: number
  top1_top2_margin: number
  verbalized_confidence: number
  self_consistency: number
  fused_score: number
  // Semantic Entropy
  semantic_entropy: number
  normalized_semantic_entropy: number
  epistemic_uncertainty: number
  aleatoric_uncertainty: number
  num_semantic_clusters: number
  semantic_confidence: number
  // EDL Fusion
  belief_correct: number
  belief_incorrect: number
  evidential_uncertainty: number
  predictive_confidence: number
  dirichlet_strength: number
  total_evidence: number
  // Conformal Prediction
  conformal_is_calibrated: boolean
  conformal_calibration_size: number
  conformal_abstention_threshold: number
  conformal_high_threshold: number
  conformal_coverage_guarantee: number
}

type CorrectiveMetadata = {
  expanded_queries: string[]
  total_candidates: number
  final_chunks: number
  rerank_scores: number[]
  method: string
}

type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: SourceItem[]
  confidence?: ConfidenceSignals
  escalated?: boolean
  abstained?: boolean
  correctiveMetadata?: CorrectiveMetadata
  warning?: string
  decision?: string
}

type HealthResponse = {
  status: string
  documents_indexed: number
  generation_model: string
  vision_model: string
  embedding_model: string
  conformal_calibrated?: boolean
  conformal_calibration_size?: number
}

type UploadResponse = {
  doc_id: string
  filename: string
  chunks: number
  message: string
}

type QueryResponse = {
  answer: string
  sources: SourceItem[]
  doc_id?: string | null
}

type UncertaintyQueryResponse = {
  request_id?: string
  answer: string
  sources: SourceItem[]
  doc_id: string
  confidence: ConfidenceSignals
  warning?: string | null
  escalated: boolean
  corrective_metadata?: CorrectiveMetadata | null
  abstained: boolean
  decision?: string
  latency_ms?: number
}

type DocumentsResponse = {
  documents: DocumentItem[]
  total?: number
}

function createMessageId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`
}

/* ── SVG ICONS ── */
const IconBrain = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2a5 5 0 0 1 5 5c0 1.5-.7 2.8-1.8 3.7.6 1.2.6 2.6 0 3.8 1.1.9 1.8 2.2 1.8 3.7a5 5 0 0 1-10 0c0-1.5.7-2.8 1.8-3.7-.6-1.2-.6-2.6 0-3.8C7.7 9.8 7 8.5 7 7a5 5 0 0 1 5-5z" />
    <path d="M12 7v10" />
  </svg>
)

const IconUpload = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
    <polyline points="17 8 12 3 7 8" />
    <line x1="12" y1="3" x2="12" y2="15" />
  </svg>
)

const IconFilePDF = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z" />
    <polyline points="14 2 14 8 20 8" />
    <line x1="9" y1="13" x2="15" y2="13" />
    <line x1="9" y1="17" x2="13" y2="17" />
  </svg>
)

const IconLayers = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="12 2 2 7 12 12 22 7 12 2" />
    <polyline points="2 17 12 22 22 17" />
    <polyline points="2 12 12 17 22 12" />
  </svg>
)

const IconTrash = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="3 6 5 6 21 6" />
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
  </svg>
)

const IconSend = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="22" y1="2" x2="11" y2="13" />
    <polygon points="22 2 15 22 11 13 2 9 22 2" />
  </svg>
)

const IconShieldCheck = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <polyline points="9 12 11 14 15 10" />
  </svg>
)

const IconShieldAlert = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
    <line x1="12" y1="8" x2="12" y2="12" />
    <line x1="12" y1="16" x2="12.01" y2="16" />
  </svg>
)

const IconZap = () => (
  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
  </svg>
)

const IconChevronDown = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 12 15 18 9" />
  </svg>
)

const IconSparkles = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 3c.132 0 .263 0 .393 0a7.5 7.5 0 0 0 7.92 7.92c0 .13 0 .26 0 .39a7.5 7.5 0 0 0-7.92 7.92c-.13 0-.26 0-.393 0a7.5 7.5 0 0 0-7.92-7.92c0-.13 0-.26 0-.39A7.5 7.5 0 0 0 12 3z" />
  </svg>
)

function App() {
  const [documents, setDocuments] = useState<DocumentItem[]>([])
  const [selectedDoc, setSelectedDoc] = useState<string>('all')
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState<string>('')
  const [loading, setLoading] = useState<boolean>(false)
  const [uploading, setUploading] = useState<boolean>(false)
  const [dragActive, setDragActive] = useState<boolean>(false)
  const [uncertaintyMode, setUncertaintyMode] = useState<boolean>(true)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  
  const [expandedConfidence, setExpandedConfidence] = useState<Record<string, boolean>>({})
  const [expandedSources, setExpandedSources] = useState<Record<string, boolean>>({})

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
    void fetchHealth()
  }, [])

  const fetchHealth = async () => {
    try {
      const res = await axios.get<HealthResponse>(`${API_URL}/health`)
      setHealth(res.data)
    } catch (err) {
      console.error('Failed to fetch health:', err)
    }
  }

  const fetchDocuments = async () => {
    try {
      const res = await axios.get<DocumentsResponse>(`${API_URL}/docs`)
      setDocuments(res.data.documents || [])
    } catch (err: unknown) {
      console.error('Failed to fetch documents:', err)
    }
  }

  const handleFileUpload = async (file: File) => {
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
      void fetchHealth()
    } catch (err: unknown) {
      const message = axios.isAxiosError(err)
        ? err.response?.data?.detail || err.message
        : 'Upload failed'
      alert(`Upload Error: ${message}`)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) void handleFileUpload(file)
  }

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      void handleFileUpload(e.dataTransfer.files[0])
    }
  }

  const handleDelete = async (docId: string) => {
    try {
      await axios.delete(`${API_URL}/delete/${docId}`)
      setDocuments((prev) => prev.filter((doc) => doc.doc_id !== docId))
      if (selectedDoc === docId) setSelectedDoc('all')
      void fetchHealth()
    } catch (err: unknown) {
      const message = axios.isAxiosError(err)
        ? err.response?.data?.detail || err.message
        : 'Delete failed'
      alert(`Error deleting document: ${message}`)
    }
  }

  const handleClearAll = async () => {
    if (!confirm('Are you sure you want to delete ALL indexed documents?')) return
    try {
      await axios.delete(`${API_URL}/clear-all`)
      setDocuments([])
      setSelectedDoc('all')
      void fetchHealth()
    } catch (err: unknown) {
      const message = axios.isAxiosError(err)
        ? err.response?.data?.detail || err.message
        : 'Clear all failed'
      alert(`Error clearing documents: ${message}`)
    }
  }

  const handleSend = async () => {
    if (!input.trim() || loading) return

    const question = input.trim()
    setInput('')
    appendMessage({ role: 'user', content: question })
    setLoading(true)

    try {
      if (selectedDoc !== 'all' && uncertaintyMode) {
        // Advanced Uncertainty-Triggered RAG with Corrective Escalation
        const res = await axios.post<UncertaintyQueryResponse>(`${API_URL}/query/uncertainty`, {
          doc_id: selectedDoc,
          question,
          top_k: 5,
        })

        appendMessage({
          role: 'assistant',
          content: res.data.answer,
          sources: res.data.sources || [],
          confidence: res.data.confidence,
          escalated: res.data.escalated,
          abstained: res.data.abstained,
          correctiveMetadata: res.data.corrective_metadata || undefined,
          warning: res.data.warning || undefined,
          decision: res.data.decision,
        })
      } else if (selectedDoc === 'all') {
        const res = await axios.post<QueryResponse>(`${API_URL}/multi-query`, {
          question,
          doc_ids: [],
          top_k: 5,
        })
        appendMessage({
          role: 'assistant',
          content: res.data.answer,
          sources: res.data.sources || [],
        })
      } else {
        const res = await axios.post<QueryResponse>(`${API_URL}/query`, {
          doc_id: selectedDoc,
          question,
          top_k: 5,
        })
        appendMessage({
          role: 'assistant',
          content: res.data.answer,
          sources: res.data.sources || [],
        })
      }
    } catch (err: unknown) {
      const message = axios.isAxiosError(err)
        ? err.response?.data?.detail || err.message
        : 'Something went wrong processing your request.'

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

  const toggleConfidence = (id: string) => {
    setExpandedConfidence((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  const toggleSources = (id: string) => {
    setExpandedSources((prev) => ({ ...prev, [id]: !prev[id] }))
  }

  const selectedDocument = documents.find((doc) => doc.doc_id === selectedDoc)

  return (
    <div className="app-container">
      {/* ── LEFT SIDEBAR ── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-title">
            <div className="brand-logo">
              <IconBrain />
            </div>
            <div className="brand-name">Docu<span>Mind</span></div>
          </div>
          <span className="brand-badge">RAG 2.0</span>
        </div>

        {/* Upload Zone */}
        <div className="upload-wrapper">
          <input
            type="file"
            accept=".pdf"
            onChange={handleInputChange}
            ref={fileInputRef}
            style={{ display: 'none' }}
          />
          <div
            className={`dropzone ${dragActive ? 'drag-active' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <div className="dropzone-icon">
              <IconUpload />
            </div>
            <div className="dropzone-title">
              {uploading ? 'Processing PDF & Vision...' : 'Upload PDF Document'}
            </div>
            <div className="dropzone-sub">
              Drag & drop or click to browse (Max 25MB)
            </div>
          </div>
        </div>

        {/* Document Library */}
        <div className="doc-library">
          <div className="library-header">
            <span>Document Library ({documents.length})</span>
            {documents.length > 0 && (
              <button
                className="clear-all-btn"
                onClick={handleClearAll}
                title="Clear all documents"
              >
                Clear All
              </button>
            )}
          </div>

          <button
            className={`doc-nav-item ${selectedDoc === 'all' ? 'active' : ''}`}
            onClick={() => setSelectedDoc('all')}
            type="button"
          >
            <div className="doc-nav-info">
              <div className="doc-nav-icon">
                <IconLayers />
              </div>
              <div className="doc-nav-text">
                <span className="doc-nav-name">All Documents</span>
                <span className="doc-nav-meta">Global Search Context</span>
              </div>
            </div>
          </button>

          {documents.map((doc) => (
            <div
              key={doc.doc_id}
              className={`doc-nav-item ${selectedDoc === doc.doc_id ? 'active' : ''}`}
              onClick={() => setSelectedDoc(doc.doc_id)}
            >
              <div className="doc-nav-info">
                <div className="doc-nav-icon">
                  <IconFilePDF />
                </div>
                <div className="doc-nav-text">
                  <span className="doc-nav-name" title={doc.filename}>{doc.filename}</span>
                  <span className="doc-nav-meta">{doc.chunks} vector chunks</span>
                </div>
              </div>
              <button
                className="doc-delete-btn"
                onClick={(e) => handleDeleteClick(e, doc.doc_id)}
                title={`Delete ${doc.filename}`}
                type="button"
              >
                <IconTrash />
              </button>
            </div>
          ))}
        </div>

        {/* Sidebar Footer Toggle */}
        <div className="sidebar-footer">
          <div
            className="mode-toggle"
            onClick={() => setUncertaintyMode(!uncertaintyMode)}
            title="Toggle Uncertainty Estimation & Corrective Escalation"
          >
            <div className={`toggle-switch ${uncertaintyMode ? 'active' : ''}`}>
              <div className="toggle-knob" />
            </div>
            <span>Corrective RAG</span>
          </div>
        </div>
      </aside>

      {/* ── MAIN WORKSPACE ── */}
      <main className="main-chat">
        {/* Navbar */}
        <header className="chat-navbar">
          <div className="active-context-pill">
            <div className="context-icon">
              {selectedDoc === 'all' ? <IconLayers /> : <IconFilePDF />}
            </div>
            <div className="context-details">
              <span className="context-title">
                {selectedDoc === 'all'
                  ? 'All Indexed Documents'
                  : selectedDocument?.filename || 'Select Document'}
              </span>
              <span className="context-sub">
                {selectedDoc === 'all'
                  ? `${health?.documents_indexed || 0} total chunks in ChromaDB`
                  : `${selectedDocument?.chunks || 0} chunks active`}
              </span>
            </div>
          </div>

          {health && (
            <div className="system-status">
              <div className="status-badge">
                <span className="status-dot" />
                <span>{health.generation_model.split('/')[1] || health.generation_model}</span>
              </div>
              {health.conformal_calibrated !== undefined && (
                <div className={`status-badge ${health.conformal_calibrated ? 'conformal-active' : 'conformal-bootstrap'}`}>
                  <span className={`status-dot ${health.conformal_calibrated ? '' : 'dot-warning'}`} />
                  <span>Conformal {health.conformal_calibrated ? `(${health.conformal_calibration_size})` : 'Bootstrap'}</span>
                </div>
              )}
            </div>
          )}
        </header>

        {/* Messages */}
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-hero">
              <div className="hero-icon-glow">
                <IconSparkles />
              </div>
              <h1 className="hero-title">What would you like to analyze?</h1>
              <p className="hero-sub">
                Upload PDFs to perform multimodal analysis, topic extraction, structural comparison, or question answering with fused confidence scoring.
              </p>

              <div className="preset-grid">
                <button
                  className="preset-card"
                  onClick={() => setInput('Summarize the main points and key takeaways')}
                  type="button"
                >
                  <span className="preset-card-title"><IconZap /> Executive Summary</span>
                  <span className="preset-card-desc">Synthesize key takeaways and core concepts</span>
                </button>

                <button
                  className="preset-card"
                  onClick={() => setInput('What are the main topics and concepts covered?')}
                  type="button"
                >
                  <span className="preset-card-title"><IconLayers /> Topic Extraction</span>
                  <span className="preset-card-desc">List all major topics, themes, and sub-topics</span>
                </button>

                <button
                  className="preset-card"
                  onClick={() => setInput('Which concepts or sections are the most complex?')}
                  type="button"
                >
                  <span className="preset-card-title"><IconShieldAlert /> Complexity Analysis</span>
                  <span className="preset-card-desc">Identify detailed or technical explanations</span>
                </button>

                <button
                  className="preset-card"
                  onClick={() => setInput('Compare and contrast the content across documents')}
                  type="button"
                >
                  <span className="preset-card-title"><IconSparkles /> Comparative Analysis</span>
                  <span className="preset-card-desc">Find overlaps, differences, and gaps</span>
                </button>
              </div>
            </div>
          )}

          {messages.map((msg) => (
            <div key={msg.id} className={`chat-row ${msg.role}`}>
              <div className={`avatar-bubble ${msg.role}`}>
                {msg.role === 'user' ? 'YOU' : 'AI'}
              </div>
              <div className="message-bubble">
                <div className="message-body">
                  {msg.content.split('\n').map((line, idx) => (
                    <p key={idx}>{line}</p>
                  ))}
                </div>

                {/* Abstention Warning */}
                {msg.abstained && (
                  <div className="abstention-banner">
                    <IconShieldAlert />
                    <div>
                      <strong>Groundedness Guardrail Triggered</strong>
                      <p style={{ marginTop: 2, fontSize: '0.74rem' }}>
                        The system abstained from answering to prevent hallucinations as candidate rerank scores fell below relevance threshold.
                      </p>
                    </div>
                  </div>
                )}

                {/* Corrective RAG Badge */}
                {msg.escalated && msg.correctiveMetadata && (
                  <div className="corrective-tag">
                    <div className="corrective-title">
                      <IconZap /> Corrective Escalation Active
                    </div>
                    <p>Triggered query expansion and Cohere neural reranking:</p>
                    <ul className="queries-list">
                      {msg.correctiveMetadata.expanded_queries.map((q, i) => (
                        <li key={i}>"{q}"</li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Confidence Score Component */}
                {msg.confidence && (
                  <div className="confidence-card">
                    <div className="confidence-header" onClick={() => toggleConfidence(msg.id)}>
                      <div className="confidence-left">
                        {(msg.confidence.predictive_confidence ?? msg.confidence.fused_score) >= 0.75 ? <IconShieldCheck /> : <IconShieldAlert />}
                        <span>Predictive Confidence: {((msg.confidence.predictive_confidence ?? msg.confidence.fused_score) * 100).toFixed(0)}%</span>
                      </div>
                      <span className={`score-badge ${(msg.confidence.predictive_confidence ?? msg.confidence.fused_score) >= 0.75 ? 'high' : (msg.confidence.predictive_confidence ?? msg.confidence.fused_score) >= 0.45 ? 'moderate' : 'low'}`}>
                        {(msg.confidence.predictive_confidence ?? msg.confidence.fused_score) >= 0.75 ? 'HIGH' : (msg.confidence.predictive_confidence ?? msg.confidence.fused_score) >= 0.45 ? 'MODERATE' : 'LOW ESCALATED'}
                      </span>
                    </div>

                    {expandedConfidence[msg.id] && (
                      <div className="confidence-body">
                        {/* ── Semantic Entropy Section ── */}
                        <div className="signal-section-header">Semantic Entropy</div>

                        <div className="signal-row">
                          <span className="signal-label">Semantic Confidence</span>
                          <div className="signal-bar-wrap">
                            <div className="signal-bar-fill bar-semantic" style={{ width: `${((msg.confidence.semantic_confidence ?? 0) * 100).toFixed(0)}%` }} />
                          </div>
                          <span>{((msg.confidence.semantic_confidence ?? 0) * 100).toFixed(0)}%</span>
                        </div>

                        <div className="signal-row">
                          <span className="signal-label">Epistemic Uncertainty</span>
                          <div className="signal-bar-wrap">
                            <div className="signal-bar-fill bar-danger" style={{ width: `${((msg.confidence.epistemic_uncertainty ?? 0) * 100).toFixed(0)}%` }} />
                          </div>
                          <span>{((msg.confidence.epistemic_uncertainty ?? 0) * 100).toFixed(0)}%</span>
                        </div>

                        <div className="signal-row">
                          <span className="signal-label">Aleatoric Uncertainty</span>
                          <div className="signal-bar-wrap">
                            <div className="signal-bar-fill bar-warning" style={{ width: `${((msg.confidence.aleatoric_uncertainty ?? 0) * 100).toFixed(0)}%` }} />
                          </div>
                          <span>{((msg.confidence.aleatoric_uncertainty ?? 0) * 100).toFixed(0)}%</span>
                        </div>

                        <div className="signal-row">
                          <span className="signal-label">Meaning Clusters</span>
                          <span className="signal-value-pill">{msg.confidence.num_semantic_clusters ?? 0}</span>
                        </div>

                        {/* ── Input Signals Section ── */}
                        <div className="signal-section-header">Input Signals</div>

                        <div className="signal-row">
                          <span className="signal-label">Self Consistency</span>
                          <div className="signal-bar-wrap">
                            <div className="signal-bar-fill" style={{ width: `${(msg.confidence.self_consistency * 100).toFixed(0)}%` }} />
                          </div>
                          <span>{(msg.confidence.self_consistency * 100).toFixed(0)}%</span>
                        </div>

                        <div className="signal-row">
                          <span className="signal-label">Retrieval Score</span>
                          <div className="signal-bar-wrap">
                            <div className="signal-bar-fill" style={{ width: `${((msg.confidence.retrieval_score ?? msg.confidence.retrieval_similarity_mean) * 100).toFixed(0)}%` }} />
                          </div>
                          <span>{((msg.confidence.retrieval_score ?? msg.confidence.retrieval_similarity_mean) * 100).toFixed(0)}%</span>
                        </div>

                        {/* ── EDL Fusion Section ── */}
                        <div className="signal-section-header">Evidential Deep Learning</div>

                        <div className="signal-row">
                          <span className="signal-label">Belief (Correct)</span>
                          <div className="signal-bar-wrap">
                            <div className="signal-bar-fill bar-success" style={{ width: `${((msg.confidence.belief_correct ?? 0) * 100).toFixed(0)}%` }} />
                          </div>
                          <span>{((msg.confidence.belief_correct ?? 0) * 100).toFixed(0)}%</span>
                        </div>

                        <div className="signal-row">
                          <span className="signal-label">Belief (Incorrect)</span>
                          <div className="signal-bar-wrap">
                            <div className="signal-bar-fill bar-danger" style={{ width: `${((msg.confidence.belief_incorrect ?? 0) * 100).toFixed(0)}%` }} />
                          </div>
                          <span>{((msg.confidence.belief_incorrect ?? 0) * 100).toFixed(0)}%</span>
                        </div>

                        <div className="signal-row">
                          <span className="signal-label">Vacuity (Epistemic)</span>
                          <div className="signal-bar-wrap">
                            <div className="signal-bar-fill bar-warning" style={{ width: `${((msg.confidence.evidential_uncertainty ?? 0) * 100).toFixed(0)}%` }} />
                          </div>
                          <span>{((msg.confidence.evidential_uncertainty ?? 0) * 100).toFixed(0)}%</span>
                        </div>

                        <div className="signal-row">
                          <span className="signal-label">Dirichlet Strength</span>
                          <span className="signal-value-pill">{(msg.confidence.dirichlet_strength ?? 0).toFixed(1)}</span>
                        </div>

                        <div className="signal-row">
                          <span className="signal-label">Total Evidence</span>
                          <span className="signal-value-pill">{(msg.confidence.total_evidence ?? 0).toFixed(1)}</span>
                        </div>

                        {/* ── Conformal Prediction Status ── */}
                        <div className="signal-section-header">Conformal Prediction</div>

                        <div className="signal-row">
                          <span className="signal-label">Calibrated</span>
                          <span className={`conformal-status-badge ${msg.confidence.conformal_is_calibrated ? 'calibrated' : 'bootstrapping'}`}>
                            {msg.confidence.conformal_is_calibrated ? 'Active' : 'Bootstrapping'}
                          </span>
                        </div>

                        <div className="signal-row">
                          <span className="signal-label">Abstention Threshold</span>
                          <span className="signal-value-pill">{((msg.confidence.conformal_abstention_threshold ?? 0) * 100).toFixed(0)}%</span>
                        </div>

                        <div className="signal-row">
                          <span className="signal-label">Coverage Guarantee</span>
                          <span className="signal-value-pill">{((msg.confidence.conformal_coverage_guarantee ?? 0) * 100).toFixed(0)}%</span>
                        </div>

                        {msg.confidence.conformal_calibration_size > 0 && (
                          <div className="signal-row">
                            <span className="signal-label">Calibration Points</span>
                            <span className="signal-value-pill">{msg.confidence.conformal_calibration_size}</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Sources Accordion */}
                {msg.sources && msg.sources.length > 0 && (
                  <div className="sources-card">
                    <button
                      className="sources-toggle-btn"
                      onClick={() => toggleSources(msg.id)}
                      type="button"
                    >
                      <IconChevronDown />
                      {msg.sources.length} Cited Source Chunk{msg.sources.length > 1 ? 's' : ''}
                    </button>

                    {expandedSources[msg.id] && (
                      <div className="sources-drawer">
                        {msg.sources.map((src, i) => (
                          <div key={i} className="source-item-card">
                            <div className="source-item-header">
                              <span>{src.filename}</span>
                              <span className="source-chunk-badge">Chunk #{src.chunk_index}</span>
                            </div>
                            <div className="source-preview-text">"{src.text_preview}"</div>
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
            <div className="chat-row assistant">
              <div className="avatar-bubble assistant">AI</div>
              <div className="message-bubble">
                <div className="typing-dots">
                  <span /><span /><span />
                </div>
              </div>
            </div>
          )}

          <div ref={chatEndRef} />
        </div>

        {/* Input Bar */}
        <div className="input-container">
          <div className="input-box">
            <textarea
              className="chat-textarea"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={
                documents.length === 0
                  ? 'Upload a PDF to start asking questions...'
                  : `Ask anything about ${selectedDoc === 'all' ? 'your document library' : selectedDocument?.filename}...`
              }
              rows={1}
            />
            <button
              className="send-btn-glow"
              onClick={() => void handleSend()}
              disabled={!input.trim() || loading}
              type="button"
              title="Send Prompt (Enter)"
            >
              <IconSend />
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}

export default App

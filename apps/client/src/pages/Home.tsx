import { useState } from 'react'
import { FiEye, FiEdit2, FiEdit3, FiPlus, FiChevronRight, FiChevronDown, FiLayers } from 'react-icons/fi'
import { Toaster, toast } from 'react-hot-toast'

type Process = {
  id: number
  name: string
  level: string
  description: string
}

type Capability = {
  id: number
  vertical: string
  name: string
  description: string
  processes: Process[]
}

export default function Home() {
  const verticalOptions = ['Vertical A', 'Vertical B', 'Vertical C']
  const [selectedVertical, setSelectedVertical] = useState('')
  const [capabilities, setCapabilities] = useState<Capability[]>([])

  const [isModalOpen, setIsModalOpen] = useState(false)
  const [modalMode, setModalMode] = useState<'add' | 'edit' | 'view'>('add')
  const [editingId, setEditingId] = useState<number | null>(null)

  const [formName, setFormName] = useState('')
  const [formDescription, setFormDescription] = useState('')

  function openAddModal() {
    setModalMode('add')
    setEditingId(null)
    setFormName('')
    setFormDescription('')
    setIsModalOpen(true)
  }

  function openEditModal(cap: Capability) {
    setModalMode('edit')
    setEditingId(cap.id)
    setFormName(cap.name)
    setFormDescription(cap.description)
    setIsModalOpen(true)
  }

  function openViewModal(cap: Capability) {
    setModalMode('view')
    setEditingId(cap.id)
    setFormName(cap.name)
    setFormDescription(cap.description)
    setIsModalOpen(true)
  }

  function saveCapability() {
    if (modalMode === 'add') {
      const newCap: Capability = {
        id: Date.now(),
        vertical: selectedVertical,
        name: formName,
        description: formDescription,
        processes: [],
      }
      setCapabilities((s) => [newCap, ...s])
      toast.success('Successfully added capability')
    } else if (modalMode === 'edit' && editingId != null) {
      setCapabilities((s) =>
        s.map((c) => (c.id === editingId ? { ...c, name: formName, description: formDescription } : c)),
      )
      toast.success('Successfully updated capability')
    }
    setIsModalOpen(false)
  }

 
  const [isProcessModalOpen, setIsProcessModalOpen] = useState(false)
  const [processCapId, setProcessCapId] = useState<number | null>(null)
  const [processName, setProcessName] = useState('')
  const [processLevel, setProcessLevel] = useState('Level 1 - Enterprise process')
  const [processDescription, setProcessDescription] = useState('')

  const processLevelOptions = [
    'Level 1 - Enterprise process',
    'Level 2 - Core process',
    'Level 3 - Process',
    'Level 4 - Subprocess',
  ]

  function openProcessModal(capId: number) {
    setProcessCapId(capId)
    setProcessName('')
    setProcessLevel(processLevelOptions[0])
    setProcessDescription('')
    setIsProcessModalOpen(true)
  }

  function saveProcess() {
    if (!processName.trim() || processCapId == null) return
    const newProcess: Process = {
      id: Date.now(),
      name: processName.trim(),
      level: processLevel,
      description: processDescription.trim(),
    }
    setCapabilities((s) => s.map((c) => (c.id === processCapId ? { ...c, processes: [...c.processes, newProcess] } : c)))
    setIsProcessModalOpen(false)
    toast.success('Successfully added Process')
  }

  const [expandedIds, setExpandedIds] = useState<number[]>([])

  function toggleExpand(capId: number) {
    setExpandedIds((prev) => (prev.includes(capId) ? prev.filter((id) => id !== capId) : [capId, ...prev]))
  }

  const parentCap = capabilities.find((c) => c.id === processCapId)

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 3000,
          style: {
            background: '#363636',
            color: '#fff',
          },
        }}
      />
      <div className="max-w-4xl mx-auto bg-white p-6 rounded-2xl shadow-sm">
        <div className="flex items-center gap-3 mb-6">
          <FiLayers className="w-8 h-8" />
          <h1 className="text-2xl font-semibold">Capabilities</h1>
        </div>

        <div className="flex items-center gap-4 mb-6">
          <select
            className="border rounded px-3 py-2"
            value={selectedVertical}
            onChange={(e) => setSelectedVertical(e.target.value)}
          >
            <option value="">Select vertical</option>
            {verticalOptions.map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>

          <button
            className={`px-4 py-2 rounded-md text-white ${selectedVertical ? 'bg-white text-primary border border-primary rounded font-normal hover:bg-primary hover:text-white transition-colors' : 'bg-gray-400 cursor-not-allowed'}`}
            disabled={!selectedVertical}
            onClick={openAddModal}
          >
            Add capability
          </button>
        </div>

        <div>
          {capabilities.length === 0 ? (
            <p className="text-gray-500">No capabilities yet.</p>
          ) : (
            <ul className="space-y-4">
              {capabilities.map((c) => {
                const isExpanded = expandedIds.includes(c.id)
                

                return (
                  <li key={c.id} className="bg-gray-50 border border-gray-200 rounded-2xl">
                    <div className="p-4 flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3">
                          <button
                            className="text-gray-500 p-2 rounded hover:bg-gray-100 flex items-center justify-center"
                            onClick={() => toggleExpand(c.id)}
                            aria-expanded={isExpanded}
                            title={isExpanded ? 'Collapse' : 'Expand'}
                          >
                            {isExpanded ? <FiChevronDown size={18} /> : <FiChevronRight size={18} />}
                          </button>

                          <div>
                            <div className="text-lg font-semibold">{c.name}</div>
                            <div className="text-xs text-gray-500">Vertical: {c.vertical}</div>
                            <div className="mt-2 text-sm text-gray-600">{c.description}</div>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <button
                          className="w-9 h-9 flex items-center justify-center rounded-md text-gray-600 hover:bg-gray-100"
                          onClick={() => openViewModal(c)}
                          title="View"
                          aria-label="View capability"
                        >
                          <FiEye size={16} />
                        </button>

                        <button
                          className="w-9 h-9 flex items-center justify-center rounded-md text-gray-600 hover:bg-gray-100"
                          onClick={() => openEditModal(c)}
                          title="Edit"
                          aria-label="Edit capability"
                        >
                          <FiEdit2 size={16} />
                        </button>

                        <button
                          className="w-9 h-9 flex items-center justify-center rounded-md bg-blue-600 text-white hover:bg-blue-700"
                          onClick={() => openProcessModal(c.id)}
                          title="Add process"
                          aria-label="Add process"
                        >
                          <FiPlus size={16} />
                        </button>
                      </div>
                    </div>

                    {isExpanded && (
                      <div className="p-4 border-t bg-transparent">
                        <div className="space-y-3">
                            <h1 className="text-xl font-semibold">Processes</h1>
                            {c.processes.length === 0 ? (
                            <div className="text-gray-500">No processes yet.</div>
                          ) : (
                            c.processes.map((p) => (
                              <div key={p.id} className="ml-8 bg-white border border-gray-200 rounded-md p-4 shadow-sm">
                                  <div className="flex items-center justify-between">
                                    <div>
                                      <div className="font-medium text-gray-800">{p.name}</div>
                                      <div className="text-xs text-gray-500 mt-1">{p.level}</div>
                                    </div>
                                    <div className="flex items-center gap-2">
                                    <button
                                      className="w-8 h-8 flex items-center justify-center rounded-md text-gray-600 hover:bg-gray-100"
                                      title="View process"
                                      aria-label="View process"
                                    >
                                      <FiEye size={14} />
                                    </button>

                                    <button
                                      className="w-8 h-8 flex items-center justify-center rounded-md text-gray-600 hover:bg-gray-100"
                                      title="Edit process"
                                      aria-label="Edit process"
                                    >
                                      <FiEdit2 size={14} />
                                    </button>

                                    <button
                                      className="w-8 h-8 flex items-center justify-center rounded-md bg-blue-600 text-white hover:bg-blue-700"
                                      onClick={() => openProcessModal(c.id)}
                                      title="Add subprocess"
                                      aria-label="Add subprocess"
                                    >
                                      <FiPlus size={14} />
                                    </button>
                                  </div>
                                  </div>
                                  {p.description && <div className="mt-3 text-sm text-gray-600">{p.description}</div>}
                                </div>
                              ))
                          )}
                        </div>
                      </div>
                    )}
                  </li>
                )
              })}
            </ul>
          )}
        </div>
      </div>

      {isModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center z-50">
          <div className="fixed inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setIsModalOpen(false)} />
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-md z-50 overflow-hidden">
            <div className="border-b border-gray-100 px-6 py-3 bg-gray-50 flex items-center gap-3">
              <FiEdit3 className="w-5 h-5 text-blue-600" />
              <h2 className="text-lg font-bold text-gray-900">
                {modalMode === 'view' ? 'View capability' : modalMode === 'edit' ? 'Edit capability' : 'Add capability'}
              </h2>
              {selectedVertical && <span className="text-xs text-gray-400 ml-2">to {selectedVertical}</span>}
            </div>

            <div className="p-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">Name</label>
                <input
                  className="w-full bg-gray-50 border border-indigo-100 rounded-xl px-4 py-3 text-gray-800 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="Enter capability name..."
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  disabled={modalMode === 'view'}
                />
               <label className="block text-sm font-medium text-gray-700 mb-2 mt-4">Description</label>
                <textarea
                  className="w-full bg-gray-50 border border-indigo-100 rounded-xl px-4 py-3 text-gray-800 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 min-h-[100px] resize-y"
                placeholder="Enter capability description..."
                  rows={4}
                  value={formDescription}
                  onChange={(e) => setFormDescription(e.target.value)}
                  disabled={modalMode === 'view'}
                />
              <div className="flex justify-end gap-3 mt-6">
                <button
                  className="px-3 py-1.5 rounded-md text-gray-600 hover:bg-gray-100 font-medium"
                  onClick={() => {
                    setIsModalOpen(false)
                    setFormName('')
                    setFormDescription('')
                  }}
                >
                Cancel
              </button>
              {modalMode !== 'view' && (
                <button
                  className="px-4 py-1.5 bg-blue-600 text-white rounded-md font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  onClick={saveCapability}
                  disabled={!formName.trim()}
                >
                  <FiPlus className="w-4 h-4" />
                    {modalMode === 'edit' ? 'Save changes' : 'Add capability'}
                </button>
              )}
              </div>
            </div>
          </div>
        </div>
      )}

      {isProcessModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center z-50">
          <div className="fixed inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setIsProcessModalOpen(false)} />
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-md z-50 overflow-hidden">
            <div className="border-b border-gray-100 px-6 py-3 bg-gray-50 flex items-center gap-3">
              <FiEdit3 className="w-5 h-5 text-blue-600" />
              <h2 className="text-lg font-bold text-gray-900">Add process</h2>
              {parentCap && <span className="text-xs text-gray-400 ml-2">to {parentCap.name}</span>}
            </div>

            <div className="p-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">Process name</label>
              <input
                className="w-full bg-gray-50 border border-indigo-100 rounded-xl px-4 py-3 text-gray-800 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="Enter process name..."
                value={processName}
                onChange={(e) => setProcessName(e.target.value)}
              />

              <label className="block text-sm font-medium text-gray-700 mb-2 mt-4">Level</label>
              <select
                className="w-full bg-gray-50 border border-indigo-100 rounded-xl px-4 py-3 text-gray-800 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                value={processLevel}
                onChange={(e) => setProcessLevel(e.target.value)}
              >
                {processLevelOptions.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt}
                  </option>
                ))}
              </select>

              <label className="block text-sm font-medium text-gray-700 mb-2 mt-4">Description</label>
              <textarea
                className="w-full bg-gray-50 border border-indigo-100 rounded-xl px-4 py-3 text-gray-800 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 min-h-[100px] resize-y"
                placeholder="Enter process description..."
                rows={3}
                value={processDescription}
                onChange={(e) => setProcessDescription(e.target.value)}
              />

              <div className="flex justify-end gap-3 mt-6">
                <button
                  className="px-3 py-1.5 rounded-md text-gray-600 hover:bg-gray-100 font-medium"
                  onClick={() => {
                    setIsProcessModalOpen(false)
                    setProcessName('')
                    setProcessLevel(processLevelOptions[0])
                    setProcessDescription('')
                  }}
                >
                  Cancel
                </button>
                <button
                  className="px-4 py-1.5 bg-blue-600 text-white rounded-md font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  onClick={saveProcess}
                  disabled={!processName.trim()}
                >
                  <FiPlus className="w-4 h-4" />
                  Add process
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

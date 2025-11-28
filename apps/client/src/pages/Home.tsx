import { useState, useEffect, useRef } from 'react'
import { FiEye, FiEdit2, FiEdit3, FiPlus, FiChevronRight, FiChevronDown, FiLayers, FiTrash2 } from 'react-icons/fi'
import { Toaster, toast } from 'react-hot-toast'


import { useCapabilityApi } from '../hooks/useCapability';
import type { Capability, Process, Domain } from '../hooks/useCapability';
import favicon from '../assets/favicon.png';


export default function Home() {
  const [domains, setDomains] = useState<Domain[]>([]);
  const [selectedDomain, setSelectedDomain] = useState('');
  const [capabilities, setCapabilities] = useState<Capability[]>([]);

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [modalMode, setModalMode] = useState<'add' | 'edit' | 'view'>('add');
  const [editingId, setEditingId] = useState<number | null>(null);

  const [formName, setFormName] = useState('');
  const [formDescription, setFormDescription] = useState('');

  const {
    listDomains,
    listCapabilities,
    createCapability,
    updateCapability,
    listProcesses,
    createProcess,
    deleteProcess,
    deleteCapability,
    generateProcesses,
  } = useCapabilityApi();

  const loadedRef = useRef(false);


  useEffect(() => {
    const prev = document.documentElement.style.overflowY;
    try {
      document.documentElement.style.overflowY = 'scroll';
    } catch (e) {

    }
    return () => {
      try {
        document.documentElement.style.overflowY = prev;
      } catch (e) {

      }
    };
  }, []);

  useEffect(() => {
    if (loadedRef.current) return;
    loadedRef.current = true;
    async function load() {
      try {
        const doms = await listDomains();
        setDomains(doms);
        
        const caps = await listCapabilities();

        const first = caps && caps.length > 0 ? (caps[0] as any) : null;
        if (first && Array.isArray((first as any).processes)) {

          const normalized = caps.map((c: any) => ({ ...c, processes: c.processes || [] }));
          setCapabilities(normalized as Capability[]);
        } else {

          const allProcesses = await listProcesses();

          const capMap: Record<number, Process[]> = {};
          allProcesses.forEach((p: any) => {
            const capId = p.capability_id || p.capability || p.capabilityId || p.capability?.id;
            if (capId) {
              if (!capMap[capId]) capMap[capId] = [];
              capMap[capId].push(p);
            }
          });
          setCapabilities(
            caps.map((c: any) => ({ ...c, processes: capMap[c.id] || [] }))
          );
        }
      } catch (e) {
        toast.error('Failed to load capabilities');
      }
    }
    load();
  }, [listCapabilities, listProcesses]);

  function openAddModal() {
    setModalMode('add');
    setEditingId(null);
    setFormName('');
    setFormDescription('');
    setIsModalOpen(true);
  }

  function openEditModal(cap: Capability) {
    setModalMode('edit');
    setEditingId(cap.id);
    setFormName(cap.name);
    setFormDescription(cap.description);
    setIsModalOpen(true);
  }

  function openViewModal(cap: Capability) {
    setModalMode('view');
    setEditingId(cap.id);
    setFormName(cap.name);
    setFormDescription(cap.description);
    setIsModalOpen(true);
  }

  async function saveCapability() {
    try {
      if (modalMode === 'add') {
        const newCap = await createCapability({
          domain: selectedDomain,
          name: formName,
          description: formDescription,
        });
        // Ensure the UI shows the domain name immediately without a full refresh.
        // The API may return the created capability with a domain_id or without a resolved domain name,
        // so prefer any domain value from the response, otherwise lookup from `domains` state.
        const domainName =
          (newCap as any).domain || domains.find((d) => String(d.id) === String(selectedDomain))?.name || selectedDomain;
        setCapabilities((s) => [
          { ...newCap, domain: domainName, processes: [] },
          ...s,
        ]);
        toast.success('Successfully added capability');
      } else if (modalMode === 'edit' && editingId != null) {
        const updated = await updateCapability(editingId, {
          name: formName,
          description: formDescription,
        });
        setCapabilities((s) =>
          s.map((c) => (c.id === editingId ? { ...c, ...updated } : c))
        );
        toast.success('Successfully updated capability');
      }
    } catch (e) {
      toast.error('Failed to save capability');
    }
    setIsModalOpen(false);
  }


  const [isProcessModalOpen, setIsProcessModalOpen] = useState(false);
  const [processCapId, setProcessCapId] = useState<number | null>(null);
  const [processLevel, setProcessLevel] = useState('core');
  const [isGenerating, setIsGenerating] = useState(false);
  const [processMode, setProcessMode] = useState<'manual' | 'ai'>('ai'); // Toggle between manual and AI
  const [manualProcessName, setManualProcessName] = useState('');
  const [manualProcessDescription, setManualProcessDescription] = useState('');
  const [isSavingManual, setIsSavingManual] = useState(false);

  const processLevelOptions = [
    'enterprise',
    'core',
    'process',
    'subprocess',
  ];

  function openProcessModal(capId: number) {
    setProcessCapId(capId);
    setProcessLevel('core');
    setProcessMode('ai');
    setManualProcessName('');
    setManualProcessDescription('');
    setIsProcessModalOpen(true);
  }

  async function handleSaveManualProcess() {
    if (!manualProcessName.trim() || processCapId == null) return;
    try {
      setIsSavingManual(true);
      const newProcess = await createProcess({
        name: manualProcessName.trim(),
        level: processLevel,
        description: manualProcessDescription.trim(),
        capability_id: processCapId,
      });
      setCapabilities((s) =>
        s.map((c) =>
          c.id === processCapId ? { ...c, processes: [...(c.processes || []), newProcess] } : c
        )
      );
      setIsProcessModalOpen(false);
      setManualProcessName('');
      setManualProcessDescription('');
      toast.success('Process created successfully');
    } catch (e) {
      toast.error('Failed to create process');
      console.error(e);
    } finally {
      setIsSavingManual(false);
    }
  }

  async function handleGenerateProcess() {
    if (processCapId == null) return;
    try {
      setIsGenerating(true);
      // Get the capability name and domain to pass to the LLM
      const parentCapability = capabilities.find((c) => c.id === processCapId);
      if (!parentCapability) {
        toast.error('Capability not found');
        return;
      }

      const result = await generateProcesses(parentCapability.name, processCapId, parentCapability.domain || '', processLevel);

      if (result.status === 'success') {
        // If backend returned created DB processes, try to refresh from server to show persisted entries.
        // There may be a short eventual-consistency delay on the server; retry a few times before falling back to the LLM preview.
        const created = result.processes || [];
        const coreFromLLM = result.data?.core_processes || result.data?.['Core Processes'] || [];

        async function fetchProcessesForCapWithRetry(capId: number, attempts = 5, delayMs = 300) {
          for (let i = 0; i < attempts; i++) {
            try {
              const procs = await listProcesses(capId);
              if (Array.isArray(procs) && procs.length > 0) return procs;
            } catch (err) {
              console.error('listProcesses failed during retry', err);
            }
            // wait before retrying
            // small sleep util
            await new Promise((res) => setTimeout(res, delayMs));
            delayMs *= 1.5;
          }
          return null;
        }

        if (Array.isArray(created) && created.length > 0) {
          // Try to fetch persisted processes for the capability with retries
          const fetched = await fetchProcessesForCapWithRetry(processCapId as number, 6, 300);
          if (fetched && fetched.length > 0) {
            // merge fetched processes into capabilities state
            setCapabilities((prev) => prev.map((c) => (c.id === processCapId ? { ...c, processes: fetched } : c)));
            if (processCapId != null) setExpandedIds((prev) => (prev.includes(processCapId) ? prev : [processCapId, ...prev]));
            toast.success(`Successfully generated and saved ${fetched.length} processes`);
          } else {
            // persisted data not available yet; use created payload if present, otherwise fall back to LLM preview
            // prefer created if it contains usable entries
            const prefer = created.length > 0 ? created : coreFromLLM;
            if (Array.isArray(prefer) && prefer.length > 0) {
              setCapabilities((prevCaps) =>
                prevCaps.map((c) =>
                  c.id === processCapId
                    ? {
                        ...c,
                        processes: prefer.map((proc: any, idx: number) => ({
                          id: proc.id ?? idx + 10000,
                          name: proc.name,
                          description: proc.description,
                          level: proc.level || 'core',
                          subprocesses: Array.isArray(proc.subprocesses)
                            ? proc.subprocesses.map((sub: any, subIdx: number) => ({
                                id: sub.id ?? `${idx + 10000}-${subIdx}`,
                                name: sub.name,
                                lifecycle_phase: sub.lifecycle_phase,
                              }))
                            : [],
                        })),
                      }
                    : c
                )
              );
              if (processCapId != null) setExpandedIds((prev) => (prev.includes(processCapId) ? prev : [processCapId, ...prev]));
              toast.success(`Successfully generated ${prefer.length} processes (preview)`);
            } else {
              toast.success('Generation triggered but no processes returned yet; they should appear shortly');
            }
          }
        } else {
          // No created processes from backend; show LLM-parsed preview immediately and expand capability
          const coreProcesses = coreFromLLM;
          setCapabilities((prevCaps) =>
            prevCaps.map((c) =>
              c.id === processCapId
                ? {
                    ...c,
                    processes: coreProcesses.map((proc: any, idx: number) => ({
                      id: idx + 10000,
                      name: proc.name,
                      description: proc.description,
                      level: 'core',
                      subprocesses: Array.isArray(proc.subprocesses)
                        ? proc.subprocesses.map((sub: any, subIdx: number) => ({
                            id: `${idx + 10000}-${subIdx}`,
                            name: sub.name,
                            lifecycle_phase: sub.lifecycle_phase,
                          }))
                        : [],
                    })),
                  }
                : c
            )
          );
          if (processCapId != null) setExpandedIds((prev) => (prev.includes(processCapId) ? prev : [processCapId, ...prev]));
          toast.success(`Successfully generated ${coreProcesses.length} processes (preview)`);
        }
        setIsProcessModalOpen(false);
      } else {
        toast.error('Failed to generate processes');
      }
    } catch (e) {
      toast.error('Failed to generate processes');
      console.error(e);
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleDeleteProcess(processId: number | string, capId: number, parentProcessId?: number | string) {
    const ok = window.confirm('Are you sure you want to delete this process? This cannot be undone.');
    if (!ok) return;
    try {
      // If it's a persisted process (numeric id) call backend
      if (typeof processId === 'number') {
        await deleteProcess(processId);
      }

      // Remove from local state (works for both persisted and temp frontend-only entries)
      setCapabilities((prev) =>
        prev.map((c) => {
          if (c.id !== capId) return c;
          const processes = (c.processes || []).map((p: any) => ({ ...p }));
          if (parentProcessId == null) {
            // deleting a top-level process
            return { ...c, processes: processes.filter((p: any) => String(p.id) !== String(processId)) };
          }
          // deleting a subprocess
          return {
            ...c,
            processes: processes.map((p: any) => {
              if (String(p.id) !== String(parentProcessId)) return p;
              const subs = Array.isArray(p.subprocesses) ? p.subprocesses.filter((s: any) => String(s.id) !== String(processId)) : [];
              return { ...p, subprocesses: subs };
            }),
          };
        })
      );

      toast.success('Process deleted');
    } catch (e) {
      console.error(e);
      toast.error('Failed to delete process');
    }
  }

  async function handleDeleteCapability(capId: number) {
    const ok = window.confirm('Delete this capability and all its processes? This cannot be undone.');
    if (!ok) return;
    try {
      await deleteCapability(capId);
      setCapabilities((prev) => prev.filter((c) => c.id !== capId));
      toast.success('Capability deleted');
    } catch (e) {
      console.error(e);
      toast.error('Failed to delete capability');
    }
  }


  const [expandedIds, setExpandedIds] = useState<number[]>([]);

  function toggleExpand(capId: number) {
    setExpandedIds((prev) => (prev.includes(capId) ? prev.filter((id) => id !== capId) : [capId, ...prev]));
  }


  const parentCap = capabilities.find((c) => c.id === processCapId);
  const selectedDomainName = domains.find((d) => String(d.id) === selectedDomain)?.name;
  const currentCap = editingId != null ? capabilities.find((c) => c.id === editingId) : undefined;
  const capDomainName = currentCap?.domain ?? selectedDomainName;

  return (
    <div className="min-h-screen bg-gray-50">
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
      <header className="border-b sticky top-0 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 z-50">
          <div className="container px-6 py-4">
              <div className="flex items-center gap-3">
                <img src={favicon} width={40} height={40} alt="favicon" />
                <div>
                    <h1 className="text-xl font-semibold">Capability Master™</h1>
                    <p className="text-xs text-muted-foreground">
                      Manage your enterprise capabilities and their associated processes.
                    </p>
                </div>
              </div>
          </div>
      </header>

      <div className="mx-auto mt-6 bg-white p-6 px-8 rounded-lg shadow-sm max-w-6xl">
        <div className="flex items-center gap-3 mb-6">
          <FiLayers className="w-8 h-8" />
          <h1 className="text-2xl font-semibold text-gray-900">Capabilities</h1>
        </div>

        <div className="flex items-center gap-4 mb-6">
          <div className="flex items-center gap-3">
            <select
              className="border rounded-md px-3 py-2 bg-white text-sm text-gray-700 focus:ring-2 focus:ring-indigo-200"
              value={selectedDomain}
              onChange={(e) => setSelectedDomain(e.target.value)}
            >
              <option value="">Select Domain</option>
              {domains.map((domain) => (
                <option key={domain.id} value={String(domain.id)}>
                  {domain.name}
                </option>
              ))}
            </select>
          </div>

          <button
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${selectedDomain ? 'bg-gray-100 border border-primary text-indigo-600 hover:bg-indigo-700 hover:text-white' : 'bg-gray-300 text-gray-500 cursor-not-allowed'}`}
            disabled={!selectedDomain}
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
                  <li key={c.id} className="bg-white border border-gray-100 rounded-2xl shadow-sm hover:shadow-md transition-shadow">
                    <div className="p-4 flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <div className="flex items-center gap-3">
                          <button
                            className="text-gray-400 p-2 rounded-md hover:bg-gray-50 flex items-center justify-center"
                            onClick={() => toggleExpand(c.id)}
                            aria-expanded={isExpanded}
                            title={isExpanded ? 'Collapse' : 'Expand'}
                          >
                            {isExpanded ? <FiChevronDown size={18} /> : <FiChevronRight size={18} />}
                          </button>

                          <div>
                            <div className="text-lg font-semibold text-gray-900">{c.name}</div>
                            <div className="mt-1">
                              <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-indigo-50 text-indigo-700">{c.domain ?? 'Unassigned'}</span>
                            </div>
                            <div className="mt-3 text-sm text-gray-600">{c.description}</div>
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <button
                          className="w-9 h-9 flex items-center justify-center rounded-md text-gray-600 hover:bg-gray-50"
                          onClick={() => openViewModal(c)}
                          title="View"
                          aria-label="View capability"
                        >
                          <FiEye size={16} />
                        </button>

                        <button
                          className="w-9 h-9 flex items-center justify-center rounded-md text-gray-600 hover:bg-gray-50"
                          onClick={() => openEditModal(c)}
                          title="Edit"
                          aria-label="Edit capability"
                        >
                          <FiEdit2 size={16} />
                        </button>

                        <button
                          className="w-9 h-9 flex items-center justify-center rounded-md bg-indigo-600 text-white hover:bg-indigo-700"
                          onClick={() => openProcessModal(c.id)}
                          title="Add process"
                          aria-label="Add process"
                        >
                          <FiPlus size={16} />
                        </button>

                        <button
                          className="w-9 h-9 flex items-center justify-center rounded-md text-red-600 hover:bg-red-50"
                          onClick={() => handleDeleteCapability(c.id)}
                          title="Delete capability"
                          aria-label="Delete capability"
                        >
                          <FiTrash2 size={16} />
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
                                    <div className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-teal-50 text-teal-700">{p.level}</div>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <button className="w-8 h-8 flex items-center justify-center rounded-md text-gray-600 hover:bg-gray-100" title="View process" aria-label="View process">
                                      <FiEye size={14} />
                                    </button>
                                    <button className="w-8 h-8 flex items-center justify-center rounded-md text-gray-600 hover:bg-gray-100" title="Edit process" aria-label="Edit process">
                                      <FiEdit2 size={14} />
                                    </button>
                                    <button className="w-8 h-8 flex items-center justify-center rounded-md bg-blue-600 text-white hover:bg-blue-700" onClick={() => openProcessModal(c.id)} title="Add subprocess" aria-label="Add subprocess">
                                      <FiPlus size={14} />
                                    </button>
                                    <button className="w-8 h-8 flex items-center justify-center rounded-md text-red-600 hover:bg-red-50" title="Delete process" aria-label="Delete process" onClick={() => handleDeleteProcess(p.id, c.id)}>
                                      <FiTrash2 size={14} />
                                    </button>
                                  </div>
                                </div>
                                {p.description && <div className="mt-3 text-sm text-gray-600">{p.description}</div>}
                                {/* Render subprocesses nested below core process */}
                                {Array.isArray((p as any).subprocesses) && (p as any).subprocesses.length > 0 && (
                                  <div className="ml-6 mt-3">
                                    <div className="font-semibold text-gray-700 text-sm mb-1">Subprocesses:</div>
                                    <ul className="space-y-2">
                                      {(p as any).subprocesses.map((sub: any) => (
                                        <li key={sub.id} className="pl-3 border-l-2 border-blue-200">
                                              <div className="flex items-center justify-between">
                                                <div>
                                                  <div className="text-gray-800 font-medium">{sub.name}</div>
                                                  <div className="text-xs text-gray-500">Phase: {sub.lifecycle_phase}</div>
                                                </div>
                                                <div>
                                                  <button className="text-red-600 p-1 rounded hover:bg-red-50" title="Delete subprocess" onClick={() => handleDeleteProcess(sub.id, c.id, p.id)}>
                                                    <FiTrash2 size={14} />
                                                  </button>
                                                </div>
                                              </div>
                                        </li>
                                      ))}
                                    </ul>
                                  </div>
                                )}
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
              {capDomainName && <span className="text-xs text-gray-400 ml-2">to {capDomainName}</span>}
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
              {/* Subtle Mode Toggle - Tab Style */}
              <div className="inline-flex gap-1 mb-6 p-1 bg-gray-100 rounded-lg">
                <button
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                    processMode === 'manual'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                  onClick={() => setProcessMode('manual')}
                >
                  Add Manually
                </button>
                <button
                  className={`px-4 py-2 rounded-md text-sm font-medium transition-all duration-200 ${
                    processMode === 'ai'
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                  onClick={() => setProcessMode('ai')}
                >
                  Generate with AI
                </button>
              </div>

              {/* Manual Mode */}
              {processMode === 'manual' && (
                <>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Process Name</label>
                  <input
                    className="w-full bg-gray-50 border border-indigo-100 rounded-xl px-4 py-3 text-gray-800 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 mb-4"
                    placeholder="Enter process name..."
                    value={manualProcessName}
                    onChange={(e) => setManualProcessName(e.target.value)}
                  />

                  <label className="block text-sm font-medium text-gray-700 mb-2">Process Level</label>
                  <select
                    className="w-full bg-gray-50 border border-indigo-100 rounded-xl px-4 py-3 text-gray-800 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 mb-4"
                    value={processLevel}
                    onChange={(e) => setProcessLevel(e.target.value)}
                  >
                    {processLevelOptions.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt}
                      </option>
                    ))}
                  </select>

                  <label className="block text-sm font-medium text-gray-700 mb-2">Description</label>
                  <textarea
                    className="w-full bg-gray-50 border border-indigo-100 rounded-xl px-4 py-3 text-gray-800 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 min-h-[100px] resize-y"
                    placeholder="Enter process description..."
                    rows={3}
                    value={manualProcessDescription}
                    onChange={(e) => setManualProcessDescription(e.target.value)}
                  />

                  <div className="flex justify-end gap-3 mt-6">
                    <button
                      className="px-3 py-1.5 rounded-md text-gray-600 hover:bg-gray-100 font-medium"
                      onClick={() => setIsProcessModalOpen(false)}
                    >
                      Cancel
                    </button>
                    <button
                      className="px-4 py-1.5 bg-indigo-600 text-white rounded-md font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                      onClick={handleSaveManualProcess}
                      disabled={!manualProcessName.trim() || isSavingManual}
                    >
                      <FiPlus className="w-4 h-4" />
                      {isSavingManual ? 'Creating...' : 'Create Process'}
                    </button>
                  </div>
                </>
              )}

              {/* AI Generation Mode */}
              {processMode === 'ai' && (
                <>
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                    <p className="text-sm text-gray-700">
                      <strong>Auto-generate processes</strong> based on the capability name and domain using AI. The system will create a complete process hierarchy with subprocesses.
                    </p>
                  </div>

                  <label className="block text-sm font-medium text-gray-700 mb-2">Process Level</label>
                  <select
                    className="w-full bg-gray-50 border border-indigo-100 rounded-xl px-4 py-3 text-gray-800 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 mb-4"
                    value={processLevel}
                    onChange={(e) => setProcessLevel(e.target.value)}
                  >
                    {processLevelOptions.map((opt) => (
                      <option key={opt} value={opt}>
                        {opt}
                      </option>
                    ))}
                  </select>

                  <div className="flex justify-end gap-3 mt-6">
                    <button
                      className="px-3 py-1.5 rounded-md text-gray-600 hover:bg-gray-100 font-medium"
                      onClick={() => setIsProcessModalOpen(false)}
                    >
                      Cancel
                    </button>
                    <button
                      className="px-4 py-1.5 bg-blue-600 text-white rounded-md font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                      onClick={handleGenerateProcess}
                      disabled={isGenerating}
                      title="Generate processes using AI"
                    >
                      <FiPlus className="w-4 h-4" />
                      {isGenerating ? 'Generating...' : 'Generate with AI'}
                    </button>
                  </div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

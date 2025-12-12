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
    createDomain,
    createCapability,
    updateCapability,
    listProcesses,
    createProcess,
    deleteProcess,
    deleteCapability,
    generateProcesses,
  } = useCapabilityApi();

  const [newDomainName, setNewDomainName] = useState('');
  const [isSavingDomain, setIsSavingDomain] = useState(false);

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

  async function handleCreateDomain() {
    const name = newDomainName.trim();
    if (!name) {
      toast.error('Enter a domain name');
      return;
    }
    try {
      setIsSavingDomain(true);
      const created = await createDomain({ name });
      setDomains((s) => [...s, created]);
      setSelectedDomain(String((created as any).id));
      setNewDomainName('');
      toast.success('Domain created successfully');
    } catch (e) {
      console.error(e);
      toast.error('Failed to create domain');
    } finally {
      setIsSavingDomain(false);
    }
  }


  const [isProcessModalOpen, setIsProcessModalOpen] = useState(false);
  const [processCapId, setProcessCapId] = useState<number | null>(null);
  const [processLevel, setProcessLevel] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [processMode, setProcessMode] = useState<'manual' | 'ai'>('ai'); // Toggle between manual and AI
  const [manualProcessName, setManualProcessName] = useState('');
  const [manualProcessDescription, setManualProcessDescription] = useState('');
  const [isSavingManual, setIsSavingManual] = useState(false);
  // Generated processes preview modal (LLM responses) - user must approve which to save
  const [generatedPreview, setGeneratedPreview] = useState<any[]>([]);
  const [isGeneratedModalOpen, setIsGeneratedModalOpen] = useState(false);
  // Track selected processes by their index: { processIdx: set of selected subprocess indices, or null if no subprocesses selected }
  const [selectedGeneratedIdxs, setSelectedGeneratedIdxs] = useState<Set<string>>(new Set());
  const [isSavingGenerated, setIsSavingGenerated] = useState(false);

  const processLevelOptions = [
    'enterprise',
    'core',
    'process',
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
      
      const parentCapability = capabilities.find((c) => c.id === processCapId);
      if (!parentCapability) {
        toast.error('Capability not found');
        return;
      }

      const result = await generateProcesses(parentCapability.name, processCapId, parentCapability.domain || '', processLevel, parentCapability.description || '');

      if (result.status === 'success') {
        const created = result.processes || [];
        
        // Handle the new response structure: result.data is the LLM JSON with 'processes' array
        let processesFromLLM: any[] = [];
        
        console.log('[DEBUG] LLM Response:', result);
        console.log('[DEBUG] result.data:', result.data);
        
        if (result.data?.processes && Array.isArray(result.data.processes)) {
          processesFromLLM = result.data.processes;
          console.log('[DEBUG] Using result.data.processes');
        } else if (result.data?.core_processes && Array.isArray(result.data.core_processes)) {
          // Fallback for old response format
          processesFromLLM = result.data.core_processes;
          console.log('[DEBUG] Using result.data.core_processes');
        } else if (result.data?.['Core Processes'] && Array.isArray(result.data['Core Processes'])) {
          // Fallback for another old format
          processesFromLLM = result.data['Core Processes'];
          console.log('[DEBUG] Using result.data[Core Processes]');
        } else if (Array.isArray(result.data)) {
          // If data is directly an array
          processesFromLLM = result.data;
          console.log('[DEBUG] Using result.data directly as array');
        } else {
          console.log('[DEBUG] No processes found in result.data, checking result.data keys:', Object.keys(result.data || {}));
        }

        const preferSource = Array.isArray(created) && created.length > 0 ? created : processesFromLLM;
        console.log('[DEBUG] processesFromLLM:', processesFromLLM);
        console.log('[DEBUG] preferSource length:', preferSource.length);

        
        const normalized = (Array.isArray(preferSource) ? preferSource : []).map((proc: any, idx: number) => ({
          tempId: idx,
          name: proc.name,
          description: proc.description,
          category: proc.category || '',
          level: proc.level || processLevel || 'core',
          subprocesses: Array.isArray(proc.subprocesses)
            ? proc.subprocesses.map((sub: any, subIdx: number) => ({
                id: sub.id ?? `${idx + 10000}-${subIdx}`,
                name: sub.name,
                description: sub.description,
                category: sub.category || '',
                lifecycle_phase: sub.lifecycle_phase,
              }))
            : [],
        }));

        console.log('[DEBUG] normalized processes:', normalized);
        setGeneratedPreview(normalized);
        setSelectedGeneratedIdxs(new Set());
        setIsGeneratedModalOpen(true);
        toast.success(`AI generated ${normalized.length} processes (preview). Please select which to save.`);
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
      
      if (typeof processId === 'number') {
        await deleteProcess(processId);
      }

      
      setCapabilities((prev) =>
        prev.map((c) => {
          if (c.id !== capId) return c;
          const processes = (c.processes || []).map((p: any) => ({ ...p }));
          if (parentProcessId == null) {
            
            return { ...c, processes: processes.filter((p: any) => String(p.id) !== String(processId)) };
          }
          
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

   
  async function saveSelectedGeneratedProcesses() {
    if (processCapId == null) return;
    if (!Array.isArray(generatedPreview) || generatedPreview.length === 0) return;
    if (selectedGeneratedIdxs.size === 0) {
      toast('No processes selected to save');
      return;
    }

    try {
      setIsSavingGenerated(true);
      const createdProcs: any[] = [];
      
      
      for (let procIdx = 0; procIdx < generatedPreview.length; procIdx++) {
        const procKey = `proc-${procIdx}`;
        const isProcessSelected = selectedGeneratedIdxs.has(procKey);
        
        if (!isProcessSelected) continue; 
        
        const p = generatedPreview[procIdx];
        if (!p) continue;
        
        try {
          
          const selectedSubs: any[] = [];
          if (Array.isArray(p.subprocesses)) {
            for (let subIdx = 0; subIdx < p.subprocesses.length; subIdx++) {
              const subKey = `proc-${procIdx}-sub-${subIdx}`;
                if (selectedGeneratedIdxs.has(subKey)) {
                const sub = p.subprocesses[subIdx];
                selectedSubs.push({ name: sub.name, description: sub.description || '', category: sub.category || undefined });
              }
            }
          }

          
          const created = await createProcess({
            name: p.name,
            level: p.level,
            description: p.description || '',
            capability_id: processCapId,
            category: p.category || undefined,
            subprocesses: selectedSubs.length > 0 ? selectedSubs : undefined,
          });
          createdProcs.push(created);
        } catch (err) {
          console.error('createProcess failed for generated item', err);
        }
      }

      
      if (createdProcs.length > 0) {
        setCapabilities((prev) =>
          prev.map((c) => (c.id === processCapId ? { ...c, processes: [...(c.processes || []), ...createdProcs] } : c))
        );
        if (processCapId != null) setExpandedIds((prev) => (prev.includes(processCapId) ? prev : [processCapId, ...prev]));
        toast.success(`Saved ${createdProcs.length} processes`);
      } else {
        toast.error('No processes were saved');
      }
    } catch (e) {
      console.error(e);
      toast.error('Failed to save selected processes');
    } finally {
      setIsSavingGenerated(false);
      setIsGeneratedModalOpen(false);
      setGeneratedPreview([]);
      setSelectedGeneratedIdxs(new Set());
    }
  }


  const [expandedIds, setExpandedIds] = useState<number[]>([]);
  const [expandedProcessIds, setExpandedProcessIds] = useState<Set<number>>(new Set());

  function toggleExpand(capId: number) {
    setExpandedIds((prev) => (prev.includes(capId) ? prev.filter((id) => id !== capId) : [capId, ...prev]));
  }

  function toggleProcessExpand(processId: number) {
    setExpandedProcessIds((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(processId)) {
        newSet.delete(processId);
      } else {
        newSet.add(processId);
      }
      return newSet;
    });
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
          <h1 className="text-2xl font-semibold text-gray-900">Sub-Vertical</h1>
        </div>

        <div className="flex items-center gap-4 mb-6">
          <div className="flex items-center gap-3">
            {domains.length === 0 ? (
              <div className="flex items-center gap-2">
                <input
                  className="border rounded-md px-3 py-2 bg-white text-sm text-gray-700 focus:ring-2 focus:ring-indigo-200"
                  placeholder="Enter domain name"
                  value={newDomainName}
                  onChange={(e) => setNewDomainName(e.target.value)}
                />
                <button
                  className={`px-3 py-2 rounded-md text-sm font-medium ${newDomainName.trim() ? 'bg-indigo-600 text-white' : 'bg-gray-300 text-gray-500 cursor-not-allowed'}`}
                  disabled={!newDomainName.trim() || isSavingDomain}
                  onClick={handleCreateDomain}
                >
                  {isSavingDomain ? 'Saving...' : 'Add Domain'}
                </button>
              </div>
            ) : (
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
            )}
          </div>

          <button
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${selectedDomain ? 'bg-gray-100 border border-primary text-indigo-600 hover:bg-indigo-700 hover:text-white' : 'bg-gray-300 text-gray-500 cursor-not-allowed'}`}
            disabled={!selectedDomain}
            onClick={openAddModal}
          >
            Add sub-vertical
          </button>
        </div>

        <div>
          {capabilities.length === 0 ? (
            <p className="text-gray-500">No Sub-Verticals yet.</p>
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
                      <div className="p-6 border-t bg-gray-50">
                        <h3 className="text-lg font-semibold text-gray-900 mb-6 flex items-center gap-2">
                          <FiLayers className="w-5 h-5 text-indigo-600" />
                          Processes
                        </h3>
                        {c.processes.length === 0 ? (
                          <div className="text-gray-500 text-center py-8">No processes yet.</div>
                        ) : (
                          <div className="space-y-3">
                            {c.processes.map((p) => {
                              const subprocesses = Array.isArray(p.subprocesses) ? p.subprocesses : [];
                              const hasSubprocesses = subprocesses.length > 0;
                              const isSubprocessesExpanded = expandedProcessIds.has(p.id);

                              return (
                                <div key={p.id} className="bg-white border border-gray-200 rounded-lg overflow-hidden hover:shadow-md transition-shadow">
                                  {/* Core Process Header */}
                                  <div className="p-4">
                                    <div className="flex items-start justify-between gap-4">
                                      <div className="flex-1">
                                        <div className="flex items-center gap-3">
                                          <div className="w-2 h-2 rounded-full bg-indigo-600 flex-shrink-0 mt-1"></div>
                                          <div className="flex-1">
                                            <h4 className="font-semibold text-gray-900">{p.name}</h4>
                                            <div className="flex items-center gap-2 mt-2">
                                              <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-indigo-100 text-indigo-800">
                                                {p.level}
                                              </span>
                                              {p.category && (
                                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-purple-100 text-purple-800">
                                                  {p.category}
                                                </span>
                                              )}
                                              {hasSubprocesses && (
                                                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs bg-gray-200 text-gray-700 font-medium">
                                                  {subprocesses.length} subprocess{subprocesses.length !== 1 ? 'es' : ''}
                                                </span>
                                              )}
                                            </div>
                                          </div>
                                        </div>
                                        {p.description && <p className="mt-2 text-sm text-gray-600 ml-5">{p.description}</p>}
                                      </div>

                                      <div className="flex items-center gap-1 flex-shrink-0">
                                        <button
                                          className="w-8 h-8 flex items-center justify-center rounded-md text-red-600 hover:bg-red-50 transition-colors"
                                          title="Delete process"
                                          onClick={() => handleDeleteProcess(p.id, c.id)}
                                        >
                                          <FiTrash2 size={16} />
                                        </button>
                                      </div>
                                    </div>
                                  </div>

                                  {/* Subprocesses Dropdown */}
                                  {hasSubprocesses && (
                                    <>
                                      <button
                                        onClick={() => toggleProcessExpand(p.id)}
                                        className="w-full px-4 py-3 bg-indigo-50 border-t border-gray-200 hover:bg-indigo-100 transition-colors flex items-center justify-between gap-2"
                                      >
                                        <div className="flex items-center gap-2">
                                          {isSubprocessesExpanded ? <FiChevronDown size={16} className="text-indigo-600" /> : <FiChevronRight size={16} className="text-indigo-600" />}
                                          <span className="text-sm font-semibold text-indigo-900">Subprocesses</span>
                                        </div>
                                      </button>

                                      {isSubprocessesExpanded && (
                                        <ul className="divide-y divide-gray-200 bg-indigo-50">
                                          {subprocesses.map((sub: any) => (
                                            <li
                                              key={sub.id}
                                              className="px-4 py-3 hover:bg-indigo-100 transition-colors"
                                            >
                                              <div className="ml-6 flex items-start justify-between gap-4">
                                                <div className="flex items-start gap-3 flex-1">
                                                  <div className="w-1.5 h-1.5 rounded-full bg-teal-500 flex-shrink-0 mt-1.5"></div>

                                                  <div className="flex-1 min-w-0">
                                                    <h5 className="font-semibold text-gray-800">{sub.name}</h5>
                                                    <div className="flex items-center gap-2 mt-1">
                                                      <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-teal-100 text-teal-700">
                                                        subprocess
                                                      </span>
                                                      {sub.category && (
                                                        <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-100 text-emerald-700">
                                                          {sub.category}
                                                        </span>
                                                      )}
                                                    </div>
                                                    {sub.description && <p className="text-xs text-gray-600 mt-2">{sub.description}</p>}
                                                  </div>
                                                </div>

                                                <button
                                                  className="w-8 h-8 flex items-center justify-center rounded-md text-red-600 hover:bg-red-50 transition-colors flex-shrink-0"
                                                  title="Delete subprocess"
                                                  onClick={() => handleDeleteProcess(sub.id, c.id, p.id)}
                                                >
                                                  <FiTrash2 size={14} />
                                                </button>
                                              </div>
                                            </li>
                                          ))}
                                        </ul>
                                      )}
                                    </>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        )}
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
                {modalMode === 'view' ? 'View capability' : modalMode === 'edit' ? 'Edit capability' : 'Add sub-vertical'}
              </h2>
              {capDomainName && <span className="text-xs text-gray-400 ml-2">to {capDomainName}</span>}
            </div>

            <div className="p-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">Name</label>
                <input
                  className="w-full bg-gray-50 border border-indigo-100 rounded-xl px-4 py-3 text-gray-800 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                placeholder="Enter sub-vertical name..."
                  value={formName}
                  onChange={(e) => setFormName(e.target.value)}
                  disabled={modalMode === 'view'}
                />
               <label className="block text-sm font-medium text-gray-700 mb-2 mt-4">Description</label>
                <textarea
                  className="w-full bg-gray-50 border border-indigo-100 rounded-xl px-4 py-3 text-gray-800 placeholder:text-gray-400 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 min-h-[100px] resize-y"
                placeholder="Enter sub-vertical description..."
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
                    {modalMode === 'edit' ? 'Save changes' : 'Add sub-vertical'}
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

      {/* Generated LLM preview modal - show checkbox list, select all, and save selected */}
      {isGeneratedModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center z-50">
          <div className="fixed inset-0 bg-black/40 backdrop-blur-sm" onClick={() => setIsGeneratedModalOpen(false)} />
          <div className="bg-white rounded-2xl shadow-xl w-full max-w-2xl z-50 overflow-y-auto max-h-[80vh]">
            <div className="border-b border-gray-100 px-6 py-3 bg-gray-50 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <FiEdit3 className="w-5 h-5 text-blue-600" />
                <h2 className="text-lg font-bold text-gray-900">Review generated processes</h2>
                <span className="text-sm text-gray-500">Select which generated processes to save</span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  className="px-3 py-1.5 rounded-md text-sm bg-gray-100 hover:bg-gray-200"
                  onClick={() => {
                    // toggle select all (both top-level and all subprocesses)
                    if (!generatedPreview || generatedPreview.length === 0) return;
                    const allKeys = new Set<string>();
                    generatedPreview.forEach((proc, idx) => {
                      allKeys.add(`proc-${idx}`);
                      if (Array.isArray(proc.subprocesses)) {
                        proc.subprocesses.forEach((_sub: any, subIdx: number) => {
                          allKeys.add(`proc-${idx}-sub-${subIdx}`);
                        });
                      }
                    });
                    
                    if (selectedGeneratedIdxs.size === allKeys.size) {
                      setSelectedGeneratedIdxs(new Set());
                    } else {
                      setSelectedGeneratedIdxs(allKeys);
                    }
                  }}
                >
                  {selectedGeneratedIdxs.size === 0 ? 'Select all' : 'Unselect all'}
                </button>
              </div>
            </div>

            <div className="p-6 space-y-4">
              {(!generatedPreview || generatedPreview.length === 0) ? (
                <div className="text-gray-500">No generated processes to review.</div>
              ) : (
                <ul className="space-y-4">
                  {generatedPreview.map((proc: any, idx: number) => {
                    const procKey = `proc-${idx}`;
                    const procChecked = selectedGeneratedIdxs.has(procKey);
                    
                    return (
                      <li key={idx} className="border rounded-lg p-4 bg-gray-50">
                        <div className="flex items-start gap-3">
                          <input
                            type="checkbox"
                            checked={procChecked}
                            onChange={(e) => {
                              const newSet = new Set(selectedGeneratedIdxs);
                              if (e.target.checked) {
                                newSet.add(procKey);
                                // Auto-select all subprocesses when parent is checked
                                if (Array.isArray(proc.subprocesses)) {
                                  proc.subprocesses.forEach((_sub: any, subIdx: number) => {
                                    newSet.add(`proc-${idx}-sub-${subIdx}`);
                                  });
                                }
                              } else {
                                newSet.delete(procKey);
                                // Auto-uncheck all subprocesses when parent is unchecked
                                if (Array.isArray(proc.subprocesses)) {
                                  proc.subprocesses.forEach((_sub: any, subIdx: number) => {
                                    newSet.delete(`proc-${idx}-sub-${subIdx}`);
                                  });
                                }
                              }
                              setSelectedGeneratedIdxs(newSet);
                            }}
                            className="mt-1"
                          />
                          <div className="flex-1">
                            <div className="flex items-center justify-between">
                              <div className="font-medium text-gray-900">{proc.name}</div>
                              <div className="flex items-center gap-2">
                                <div className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-teal-50 text-teal-700">{proc.level}</div>
                                {proc.category && (
                                  <div className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium bg-purple-50 text-purple-700">{proc.category}</div>
                                )}
                              </div>
                            </div>
                            {proc.description && <div className="mt-1 text-sm text-gray-600">{proc.description}</div>}
                            
                            {/* Subprocesses with individual checkboxes */}
                            {Array.isArray(proc.subprocesses) && proc.subprocesses.length > 0 && (
                              <div className="mt-3 ml-6 space-y-2 border-l-2 border-gray-300 pl-4">
                                <div className="text-xs font-semibold text-gray-700">Subprocesses:</div>
                                {proc.subprocesses.map((sub: any, subIdx: number) => {
                                  const subKey = `proc-${idx}-sub-${subIdx}`;
                                  const subChecked = selectedGeneratedIdxs.has(subKey);
                                  return (
                                    <div key={subIdx} className="flex items-start gap-2">
                                      <input
                                        type="checkbox"
                                        checked={subChecked}
                                        onChange={(e) => {
                                          const newSet = new Set(selectedGeneratedIdxs);
                                          if (e.target.checked) {
                                            newSet.add(subKey);
                                            // Auto-check parent if any subprocess is checked
                                            newSet.add(procKey);
                                          } else {
                                            newSet.delete(subKey);
                                          }
                                          setSelectedGeneratedIdxs(newSet);
                                        }}
                                        className="mt-1"
                                      />
                                      <div className="text-sm text-gray-700 flex-1">
                                        <div className="flex items-center gap-2">
                                          <span className="font-medium">{sub.name}</span>
                                          {sub.category && (
                                            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-emerald-50 text-emerald-700">{sub.category}</span>
                                          )}
                                        </div>
                                        {sub.lifecycle_phase && <div className="text-xs text-gray-500 mt-0.5">Phase: {sub.lifecycle_phase}</div>}
                                        {sub.description && <div className="text-xs text-gray-600 mt-0.5">{sub.description}</div>}
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              )}

              <div className="flex justify-end gap-3 mt-6 pt-4 border-t">
                <button
                  className="px-3 py-1.5 rounded-md text-gray-600 hover:bg-gray-100"
                  onClick={() => {
                    setIsGeneratedModalOpen(false);
                    setGeneratedPreview([]);
                    setSelectedGeneratedIdxs(new Set());
                  }}
                >
                  Cancel
                </button>

                <button
                  className="px-4 py-1.5 bg-blue-600 text-white rounded-md font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  onClick={saveSelectedGeneratedProcesses}
                  disabled={isSavingGenerated || selectedGeneratedIdxs.size === 0}
                >
                  {isSavingGenerated ? 'Saving...' : `Save selected (${selectedGeneratedIdxs.size})`}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

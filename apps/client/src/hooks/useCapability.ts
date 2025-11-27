import { useCallback } from 'react';


export type Domain = {
	id: number;
	name: string;
	created_at: string;
	updated_at: string;
};

export type Process = {
	id: number | string;
	name: string;
	level: string;
	description: string;
	lifecycle_phase?: string;
	subprocesses?: Process[];
};

export type Capability = {
	id: number;
	domain: string;
	name: string;
	description: string;
	processes: Process[];
};


const BASE_URL = '/api';

async function fetcher<T>(url: string, options?: RequestInit): Promise<T> {
	const res = await fetch(url, options);
	if (!res.ok) {
		const error = await res.text();
		throw new Error(error || 'API error');
	}
	return res.json();
}

export function useCapabilityApi() {

	const listDomains = useCallback(async () => {
		return fetcher<Domain[]>(`${BASE_URL}/domains`);
	}, []);

	const createDomain = useCallback(async (data: Omit<Domain, 'id' | 'created_at' | 'updated_at'>) => {
		const res = await fetcher<Domain>(`${BASE_URL}/domains`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data),
		});
		return res;
	}, []);

	const updateDomain = useCallback(async (id: number, data: Partial<Domain>) => {
		const res = await fetcher<Domain>(`${BASE_URL}/domains/${id}`, {
			method: 'PUT',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data),
		});
		return res;
	}, []);

	const deleteDomain = useCallback(async (id: number) => {
		await fetcher(`${BASE_URL}/domains/${id}`, { method: 'DELETE' });
	}, []);

	const listCapabilities = useCallback(async () => {
		return fetcher<Capability[]>(`${BASE_URL}/capabilities`);
	}, []);

	const createCapability = useCallback(async (data: Omit<Capability, 'id' | 'processes'>) => {
		// Accept either `domain` or `domain_id` in the payload and normalize to `domain_id`
		const payload: any = { ...data } as any;
		if ((payload as any).domain !== undefined && (payload as any).domain !== null) {
			// selectedDomain in UI may be a string; ensure it's a number when possible
			const parsed = Number((payload as any).domain);
			payload.domain_id = Number.isNaN(parsed) ? (payload as any).domain : parsed;
			delete payload.domain;
		}
		const res = await fetcher<Capability>(`${BASE_URL}/capabilities`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(payload),
		});
		return res;
	}, []);

	const updateCapability = useCallback(async (id: number, data: Partial<Capability>) => {
		const res = await fetcher<Capability>(`${BASE_URL}/capabilities/${id}`, {
			method: 'PUT',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data),
		});
		return res;
	}, []);

	const deleteCapability = useCallback(async (id: number) => {
		await fetcher(`${BASE_URL}/capabilities/${id}`, { method: 'DELETE' });
	}, []);


	const listProcesses = useCallback(async (capabilityId?: number) => {
		if (capabilityId) {
			return fetcher<Process[]>(`${BASE_URL}/processes?capability_id=${capabilityId}`);
		}
		return fetcher<Process[]>(`${BASE_URL}/processes`);
	}, []);


	const createProcess = useCallback(async (data: any) => {
		const res = await fetcher<Process>(`${BASE_URL}/processes`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data),
		});
		return res;
	}, []);

	const updateProcess = useCallback(async (id: number, data: Partial<Process>) => {
		const res = await fetcher<Process>(`${BASE_URL}/processes/${id}`, {
			method: 'PUT',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify(data),
		});
		return res;
	}, []);

	const deleteProcess = useCallback(async (id: number) => {
		await fetcher(`${BASE_URL}/processes/${id}`, { method: 'DELETE' });
	}, []);

	const generateProcesses = useCallback(async (processName: string, capabilityId: number, domain: string, processType: string) => {
		const res = await fetcher<any>(`${BASE_URL}/processes/generate`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				process_name: processName,
				capability_id: capabilityId,
				domain: domain,
				process_type: processType,
			}),
		});
		return res;
	}, []);

	return {
		listDomains,
		createDomain,
		updateDomain,
		deleteDomain,
		listCapabilities,
		createCapability,
		updateCapability,
		deleteCapability,
		listProcesses,
		createProcess,
		updateProcess,
		deleteProcess,
		generateProcesses,
	};

}


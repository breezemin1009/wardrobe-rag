import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react';
import { api, type Health } from '@/services/rag';

type Connection = { health: Health | null; error: string; checking: boolean; refresh: () => void; revision: number };
const Context = createContext<Connection | null>(null);
export function RagProvider({ children }: { children: ReactNode }) {
  const [connection, setConnection] = useState({ health: null as Health | null, error: '', checking: true, revision: 0 });
  const refresh = useCallback(() => {
    setConnection(c => ({ ...c, checking: true }));
    api<Health>('/health').then(health => setConnection(c => ({ health, error: '', checking: false, revision: c.revision+1 })))
      .catch((e: Error) => setConnection(c => ({ health: null, error: e.message, checking: false, revision: c.revision+1 })));
  }, []);
  useEffect(() => {
    const controller = new AbortController();
    api<Health>('/health', undefined, controller.signal).then(health => setConnection({ health, error: '', checking: false, revision: 1 }))
      .catch((e: Error) => { if (!controller.signal.aborted) setConnection({ health: null, error: e.message, checking: false, revision: 1 }); });
    return () => controller.abort();
  }, []);
  return <Context.Provider value={{ ...connection, refresh }}>{children}</Context.Provider>;
}
export function useRag() { const context = useContext(Context); if (!context) throw new Error('RagProvider is missing'); return context; }

import { wardrobeStorage } from '@/services/wardrobe-storage';
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import type { AppState, Garment } from '@/domain/model';
import { initialState } from '@/domain/demo';
import { archiveGarment, parseState, restoreGarment, saveGarment } from '@/domain/rules';
import { fingerprintPhoto } from '@/services/photo-fingerprint';

const KEY = 'wardrobe-rag:frontend:v1';
type Store = AppState & {
  ready: boolean; error: string | null;
  save: (g: Garment) => Promise<void>;
  archive: (id: string) => Promise<void>;
  restore: (id: string) => Promise<void>;
  toggleSaved: (id: string) => Promise<void>;
};
const Context = createContext<Store | null>(null);
export function WardrobeProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState(initialState);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const current = useRef(state);
  const queue = useRef<Promise<unknown>>(Promise.resolve());
  const writable = useRef(false);
  useEffect(() => {
    let mounted = true;
    wardrobeStorage.getItem(KEY).then(async raw => {
      const restored = raw ? parseState(raw) : initialState();
      // Backfill legacy photos locally. Unreadable old files remain preserved.
      const hydrate = async (g: Garment): Promise<Garment> => !g.imageUri || g.imageHash ? g :
        { ...g, imageHash: await fingerprintPhoto(g.imageUri).catch(() => undefined) };
      restored.garments = await Promise.all(restored.garments.map(hydrate));
      restored.archived = await Promise.all(restored.archived.map(hydrate));
      if (!mounted) return;
      current.current = restored; setState(restored); writable.current = true;
    }).catch(() => {
      if (mounted) setError('저장된 옷장을 불러오지 못했어요. 기존 데이터를 보호하기 위해 저장을 중단했습니다.');
    }).finally(() => { if (mounted) setReady(true); });
    return () => { mounted = false; };
  }, []);
  function mutate(fn: (s: AppState) => AppState) {
    const operation = queue.current.then(async () => {
      if (!writable.current) throw new Error('옷장을 불러온 후 다시 시도해주세요.');
      const next = fn(current.current);
      try { await wardrobeStorage.setItem(KEY, JSON.stringify(next)); }
      catch { throw new Error('기기에 저장하지 못했어요. 저장 공간과 사진 용량을 확인해주세요.'); }
      current.current = next; setState(next);
    });
    queue.current = operation.catch(() => undefined);
    return operation;
  }
  return <Context.Provider value={{ ...state, ready, error,
    save: g => mutate(s => saveGarment(s, g)),
    archive: id => mutate(s => archiveGarment(s, id)),
    restore: id => mutate(s => restoreGarment(s, id)),
    toggleSaved: id => mutate(s => ({ ...s, savedIds: s.savedIds.includes(id) ? s.savedIds.filter(v => v !== id) : [...s.savedIds, id] })),
  }}>{children}</Context.Provider>;
}
export function useWardrobe() {
  const value = useContext(Context);
  if (!value) throw new Error('WardrobeProvider is missing');
  return value;
}

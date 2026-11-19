import { useEffect, useState } from 'react';
import { api, type Assessment } from './rag';
import { useRag } from '@/state/RagProvider';

export function useAssessment(input: string | null) {
  const connection = useRag();
  const [attempt, setAttempt] = useState(0);
  const key = `${connection.revision}:${attempt}:${input}`;
  const [state, setState] = useState<{ key: string; data: Assessment | null; error: string }>({ key: '', data: null, error: '' });
  useEffect(() => {
    if (!input || !connection.health) return;
    const controller = new AbortController();
    api<Assessment>('/assess', JSON.parse(input), controller.signal)
      .then(data => { if (!controller.signal.aborted) setState({ key, data, error: '' }); })
      .catch((e: Error) => { if (!controller.signal.aborted) setState({ key, data: null, error: e.message }); });
    return () => controller.abort();
  }, [input, key, connection.health]);
  const current = state.key === key;
  return { data: current ? state.data : null, error: current ? state.error : '',
    loading: Boolean(input && connection.health && !current), retry: () => setAttempt(a => a+1) };
}

export function useRankings(input: string | null) {
  const connection = useRag();
  const [attempt, setAttempt] = useState(0);
  const key = connection.revision + ':' + attempt + ':' + input;
  const [state, setState] = useState<{ key: string; data: { results: Assessment[]; comparable: boolean } | null; error: string }>({ key: '', data: null, error: '' });
  useEffect(() => {
    if (!input || !connection.health) return;
    const controller = new AbortController();
    api<{ results: Assessment[]; comparable: boolean }>('/rank', JSON.parse(input), controller.signal)
      .then(data => { if (!controller.signal.aborted) setState({ key, data, error: '' }); })
      .catch((e: Error) => { if (!controller.signal.aborted) setState({ key, data: null, error: e.message }); });
    return () => controller.abort();
  }, [input, key, connection.health]);
  const current = state.key === key;
  return { data: current ? state.data : null, error: current ? state.error : '',
    loading: Boolean(input && connection.health && !current), retry: () => setAttempt(a => a+1) };
}

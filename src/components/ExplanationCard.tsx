import { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { api, type Assessment, type GeneratedAnswer } from '@/services/rag';
import { Button, Notice, Tag } from './ui';
import { C, font } from './theme';

export function ExplanationCard({ result, onEvidence }: { result: Assessment; onEvidence: () => void }) {
  const [answer, setAnswer] = useState<GeneratedAnswer | null>(null), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const active = useRef<AbortController | null>(null);
  useEffect(() => () => active.current?.abort(), []);
  async function generate() {
    if (active.current) return;
    const controller = new AbortController(); active.current = controller; setBusy(true); setError('');
    try {
      const data = await api<{ answer: GeneratedAnswer }>('/explain', { runId: result.runId }, controller.signal);
      if (!controller.signal.aborted) setAnswer(data.answer);
    } catch (e) { if (!controller.signal.aborted) setError((e as Error).message); }
    finally { if (!controller.signal.aborted) { setBusy(false); active.current = null; } }
  }
  return <View style={st.card}><View style={st.heading}><Text style={st.title}>근거를 읽어주는 AI</Text><Tag blue>{answer?.mode === 'nvidia' ? 'NVIDIA' : answer ? '원문 근거' : '설명 도우미'}</Tag></View>
    <Text style={st.copy}>보유 옷과의 조합과 기존 옷 대비 추가 이점을 근거와 함께 설명해요.</Text>
    {answer && <><Text style={st.answer}>{answer.text}</Text><Text style={st.small}>{answer.mode === 'nvidia' ? `${answer.cached ? '같은 평가의 설명 재사용 · ' : ''}${answer.model}` : answer.warning || '설명 대신 원문 근거를 확인해주세요.'}</Text><Text style={st.small}>인용 근거 {answer.citations.length}개 · 설명은 원문과 함께 확인해주세요.</Text></>}
    {Boolean(error) && <Text accessibilityRole="alert" style={st.error}>{error}</Text>}
    <View style={st.actions}><Button secondary icon="sparkles-outline" loading={busy} onPress={generate} disabled={result.evidence.length === 0}>{answer ? '설명 다시 확인' : 'AI 근거 설명 보기'}</Button><Button secondary icon="document-text-outline" onPress={onEvidence}>원문 근거 확인</Button></View>
    <View style={{ marginTop: 16 }}><Notice>설명 요청 시 이름·사진 없이 속성 그룹, 계산 결과와 공개 근거가 NVIDIA API에 전달됩니다.</Notice></View>
  </View>;
}
const st = StyleSheet.create({ card: { padding: 24, marginTop: 24, borderRadius: 18, backgroundColor: C.white, borderWidth: 1, borderColor: C.line }, heading: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12 }, title: { fontFamily: font, fontSize: 16, fontWeight: '700', color: C.ink }, copy: { fontFamily: font, fontSize: 12, color: C.muted, lineHeight: 22, marginTop: 12 }, answer: { fontFamily: font, fontSize: 13, color: C.ink, lineHeight: 24, marginTop: 18 }, small: { fontFamily: font, fontSize: 10, color: C.muted, lineHeight: 19, marginTop: 7 }, actions: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, marginTop: 18 }, error: { fontFamily: font, color: '#B54848', fontSize: 12, marginTop: 14 } });

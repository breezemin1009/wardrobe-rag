import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { api } from '@/services/rag';
import { Button, Tag } from '@/components/ui';
import { C, font } from '@/components/theme';

type Garment = { id: string; label: string; originalTitle: string };
type Example = { id: string; title: string; basis: string; request: { candidates: string[]; contexts: string[]; owned: string[] }; garments: Record<'candidates' | 'contexts' | 'owned', Garment[]> };
type Summary = { corpus: { verifiedMenswearItems: number; referencePairs: number }; validation: { cases: number; selectedRate: number; strongestComparisonRate: number; difference: { ci95: number[] }; metric: string; scope: string } };
type Result = { evaluationFingerprint: string; results: { id: string; score: number; combinationSupport: number; additionalContribution: number; rank: number | null; alreadyOwned: boolean; evidence: { top: { title: string }; bottom: { title: string }; sourceOutfits: string[] }[] }[]; method: { formula: string } };

const garmentLabel = (label: string) => label.replaceAll('팬츠', '바지');

export function MenswearCaseStudy() {
  const [summary, setSummary] = useState<Summary | null>(null), [examples, setExamples] = useState<Example[]>([]);
  const [index, setIndex] = useState(0), [result, setResult] = useState<Result | null>(null), [error, setError] = useState('');
  const explanationRequest = useRef<AbortController | null>(null);
  const [explanation, setExplanation] = useState<{ example: string; text: string; busy: boolean; error: string } | null>(null);
  useEffect(() => () => explanationRequest.current?.abort(), []);
  const [original, setOriginal] = useState(false), [retry, setRetry] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    Promise.all([api<Summary>('/menswear/case-study', undefined, controller.signal), api<{ examples: Example[] }>('/menswear/case-examples', undefined, controller.signal)])
      .then(([data, source]) => { if (!controller.signal.aborted) { setSummary(data); setExamples(source.examples); setError(''); } })
      .catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [retry]);
  const example = examples[index];
  useEffect(() => {
    if (!example) return;
    const controller = new AbortController();
    api<Result>('/menswear/case-utility', example.request, controller.signal)
      .then(data => { if (!controller.signal.aborted) { setResult(data); setError(''); } })
      .catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [example, retry]);
  async function explain() {
    if (!example || !result) return;
    explanationRequest.current?.abort();
    const controller = new AbortController(); explanationRequest.current = controller;
    const id = example.id;
    setExplanation({ example: id, text: '', busy: true, error: '' });
    try {
      const answer = await api<{ text: string; evaluationFingerprint: string }>('/menswear/case-explain', example.request, controller.signal);
      if (answer.evaluationFingerprint !== result.evaluationFingerprint) throw new Error('비교 결과를 다시 불러온 뒤 설명을 확인해주세요.');
      if (!controller.signal.aborted) setExplanation({ example: id, text: answer.text, busy: false, error: '' });
    } catch (e) { if (!controller.signal.aborted) setExplanation({ example: id, text: '', busy: false, error: (e as Error).message }); }
  }
  return <View style={st.container}>
    <View style={st.hero}><Tag blue>남성복 코디 사례 RAG</Tag><Text style={st.title}>새 옷이 더해 주는 조합을 비교해요.</Text><Text style={st.body}>기존 하의의 코디 기록을 먼저 확인하고, 새 후보가 내 상의들에 더해 주는 조합 근거를 비교합니다.</Text>
      {summary && <View style={st.stats}><View><Text style={st.number}>{summary.corpus.referencePairs.toLocaleString()}</Text><Text style={st.body}>검색에 사용하는 상·하의 사례</Text></View><View><Text style={st.number}>{summary.validation.cases}</Text><Text style={st.body}>의류가 겹치지 않는 평가 상황</Text></View></View>}
    </View>
    {Boolean(error) && <View style={st.card}><Text accessibilityRole="alert" style={st.error}>{error}</Text><Button secondary onPress={() => setRetry(v => v + 1)}>다시 불러오기</Button></View>}
    {!error && (!summary || !result) && <ActivityIndicator accessibilityLabel="코디 사례를 검색하고 있어요" color={C.blue} />}
    {summary && <View style={st.card}><Text style={st.heading}>추가 조합을 얻은 수준</Text><View style={st.resultRow}><Text style={st.body}>추가 기여 반영</Text><Text style={st.emphasis}>{(summary.validation.selectedRate * 100).toFixed(1)}%</Text></View><View style={st.resultRow}><Text style={st.body}>조합 지원만 비교</Text><Text style={st.value}>{(summary.validation.strongestComparisonRate * 100).toFixed(1)}%</Text></View><Text style={st.body}>{summary.validation.scope}. 기록에서 최대로 더할 수 있는 조합과 비교해, 선택한 후보가 얻은 수준입니다.</Text></View>}
    {example && result && <View style={st.card}><View style={st.resultRow}><Text style={st.heading}>{example.title}</Text><Button secondary onPress={() => { explanationRequest.current?.abort(); setResult(null); setOriginal(false); setIndex(v => (v + 1) % examples.length); }}>다른 예시</Button></View>
      <View style={st.inputBlock}><Text style={st.label}>보유 상의</Text><Text style={st.body}>{example.garments.contexts.map((g, i) => `${String.fromCharCode(65 + i)} ${garmentLabel(g.label)}`).join(' · ')}</Text><Text style={st.label}>이미 보유한 하의</Text><Text style={st.body}>{example.garments.owned.map(g => garmentLabel(g.label)).join(' · ')}</Text></View>
      {result.results.map(row => { const garment = example.garments.candidates.find(g => g.id === row.id); return <View key={row.id} style={[st.candidate, row.rank === 1 && st.preferred]}><View style={st.resultRow}><Text style={st.heading}>{`후보 ${String.fromCharCode(65 + example.garments.candidates.findIndex(g => g.id === row.id))} · ${garmentLabel(garment?.label || '하의')}`} </Text><Tag blue={row.rank === 1}>{row.alreadyOwned ? '보유 중' : row.rank === 1 ? '비교 우선' : '비교 후보'}</Tag></View><View style={st.resultRow}><Text style={st.body}>조합 활용도</Text><Text style={st.value}>{row.combinationSupport.toFixed(1)}</Text></View><View style={st.resultRow}><Text style={st.body}>추가 코디 기여</Text><Text style={st.emphasis}>{row.additionalContribution.toFixed(1)}</Text></View></View>; })}
      <Text style={st.body}>조합 활용도는 후보가 내 상의들과 얼마나 조합되는지를, 추가 코디 기여는 기존 옷의 코디 기록이 채우지 못한 부분에 후보가 더하는 근거를 보여줍니다.</Text>
      <Button secondary disabled={explanation?.example === example.id && explanation.busy} onPress={explain}>{explanation?.example === example.id && explanation.busy ? '검색 근거를 설명하고 있어요' : 'AI로 검색 근거 설명'}</Button>
      {explanation?.example === example.id && Boolean(explanation.text) && <Text style={st.body}>{explanation.text}</Text>}
      {explanation?.example === example.id && Boolean(explanation.error) && <Text accessibilityRole="alert" style={st.error}>{explanation.error}</Text>}
      <Button secondary onPress={() => setOriginal(v => !v)}>{original ? '원자료 접기' : '상품 원문과 검색한 코디 보기'}</Button>
      {original && <View style={st.inputBlock}>{example.garments.candidates.map(g => <View key={g.id}><Text style={st.label}>{garmentLabel(g.label)} · 원문 상품 정보</Text><Text selectable style={st.source}>{g.originalTitle}</Text></View>)}{result.results[0]?.evidence.slice(0, 2).map((e, i) => <View key={i}><Text style={st.label}>검색한 실제 코디</Text><Text selectable style={st.source}>{e.top.title}{'\n'}{e.bottom.title}</Text></View>)}</View>}
    </View>}
  </View>;
}

const st = StyleSheet.create({ container: { gap: 18 }, hero: { backgroundColor: '#F0F2FF', borderRadius: 20, padding: 24, gap: 14 }, title: { color: C.ink, fontFamily: font, fontSize: 24, lineHeight: 34, fontWeight: '800' }, body: { color: '#52617A', fontFamily: font, fontSize: 14, lineHeight: 23 }, stats: { flexDirection: 'row', flexWrap: 'wrap', gap: 28, marginTop: 10 }, number: { color: C.blue, fontFamily: font, fontSize: 28, fontWeight: '800' }, card: { backgroundColor: C.white, borderWidth: 1, borderColor: C.line, borderRadius: 18, padding: 22, gap: 18 }, heading: { color: C.ink, fontFamily: font, fontWeight: '700', fontSize: 17, lineHeight: 26, flexShrink: 1 }, resultRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 14 }, emphasis: { color: C.blue, fontFamily: font, fontSize: 23, fontWeight: '800' }, value: { color: C.ink, fontFamily: font, fontSize: 19, fontWeight: '700' }, inputBlock: { backgroundColor: C.bg, borderRadius: 12, padding: 18, gap: 8 }, label: { color: C.ink, fontFamily: font, fontSize: 14, fontWeight: '700', lineHeight: 22 }, candidate: { borderWidth: 1, borderColor: C.line, borderRadius: 14, padding: 18, gap: 14 }, preferred: { borderColor: '#B6BCFF', backgroundColor: '#F8F9FF' }, source: { color: C.muted, fontFamily: font, fontSize: 13, lineHeight: 22 }, error: { color: '#B54848', fontFamily: font, fontSize: 14, lineHeight: 22 } });

import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Linking, Pressable, StyleSheet, Text, TextInput, View } from 'react-native';
import { Shell } from '@/components/Shell';
import { Choices } from '@/components/Choices';
import { EvidenceCard } from '@/components/EvidenceCard';
import { RagStatus } from '@/components/RagStatus';
import { Button, Empty, Icon, Notice, Tag } from '@/components/ui';
import { C, font } from '@/components/theme';
import { api, type Source, type SearchResult, type Run } from '@/services/rag';
import { useRag } from '@/state/RagProvider';

const modes = { '속성 결합': 'conditioned', '결합 검색': 'hybrid', '의미 검색': 'vector', '키워드 검색': 'keyword' } as const;
const starters = ['RRF는 검색 순위를 어떻게 결합하는가?', '보유 의류의 중복을 활용도에 어떻게 반영하는가?', '블랙 상의와 베이지 하의 코디'];
export default function KnowledgeScreen() {
  const connection = useRag();
  const [tab, setTab] = useState('자료 검색'), [collection, setCollection] = useState('연구 자료');
  const [mode, setMode] = useState<keyof typeof modes>('속성 결합'), [query, setQuery] = useState('');
  const [result, setResult] = useState<SearchResult | null>(null), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const [inventory, setInventory] = useState<{ sources: Source[]; limitations: string[] } | null>(null);
  const [sourceKind, setSourceKind] = useState('전체'), [sourceQuery, setSourceQuery] = useState('');
  const [runs, setRuns] = useState<Run[]>([]);
  const activeSearch = useRef<AbortController | null>(null);
  const shownSources = (inventory?.sources || []).filter(source =>
    (sourceKind === '전체' || sourceKind === '논문' && source.kind === 'paper' || sourceKind === '코디 자료' && source.collection === 'style' || sourceKind === '설계 기록' && source.kind === 'project-document')
    && source.title.toLowerCase().includes(sourceQuery.trim().toLowerCase()))
    .sort((a, b) => sourceOrder(a) - sourceOrder(b));
  useEffect(() => {
    if (!connection.health) return;
    const controller = new AbortController();
    api<{ sources: Source[]; limitations: string[] }>('/sources', undefined, controller.signal).then(data => { if (!controller.signal.aborted) setInventory(data); })
      .catch((e: Error) => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [connection.health]);
  useEffect(() => () => activeSearch.current?.abort(), []);
  async function chooseTab(next: string) {
    setTab(next); setError('');
    if (next === '실행 기록') {
      try { setRuns((await api<{ runs: Run[] }>('/runs')).runs); } catch (e) { setError((e as Error).message); }
    }
  }
  async function search(text = query, target = collection) {
    if (text.trim().length < 2) { setError('두 글자 이상 검색어를 입력해주세요.'); return; }
    activeSearch.current?.abort();
    const controller = new AbortController();
    activeSearch.current = controller;
    setBusy(true); setError(''); setResult(null); setQuery(text); setCollection(target);
    try {
      const next = await api<SearchResult>('/ask', { query: text.trim(), collection: target === '연구 자료' ? 'research' : 'style', mode: modes[mode], limit: 5, includeHistorical: false }, controller.signal);
      if (!controller.signal.aborted) setResult(next);
    }
    catch (e) { if (!controller.signal.aborted) setError((e as Error).message); }
    finally { if (!controller.signal.aborted) setBusy(false); }
  }
  return <Shell title="근거 라이브러리" subtitle="추천 방법을 다룬 논문과 남성복 코디 근거를 출처와 함께 찾아보세요.">
    {tab !== '코디 사례 RAG' && <RagStatus />}
    {tab !== '코디 사례 RAG' && <View style={st.hero}><Text style={st.kicker}>RESEARCH & EVIDENCE</Text><Text style={st.heroTitle}>추천의 이유를,{ '\n' }자료에서 확인해요.</Text><Text style={st.heroCopy}>방법을 설명하는 연구 자료와 실제 조합에 적용할 코디 근거를 구분해 검색합니다.</Text><View style={st.stats}>{[['자료', connection.health?.sources], ['문서 구간', connection.health?.chunks], ['코디 주석', connection.health?.styleEvidence]].map(([label, value]) => <View key={String(label)}><Text style={st.number}>{value ?? '—'}</Text><Text style={st.statLabel}>{label}</Text></View>)}</View></View>}
    <View style={st.tabs}><Choices options={['자료 검색', '수집 자료', '실행 기록']} value={tab} onChange={chooseTab} /></View>
    {Boolean(error) && <Text accessibilityRole="alert" style={st.error}>{error}</Text>}
    {tab === '자료 검색' && <>
      {connection.health?.generation === 'nvidia' && <Notice>AI 설명을 위해 검색어와 공개 논문·코디 근거가 NVIDIA에 전송됩니다. 개인 정보나 비공개 자료는 검색어에 입력하지 마세요. 등록 사진과 내부 설계 문서는 전송하지 않습니다.</Notice>}
      <View style={st.searchBox}><View style={st.options}><Choices options={['연구 자료', '코디 근거']} value={collection} onChange={setCollection} /><Choices options={Object.keys(modes) as (keyof typeof modes)[]} value={mode} onChange={setMode} /></View><TextInput accessibilityLabel="근거 검색어" value={query} onChangeText={setQuery} placeholder="예: 검색 결합 방식, 중복 보정, 색상 조합" placeholderTextColor={C.muted} style={st.input} returnKeyType="search" onSubmitEditing={() => search()} /><Button icon="search-outline" onPress={() => search()} disabled={!connection.health || query.trim().length < 2} loading={busy}>근거 검색하기</Button></View>
      {!result && !busy && <View style={st.starters}><Text style={st.sectionTitle}>이런 내용부터 확인해보세요</Text>{starters.map((text, i) => <Pressable key={text} accessibilityRole="button" accessibilityLabel={text} disabled={!connection.health} onPress={() => search(text, i === 2 ? '코디 근거' : '연구 자료')} style={st.starter}><Text style={st.starterText}>{text}</Text><Icon name="arrow-forward" color={C.blue} size={17} /></Pressable>)}</View>}
      {busy && <View style={st.wait}><ActivityIndicator color={C.blue} /><Text style={st.waitCopy}>관련 자료를 검색하고 출처를 확인하는 중이에요.</Text></View>}
      {result && <View style={{ marginTop: 26 }}><View style={st.resultHeading}><Text style={st.sectionTitle}>검색한 근거 {result.evidence.length}개</Text><Tag>{result.elapsedMs} ms</Tag></View><Text style={st.waitCopy}>검색한 질문 · {result.query}</Text><Notice>{['nvidia', 'local-llm'].includes(result.answer.mode) ? '검색 근거에 따른 AI 설명입니다. 점수를 생성하지 않으며 원문과 일치하는지 확인해주세요.' : '현재는 검색한 자료의 발췌를 제공합니다. 생성 모델의 답변으로 표시하지 않습니다.'}</Notice><Text style={st.answer}>{result.answer.text}</Text>{Boolean(result.answer.warning) && <Text style={st.waitCopy}>{result.answer.warning}</Text>}{result.evidence.map(e => <EvidenceCard key={e.id} evidence={e} />)}{result.evidence.length === 0 && <Empty title="관련 근거를 찾지 못했어요" description="검색어와 자료 범위를 바꿔보세요." />}<Text style={st.trace}>검색 기록 {result.runId.slice(0, 12)}</Text></View>}
    </>}
    {tab === '수집 자료' && <>
      <Notice>현재 검색에 사용하는 방법론 논문과 남성복 코디 자료입니다. 각 자료의 원문 링크와 적용 범위를 함께 확인할 수 있어요.</Notice>
      <View style={{ marginTop: 22, gap: 14 }}><Choices options={['전체', '논문', '코디 자료']} value={sourceKind} onChange={setSourceKind} /><TextInput accessibilityLabel="수집 자료 이름 검색" value={sourceQuery} onChangeText={setSourceQuery} placeholder="자료 이름으로 찾기" placeholderTextColor={C.muted} style={st.input} /><Text style={st.small}>수집 자료 {shownSources.length}개</Text></View>
      <View style={{ marginTop: 14 }}>{shownSources.map(source => <View key={source.id} style={st.source}><View style={{ flex: 1 }}><View style={st.sourceTags}><Tag blue>{source.kind === 'paper' ? '논문' : source.collection === 'style' ? '코디 자료' : source.currentScope ? '현재 연구안' : '설계 기록'}</Tag><Text style={st.small}>{source.chunks}개 구간{source.pages ? ' · ' + source.pages + '쪽' : source.slides ? ' · ' + source.slides + '장' : ''}</Text></View><Text style={st.sourceName}>{source.title}</Text></View>{source.url.startsWith('https://') && <Pressable accessibilityRole="link" accessibilityLabel={source.title + ' 원문'} onPress={() => Linking.openURL(source.url).catch(() => setError('원문 링크를 열지 못했어요.'))} style={{ padding: 10 }}><Icon name="open-outline" size={18} color={C.blue} /></Pressable>}</View>)}</View>{inventory && shownSources.length === 0 && <Empty title="일치하는 자료가 없어요" description="자료 종류나 검색어를 바꿔보세요." />}{inventory?.limitations.map(text => <Text key={text} style={st.limit}>• {text}</Text>)}
    </>}
    {tab === '실행 기록' && <>{runs.length === 0 ? <Empty title="아직 분석 기록이 없어요" description="후보를 평가하거나 자료를 검색하면 실행 버전과 결과가 여기에 남아요." /> : runs.map(run => <View key={run.id} style={st.source}><View style={{ flex: 1 }}><Text style={st.sourceName}>{run.kind === 'assessment' ? '후보 평가 · ' + (run.candidateName || run.candidateId || '') : '자료 검색 · ' + (run.query || '')}</Text><Text style={st.small}>{new Date(run.created*1000).toLocaleString('ko-KR')} · {run.version}</Text><Text style={st.trace}>{run.id.slice(0, 12)}</Text></View><Tag blue>{run.score != null ? run.score + '점' : run.status ? '점수 보류' : '근거 조회'}</Tag></View>)}</>}
  </Shell>;
}
function sourceOrder(source: Source) {
  return source.kind === 'paper' ? 0 : source.currentScope ? 1 : source.collection === 'style' ? 2 : 3;
}
const st = StyleSheet.create({
  hero: { backgroundColor: C.navy, borderRadius: 22, padding: 28 }, kicker: { color: '#A6ABFF', fontFamily: font, fontSize: 10, letterSpacing: 1.4, fontWeight: '700' }, heroTitle: { color: C.white, fontFamily: font, fontSize: 28, fontWeight: '800', lineHeight: 40, marginTop: 15 }, heroCopy: { fontFamily: font, color: '#B1B9CC', fontSize: 12, lineHeight: 22, marginTop: 13, maxWidth: 550 }, stats: { flexDirection: 'row', gap: 38, marginTop: 26, paddingTop: 23, borderTopWidth: 1, borderTopColor: '#34405C' }, number: { fontFamily: font, color: C.white, fontSize: 27, fontWeight: '800' }, statLabel: { color: '#AAB2C7', fontFamily: font, fontSize: 10, marginTop: 7 }, tabs: { marginVertical: 26 }, searchBox: { borderWidth: 1, borderColor: C.line, backgroundColor: C.white, padding: 22, borderRadius: 18, gap: 18 }, options: { gap: 14 }, input: { borderWidth: 1, borderColor: C.line, borderRadius: 11, backgroundColor: C.bg, fontFamily: font, fontSize: 13, paddingHorizontal: 15, minHeight: 50, color: C.ink }, sectionTitle: { color: C.ink, fontFamily: font, fontSize: 16, fontWeight: '700' }, starters: { marginTop: 28 }, starter: { marginTop: 10, backgroundColor: C.white, borderRadius: 12, padding: 18, flexDirection: 'row', alignItems: 'center', gap: 15 }, starterText: { flex: 1, color: '#606C82', fontFamily: font, fontSize: 12, lineHeight: 20 }, wait: { padding: 34, alignItems: 'center', gap: 16 }, waitCopy: { color: C.muted, fontFamily: font, fontSize: 12, lineHeight: 21 }, resultHeading: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }, answer: { color: C.ink, fontFamily: font, fontSize: 13, lineHeight: 23, marginVertical: 20 }, error: { color: '#B54848', fontFamily: font, fontSize: 12, lineHeight: 21, marginBottom: 18 }, source: { borderWidth: 1, borderColor: C.line, borderRadius: 14, backgroundColor: C.white, marginBottom: 10, padding: 18, flexDirection: 'row', alignItems: 'center', gap: 12 }, sourceTags: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', gap: 10, marginBottom: 9 }, sourceName: { color: C.ink, fontFamily: font, fontSize: 13, fontWeight: '600', lineHeight: 21 }, small: { fontFamily: font, color: C.muted, fontSize: 10, lineHeight: 18, marginTop: 4 }, trace: { color: '#9CA3B2', fontFamily: font, fontSize: 10, marginTop: 9 }, limit: { fontFamily: font, fontSize: 11, lineHeight: 21, color: C.muted, marginTop: 12 },
});

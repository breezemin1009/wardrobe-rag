import { useState } from 'react';
import { router } from 'expo-router';
import { ActivityIndicator, Pressable, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { catalog } from '@/domain/demo';
import { occasions } from '@/domain/model';
import { useWardrobe } from '@/state/WardrobeProvider';
import { useRankings } from '@/services/useAssessment';
import { attributes } from '@/services/rag';
import { Shell } from '@/components/Shell';
import { GarmentPhoto } from '@/components/GarmentPhoto';
import { RagStatus } from '@/components/RagStatus';
import { Button, Empty, Icon, Notice, Tag } from '@/components/ui';
import { C, font } from '@/components/theme';
import { inMenswearScope, scopeReview } from '@/domain/audience';
import { ScopeReview } from '@/components/ScopeReview';

export default function RecommendationsScreen() {
  const store = useWardrobe(), { width } = useWindowDimensions();
  const [occasion] = useState<(typeof occasions)[number]>('데일리');
  const [error, setError] = useState('');
  const [page, setPage] = useState(0);
  const [gridWidth,setGridWidth]=useState(0);
  const columns=width>=1350?3:width>=640?2:1;
  const owned = store.garments.filter(g => g.role === 'owned'), ownedIds = new Set(owned.map(g => g.id));
  const candidateMap = new Map([...catalog, ...store.garments.filter(g => g.role === 'candidate')].map(g => [g.id, g]));
  const filtered = [...candidateMap.values()].filter(g => g.category === '하의' && inMenswearScope(g) && !ownedIds.has(g.id))
    .sort((a, b) => Number(a.isExample) - Number(b.isExample) || b.createdAt - a.createdAt || a.id.localeCompare(b.id));
  const pageCount = Math.max(1, Math.ceil(filtered.length / 12)), currentPage = Math.min(page, pageCount - 1);
  const candidates = filtered.slice(currentPage * 12, (currentPage + 1) * 12);
  const input = store.ready && candidates.length && !scopeReview(owned).length ? JSON.stringify({ candidates: candidates.map(attributes), wardrobe: owned.map(attributes), occasion }) : null;
  const { data, loading, error: rankingError, retry } = useRankings(input);
  const list = data ? data.results.map(r => candidateMap.get(r.candidateId)!).filter(Boolean) : candidates;
  return <Shell title={width < 640 ? '구매 추천' : '다음 한 벌, 구매 추천'} subtitle="사진으로 등록한 후보와 예시 카탈로그를 내 옷 기준으로 비교해요." action={<Button secondary icon="add" onPress={() => router.push({ pathname: '/register', params: { role: 'candidate' } })}>직접 등록</Button>}>
    <RagStatus />
    <ScopeReview garments={store.garments} />
    <Notice>보유 상의와 기존 하의를 기준으로 구매할 하의를 비교해요. 기존 옷에 새로 더하는 활용이 큰 후보부터 표시합니다.</Notice>
    {Boolean(error || rankingError) && <View style={st.errorBox}><Text accessibilityRole="alert" style={st.error}>{error || rankingError}</Text>{Boolean(rankingError) && <Button secondary onPress={retry}>다시 평가</Button>}</View>}
    <View style={st.count}><Text style={st.countCopy}>비교 후보 {list.length}벌 · 전체 {filtered.length}벌</Text>{loading ? <ActivityIndicator color={C.blue} size="small" /> : <Tag>{data ? '현재 옷장 기준' : '후보 목록'}</Tag>}</View>
    {pageCount > 1 && <View style={{ gap: 12, marginBottom: 20 }}><Notice>후보를 최대 12벌씩 비교해요. 순위는 현재 페이지의 후보 사이에서만 표시됩니다.</Notice><View style={{ flexDirection: 'row', gap: 10, alignItems: 'center' }}><Button secondary disabled={currentPage === 0} onPress={() => setPage(currentPage - 1)}>이전</Button><Text style={st.countCopy}>{currentPage + 1} / {pageCount}</Text><Button secondary disabled={currentPage + 1 === pageCount} onPress={() => setPage(currentPage + 1)}>다음</Button></View></View>}
    <View onLayout={e=>setGridWidth(e.nativeEvent.layout.width)} style={st.grid}>{list.map(g => {
      const result = data?.results.find(r => r.candidateId === g.id), saved = store.savedIds.includes(g.id);
      return <View key={g.id} style={[st.card, { width:gridWidth?Math.floor((gridWidth-20*(columns-1))/columns):'100%' }]}>
        <View style={st.cardHeader}><Tag blue>{result?.rank ? '추가 활용 순위 ' + result.rank : g.isExample ? '예시 카탈로그' : '등록한 후보'}</Tag><Pressable accessibilityRole="button" accessibilityLabel={g.name + (saved ? ' 저장 해제' : ' 저장')} onPress={() => store.toggleSaved(g.id).catch(e => setError(e.message))} style={{ padding: 5 }}><Icon name={saved ? 'bookmark' : 'bookmark-outline'} color={saved ? C.blue : C.muted} size={21} /></Pressable></View>
        <GarmentPhoto garment={g} height={205} />
        <View style={st.nameRow}><View style={{ flex: 1 }}><Text style={st.name}>{g.name}</Text><Text style={st.meta}>{g.category} · {g.color} · {g.pattern}</Text></View>{result?.score != null ? <Text style={st.score}>{result.score}<Text style={st.unit}> 점</Text></Text> : <Text style={st.pending}>{loading ? '분석 중' : result ? '점수 보류' : '평가 대기'}</Text>}</View>
        <View style={st.reason}><Icon name="layers-outline" size={16} color={C.blue} /><Text style={st.reasonCopy}>{result ? result.assessmentScope === 'public-case' ? `조합 활용 ${result.combination}점 · 추가 활용 ${result.contribution}점\n공개 코디 검색·학습 기반` : '조합 근거 ' + result.coverage + '% · 기존 의류 비교 ' + result.gainCoverage + '%\n' + result.evidenceQuality.label : '등록된 속성과 코디 근거를 함께 확인해보세요.'}</Text></View>
        <Button secondary icon="arrow-forward" onPress={() => router.push({ pathname: '/analysis/[id]', params: { id: g.id, occasion } })}>근거와 활용도 보기</Button>
      </View>;
    })}</View>
    {list.length === 0 && <Empty title="비교할 구매 후보가 없어요" description="사진으로 새 후보를 등록하거나 의류 종류를 바꿔보세요." />}
  </Shell>;
}
const st = StyleSheet.create({ controls: { padding: 22, backgroundColor: C.white, borderRadius: 3, borderWidth: 1, borderColor: C.line, flexDirection: 'row', flexWrap: 'wrap', gap: 25, marginBottom: 20 }, controlLabel: { fontFamily: font, fontSize: 11, color: C.muted, marginBottom: 11 }, count: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginVertical: 24 }, countCopy: { fontFamily: font, fontSize: 12, color: C.muted }, grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 20, justifyContent: 'flex-start' }, card: { borderRadius: 3, padding: 20, backgroundColor: C.white, borderWidth: 1, borderColor: C.line }, cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }, nameRow: { flexDirection: 'row', alignItems: 'center', gap: 8, marginTop: 20 }, name: { fontFamily: font, fontSize: 15, color: C.ink, fontWeight: '700' }, meta: { color: C.muted, fontFamily: font, fontSize: 11, marginTop: 8 }, score: { fontFamily: font, fontSize: 34, color: C.blue, fontWeight: '800', letterSpacing: -1 }, unit: { fontSize: 11, fontWeight: '500', letterSpacing: 0 }, reason: { flexDirection: 'row', gap: 8, marginVertical: 21 }, reasonCopy: { flex: 1, fontFamily: font, color: C.muted, fontSize: 11, lineHeight: 18 }, pending: { color: C.muted, fontSize: 11, fontFamily: font }, error: { color: '#B54848', fontFamily: font, fontSize: 12, lineHeight: 21 }, errorBox: { gap: 12, marginTop: 20 } });

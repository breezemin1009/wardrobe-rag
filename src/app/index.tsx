import { useState } from 'react';
import { router } from 'expo-router';
import { Pressable, ScrollView, StyleSheet, Text, TextInput, useWindowDimensions, View } from 'react-native';
import { Shell } from '@/components/Shell';
import { GarmentCard } from '@/components/GarmentCard';
import { GarmentPhoto } from '@/components/GarmentPhoto';
import { Choices } from '@/components/Choices';
import { Button, Empty, Icon, Sheet, Tag } from '@/components/ui';
import { C, font } from '@/components/theme';
import { useWardrobe } from '@/state/WardrobeProvider';
import { catalog } from '@/domain/demo';

import { ScopeReview } from '@/components/ScopeReview';

export default function WardrobeScreen() {
  const store = useWardrobe(), { width } = useWindowDimensions(), wide = width >= 1000;
  const [role, setRole] = useState<'owned' | 'candidate'>('owned');
  const [category, setCategory] = useState('전체');
  const [query, setQuery] = useState('');
  const [recent, setRecent] = useState(true);
  const [archiveOpen, setArchiveOpen] = useState(false);
  const [error, setError] = useState('');
  const [gridWidth,setGridWidth]=useState(0);
  const columns=width>=1200?4:width>=760?3:2;
  const owned = store.garments.filter(g => g.role === 'owned');
  const candidates = store.garments.filter(g => g.role === 'candidate');
  const items = store.garments.filter(g => g.role === role && (category === '전체' || g.category === category)
    && `${g.name} ${g.color} ${g.pattern}`.toLowerCase().includes(query.trim().toLowerCase()))
    .sort((a, b) => recent ? b.createdAt - a.createdAt : a.name.localeCompare(b.name, 'ko'));
  return <Shell title="내 옷장" subtitle="옷을 모으고, 다음 한 벌의 쓰임을 비교하세요." action={<Button icon="add" onPress={() => router.push({ pathname: '/register', params: { role } })}>옷 등록</Button>}>
    <ScopeReview garments={store.garments} />
    <View style={[st.layout, { flexDirection: 'column' }]}>
      <View style={st.inventory}>
        <View style={st.segment}><Pressable accessibilityRole="button" accessibilityState={{ selected: role === 'owned' }} onPress={() => setRole('owned')} style={[st.segmentItem, role === 'owned' && st.segmentSelected]}><Text style={[st.segmentText, role === 'owned' && { color: C.ink }]}>보유 옷 <Text style={{ color: C.blue }}>{owned.length}</Text></Text></Pressable><Pressable accessibilityRole="button" accessibilityState={{ selected: role === 'candidate' }} onPress={() => setRole('candidate')} style={[st.segmentItem, role === 'candidate' && st.segmentSelected]}><Text style={[st.segmentText, role === 'candidate' && { color: C.ink }]}>구매 후보 <Text style={{ color: C.blue }}>{candidates.length}</Text></Text></Pressable></View>
        <View style={st.search}><Icon name="search-outline" size={18} color={C.muted} /><TextInput accessibilityLabel="옷 검색" value={query} onChangeText={setQuery} placeholder="이름, 색상으로 옷 찾기" placeholderTextColor="#A3A7B6" style={st.searchInput} /></View>
        <View style={st.filters}><Choices options={['전체', '상의', '하의']} value={category} onChange={setCategory} /><Pressable accessibilityRole="button" accessibilityLabel="정렬 변경" onPress={() => setRecent(!recent)} style={st.sort}><Text style={st.sortText}>{recent ? '최근 등록순' : '이름순'}</Text><Icon name="swap-vertical" size={13} color={C.muted} /></Pressable></View>
        <View style={st.countLine}><Text style={st.countCopy}>{items.length}벌의 {role === 'owned' ? '보유 의류' : '구매 후보'}</Text><Tag>{items.some(g => g.isExample) ? '예시 포함' : '등록한 의류'}</Tag></View>
        {items.length ? <View onLayout={e=>setGridWidth(e.nativeEvent.layout.width)} style={st.grid}>{items.map(g => <GarmentCard key={g.id} garment={g} width={gridWidth?Math.floor((gridWidth-13*(columns-1))/columns):'48%'} />)}</View> : <Empty title="아직 표시할 옷이 없어요" description={query ? '검색어나 의류 종류를 바꿔보세요.' : '사진 한 장으로 첫 번째 옷을 등록해보세요.'} action={!query && <Button icon="add" onPress={() => router.push({ pathname: '/register', params: { role } })}>옷 등록하기</Button>} />}
        {store.archived.length > 0 && <Pressable accessibilityRole="button" onPress={() => setArchiveOpen(true)} style={st.archive}><Icon name="refresh-outline" size={15} color={C.muted} /><Text style={st.archiveText}>제외한 옷 {store.archived.length}벌 · 복원하기</Text></Pressable>}
      </View>
      <View style={[st.aside, { width: '100%', flexDirection: wide ? 'row' : 'column' }]}>
        <View style={[st.hero, {flex:1}]}><Text style={st.heroKicker}>YOUR NEXT PIECE</Text><Text style={st.heroTitle}>새 옷을 사기 전에,{ '\n'}내 옷과 먼저.</Text><Text style={st.heroCopy}>보유 옷과의 조합과 중복을{ '\n'}함께 확인해보세요.</Text><View style={st.heroPhoto}><GarmentPhoto garment={catalog[0]} height={166} /></View><Button onPress={() => router.push('/recommendations')} icon="arrow-forward">후보 근거 확인하기</Button></View>
        <View style={[st.summary, {flex:1}]}><View style={st.summaryTitle}><Text style={st.summaryHeading}>옷장 한눈에 보기</Text><Icon name="layers-outline" color={C.muted} size={17} /></View><View style={st.statistics}>{['상의', '하의'].map(c => <View key={c} style={st.stat}><Text style={st.statNumber}>{owned.filter(g => g.category === c).length}<Text style={st.statUnit}> 벌</Text></Text><Text style={st.statLabel}>{c}</Text></View>)}</View><View style={st.summaryRule} /><Text style={st.summaryCopy}>내 옷을 등록할수록{ '\n'}비교할 수 있는 조합이 늘어나요.</Text><Pressable accessibilityRole="button" onPress={() => router.push('/recommendations')} style={st.summaryLink}><Text style={st.summaryLinkText}>구매 추천 둘러보기</Text><Icon name="arrow-forward" size={16} color={C.blue} /></Pressable></View>
      </View>
    </View>
    <Sheet title="제외한 옷 복원" visible={archiveOpen} onClose={() => setArchiveOpen(false)}><ScrollView>{store.archived.map(g => <View key={g.id} style={st.restoreRow}><Text style={{ flex: 1, color: C.ink, fontFamily: font }}>{g.name}</Text><Button secondary onPress={() => store.restore(g.id).catch(e => setError(e.message))}>복원</Button></View>)}</ScrollView>{Boolean(error) && <Text style={{ color: '#B54848' }}>{error}</Text>}</Sheet>
  </Shell>;
}
const st = StyleSheet.create({
  layout: { flexDirection: 'row', gap: 28 }, inventory: { flex: 1, minWidth: 0 }, segment: { flexDirection: 'row', backgroundColor: '#E7E8DE', borderRadius: 12, padding: 4 }, segmentItem: { flex: 1, padding: 13, borderRadius: 9, alignItems: 'center' }, segmentSelected: { backgroundColor: C.white }, segmentText: { fontFamily: font, color: C.muted, fontSize: 14, fontWeight: '700' }, search: { flexDirection: 'row', alignItems: 'center', backgroundColor: C.white, borderWidth: 1, borderColor: C.line, borderRadius: 12, paddingHorizontal: 15, height: 47, marginTop: 12, gap: 9 }, searchInput: { flex: 1, fontFamily: font, color: C.ink, fontSize: 12, height: '100%' }, filters: { marginTop: 10, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 8 }, sort: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingVertical: 10 }, sortText: { color: C.muted, fontSize: 10, fontFamily: font }, countLine: { marginTop: 12, marginBottom: 10, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' }, countCopy: { fontFamily: font, color: C.muted, fontSize: 11 }, grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 13, justifyContent: 'flex-start' }, aside: { width: 278, gap: 20 }, hero: { backgroundColor: C.navy, padding: 23, borderRadius: 3 }, heroKicker: { fontFamily: font, color: '#C9D6B7', fontSize: 9, fontWeight: '700', letterSpacing: 1.8 }, heroTitle: { fontFamily: font, color: C.white, fontSize: 23, lineHeight: 34, fontWeight: '700', marginTop: 17, letterSpacing: -.8 }, heroCopy: { fontFamily: font, color: '#A7AFC4', fontSize: 12, lineHeight: 21, marginTop: 11 }, heroPhoto: { marginTop: 20, marginBottom: 20, backgroundColor: '#F2F3F7', borderRadius: 14, padding: 8 }, summary: { backgroundColor: C.white, borderWidth: 1, borderColor: C.line, borderRadius: 3, padding: 22 }, summaryTitle: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }, summaryHeading: { color: C.ink, fontFamily: font, fontWeight: '700', fontSize: 13 }, statistics: { flexDirection: 'row', marginTop: 24 }, stat: { flex: 1 }, statNumber: { fontFamily: font, fontSize: 28, fontWeight: '800', color: C.ink }, statUnit: { fontSize: 11, color: C.muted, fontWeight: '400' }, statLabel: { fontFamily: font, color: C.muted, fontSize: 11, marginTop: 5 }, summaryRule: { height: 1, backgroundColor: C.line, marginVertical: 22 }, summaryCopy: { fontFamily: font, color: C.muted, fontSize: 12, lineHeight: 21 }, summaryLink: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 18 }, summaryLinkText: { fontFamily: font, color: C.blue, fontSize: 12, fontWeight: '700' }, archive: { flexDirection: 'row', gap: 7, alignSelf: 'center', padding: 18, alignItems: 'center' }, archiveText: { fontFamily: font, fontSize: 11, color: C.muted }, restoreRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginBottom: 13 },
});



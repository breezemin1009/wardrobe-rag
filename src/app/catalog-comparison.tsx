import { useEffect, useState } from 'react';
import { router } from 'expo-router';
import { ActivityIndicator, Image, Linking, StyleSheet, Text, TextInput, View, useWindowDimensions } from 'react-native';
import { Shell } from '@/components/Shell';
import { Choices } from '@/components/Choices';
import { Button, Notice, Tag } from '@/components/ui';
import { C, font } from '@/components/theme';
import { api, API_URL } from '@/services/rag';

type Item = { id: string; name: string; photo: string; category: string; audience: 'mens' | 'unisex'; baseColour: string; articleType: string; photoKind: 'original' | 'thumbnail' };
type ProductPage = { items: Item[]; total: number; metadataEligible: number; source: string; use: string };
type Outfit = { id: string; season: string; occasion: string; sourceUrl: string; items: { id: string; name: string; category: string; color: string | null; pattern: string | null; originalColor: string; originalPattern: string; audience: string }[] };
type OutfitPage = { outfits: Outfit[]; total: number; audit: { apparelItems: number; observedPairs: number } };
const types: Record<string, string> = { Shirts: '셔츠', Tshirts: '티셔츠', Jeans: '청바지', Trousers: '팬츠', 'Track Pants': '트레이닝 팬츠', Shorts: '반바지', Sweatshirts: '스웨트셔츠', Sweaters: '니트', Jackets: '재킷', Blazers: '블레이저', Waistcoat: '베스트' };
const occasions: Record<string, string> = { Home: '일상', Travel: '여행', Social: '모임', School: '학교', Sports: '운동' };

export default function MenswearDataScreen() {
  const { width } = useWindowDimensions(), mobile = width < 640;
  const [tab, setTab] = useState('상품 사진'), [category, setCategory] = useState('전체'), [audience, setAudience] = useState('전체');
  const [query, setQuery] = useState(''), [offset, setOffset] = useState(0), [attempt, setAttempt] = useState(0);
  const key = JSON.stringify([tab, category, audience, query, offset, attempt]);
  const [state, setState] = useState<{ key: string; products: ProductPage | null; outfits: OutfitPage | null; error: string }>({ key: '', products: null, outfits: null, error: '' });
  const [linkError, setError] = useState('');
  const current = state.key === key, loading = !current;
  const products = current ? state.products : null, outfits = current ? state.outfits : null, error = current ? state.error || linkError : '';
  useEffect(() => {
    const controller = new AbortController();
    const params = new URLSearchParams({ offset: String(offset), limit: tab === '상품 사진' ? '12' : '6' });
    if (category !== '전체') params.set('category', category);
    if (audience !== '전체') params.set('audience', audience === '남성복' ? 'mens' : 'unisex');
    if (query.trim()) params.set('query', query.trim());
    const request = tab === '상품 사진'
      ? api<ProductPage>('/menswear/catalog?' + params, undefined, controller.signal).then(data => { if (!controller.signal.aborted) setState({ key, products: data, outfits: null, error: '' }); })
      : api<OutfitPage>('/menswear/outfits?offset=' + offset + '&limit=6', undefined, controller.signal).then(data => { if (!controller.signal.aborted) setState({ key, products: null, outfits: data, error: '' }); });
    request.catch(e => { if (!controller.signal.aborted) setState({ key, products: null, outfits: null, error: e.message }); });
    return () => controller.abort();
  }, [tab, category, audience, query, offset, key]);
  const total = products?.total ?? outfits?.total ?? 0, step = tab === '상품 사진' ? 12 : 6;
  return <Shell title="남성복 데이터 둘러보기" subtitle="남성복·공용 표기를 확인한 상품과 실제 코디 기록을 살펴보세요." back>
    <View style={st.hero}><Tag blue>MEN’S WARDROBE</Tag><Text style={st.title}>상품 사진과 코디 기록</Text><Text style={st.body}>상품 사진은 의류 정보를 확인하는 데 사용하고, 실제 조합은 출처가 연결된 코디 기록에서 찾습니다.</Text></View>
    <View style={st.controls}><View style={st.filterGroup}><Text style={st.filterLabel}>자료 구분</Text><Choices options={['상품 사진', '실제 코디 기록']} value={tab} onChange={v => { setTab(v); setOffset(0); }} /></View>
      {tab === '상품 사진' && <><View style={st.filterGroup}><Text style={st.filterLabel}>의류 종류</Text><Choices options={['전체', '상의', '하의']} value={category} onChange={v => { setCategory(v); setOffset(0); }} /></View><View style={st.filterGroup}><Text style={st.filterLabel}>의류 구분</Text><Choices options={['전체', '남성복', '공용']} value={audience} onChange={v => { setAudience(v); setOffset(0); }} /></View><TextInput accessibilityLabel="남성복 상품 검색" maxLength={80} value={query} onChangeText={v => { setQuery(v); setOffset(0); }} placeholder="상품명 검색 · shirt, jeans, black" placeholderTextColor={C.muted} style={st.input} /></>}
    </View>
    {loading && <View style={st.pending}><ActivityIndicator color={C.blue} /><Text style={st.body}>확인한 자료를 불러오고 있어요.</Text></View>}
    {Boolean(error) && <View style={st.panel}><Text accessibilityRole="alert" style={st.error}>{error}</Text><Button secondary onPress={() => setAttempt(v => v + 1)}>다시 불러오기</Button></View>}
    {products && <><Text style={st.heading}>사진까지 대조한 상품 {products.total.toLocaleString()}개</Text><Text style={st.body}>남성복·공용 표기와 실제 의류 사진을 함께 확인한 상품입니다. 색상·무늬·로고는 등록할 때 사진과 비교해주세요.</Text><View style={st.grid}>{products.items.map(item => <View key={item.id} style={[st.product, { width: mobile ? '47%' : '23%' }]}><Tag blue>{item.audience === 'mens' ? '남성복' : '공용'}</Tag><Image source={{ uri: API_URL + item.photo }} accessibilityLabel={item.name} resizeMode="contain" style={st.photo} /><Text style={st.heading}>{types[item.articleType] || item.category}</Text><Text style={st.name}>{item.photoKind === 'original' ? '원본 사진' : '상품 썸네일'}</Text><Text style={st.name} numberOfLines={3}>{item.name}</Text><Button secondary onPress={() => router.push({ pathname: '/register', params: { productId: item.id } })} style={{ marginTop: 14 }}>후보로 등록</Button></View>)}</View>{products.items.length === 0 && <Notice>조건에 맞는 상품이 없어요. 검색어나 의류 구분을 바꿔보세요.</Notice>}<Text style={st.body}>사진·상품 표기 출처 · {products.source}</Text></>}
    {outfits && <><View style={st.stats}><View><Text style={st.number}>{outfits.total}</Text><Text style={st.body}>검증한 남성복 코디</Text></View><View><Text style={st.number}>{outfits.audit.observedPairs}</Text><Text style={st.body}>코디 속 상·하의 연결</Text></View></View><Text style={st.body}>같은 코디에 기록된 상의와 하의를 보여줍니다. 각 의류의 남성복·공용 표기를 대조해 확인했습니다.</Text><View style={st.grid}>{outfits.outfits.map(outfit => <View key={outfit.id} style={[st.panel, { width: mobile ? '100%' : '48%' }]}><View style={st.row}><Tag blue>{occasions[outfit.occasion] || outfit.occasion}</Tag><Text style={st.heading}>코디 {outfit.id}</Text></View>{outfit.items.map(item => <View key={item.id} style={st.outfitItem}><Tag>{item.category}</Tag><View style={{ flex: 1 }}><Text style={st.heading}>{item.color || item.originalColor} · {item.pattern || '원문 디자인'}</Text><Text style={st.name}>{item.name}</Text>{!item.pattern && <Text style={st.name}>디자인 표기: {item.originalPattern}</Text>}<Text style={st.name}>{item.audience === 'mens' ? '남성복' : '공용'} · 원본 항목 {item.id}</Text></View></View>)}<Button secondary icon="open-outline" onPress={() => Linking.openURL(outfit.sourceUrl).catch(() => setError('출처 링크를 열지 못했어요.'))}>원본 코디 기록 보기</Button></View>)}</View><Text style={st.body}>FashionStylist 저자 공개 자료 · CC BY-NC 4.0 · 항목 선별 및 속성 정규화 적용</Text></>}
    {total > step && <View style={st.pagination}><Button secondary disabled={offset === 0 || loading} onPress={() => setOffset(Math.max(0, offset - step))}>이전</Button><Text style={st.name}>{Math.floor(offset / step) + 1} / {Math.ceil(total / step)}</Text><Button secondary disabled={offset + step >= total || loading} onPress={() => setOffset(offset + step)}>다음</Button></View>}
  </Shell>;
}
const st = StyleSheet.create({
  hero: { backgroundColor: C.white, borderRadius: 20, borderWidth: 1, borderColor: C.line, padding: 20, alignItems: 'flex-start' },
  title: { fontFamily: font, color: C.ink, fontSize: 25, lineHeight: 34, fontWeight: '800', marginTop: 16 },
  body: { fontFamily: font, color: C.muted, fontSize: 13, lineHeight: 23, marginTop: 10 },
  controls: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', gap: 15, marginVertical: 20 }, input: { width: '100%', minHeight: 48, paddingHorizontal: 15, fontFamily: font, fontSize: 14, borderRadius: 12, borderWidth: 1, borderColor: C.line, backgroundColor: C.white, color: C.ink },
  grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 16, justifyContent: 'space-between', marginTop: 20 },
  product: { padding: 15, borderWidth: 1, borderColor: C.line, borderRadius: 16, backgroundColor: C.white },
  photo: { height: 160, width: '100%', marginVertical: 16 }, heading: { fontFamily: font, fontSize: 15, color: C.ink, lineHeight: 23, fontWeight: '700' },
  name: { fontFamily: font, fontSize: 12, lineHeight: 20, color: C.muted, marginTop: 6 },
  pending: { alignItems: 'center', padding: 30 }, panel: { padding: 22, borderRadius: 18, backgroundColor: C.white, borderWidth: 1, borderColor: C.line, gap: 16 },
  stats: { flexDirection: 'row', gap: 40, padding: 24, borderRadius: 18, backgroundColor: C.light }, number: { fontFamily: font, fontWeight: '800', fontSize: 30, color: C.blue },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }, outfitItem: { flexDirection: 'row', gap: 12, borderTopWidth: 1, borderTopColor: C.line, paddingTop: 15 },
  filterGroup: { gap: 8 }, filterLabel: { fontFamily: font, fontSize: 11, color: C.muted },
  pagination: { flexDirection: 'row', alignItems: 'center', gap: 20, justifyContent: 'center', marginTop: 28 }, error: { color: '#B54848', fontFamily: font, fontSize: 14 },
});

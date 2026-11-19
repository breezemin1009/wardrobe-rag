import { useEffect, useRef, useState } from 'react';
import { router, useLocalSearchParams } from 'expo-router';
import { Image, Platform, Pressable, StyleSheet, Text, TextInput, useWindowDimensions, View } from 'react-native';
import { Shell } from '@/components/Shell';
import { GarmentPhoto } from '@/components/GarmentPhoto';
import { Choices } from '@/components/Choices';
import { Button, Empty, Icon, Notice, Loading } from '@/components/ui';
import { C, font } from '@/components/theme';
import { useWardrobe } from '@/state/WardrobeProvider';
import { categories, colors, occasions, patterns, type Garment } from '@/domain/model';
import { selectPhoto } from '@/services/photos';
import { persistPhoto } from '@/services/persist-photo';
import { fingerprintPhoto } from '@/services/photo-fingerprint';
import { duplicatePhoto } from '@/domain/rules';
import { PhotoAnalysis } from '@/components/PhotoAnalysis';
import { PhotoCutout } from '@/components/PhotoCutout';
import { DesignEditor } from '@/components/DesignEditor';
import { emptyView,legacyDesignColour,designCategory,garmentTypeLabels,type DesignProfile } from '@/domain/design';
import { importCatalogProduct } from '@/services/catalog-import';

export default function RegisterScreen() {
  const { ready } = useWardrobe();
  const params = useLocalSearchParams<{ id?: string; role?: string; productId?: string }>();
  if (!ready) return <Shell title="의류 정보 불러오는 중" back>{null}</Shell>;
  if (params.productId && !params.id) return <CatalogRegister key={params.productId} productId={params.productId} />;
  return <RegisterForm key={params.id || params.role || 'new-owned'} />;
}
function CatalogRegister({ productId }: { productId: string }) {
  const [seed, setSeed] = useState<Garment | null>(null), [error, setError] = useState('');
  useEffect(() => {
    const controller = new AbortController();
    importCatalogProduct(productId, controller.signal)
      .then(g => { if (!controller.signal.aborted) setSeed(g); })
      .catch(e => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [productId]);
  if (error) return <Shell title="상품 확인" back><Empty title="상품을 불러오지 못했어요" description={error} action={<Button onPress={() => router.replace('/catalog-comparison')}>상품 목록으로</Button>} /></Shell>;
  if (!seed) return <Shell title="남성복 상품 불러오는 중" back><Loading /></Shell>;
  return <RegisterForm seed={seed} />;
}
function RegisterForm({ seed }: { seed?: Garment }) {
  const params = useLocalSearchParams<{ id?: string; role?: string }>(), store = useWardrobe(), { width } = useWindowDimensions();
  const existing = store.garments.find(g => g.id === params.id);
  const [draft, setDraft] = useState<Garment>(() => existing ? { ...existing } : seed || { id: '', name: '', audience: 'unknown', role: params.role === 'candidate' ? 'candidate' : 'owned', category: params.role === 'candidate' ? '하의' : '상의', color: '화이트', pattern: '무지', occasion: '데일리', isExample: false, createdAt: Date.now() });
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [extracting, setExtracting] = useState(false);
  const [confirmed, setConfirmed] = useState(Boolean(existing));
  const [photoRevision,setPhotoRevision]=useState(0);
  const operation = useRef(false);
  const duplicate = duplicatePhoto(store, draft);
  function change<K extends keyof Garment>(key: K, value: Garment[K]) { if (!operation.current) {
    setDraft(g => {
      let design=g.design;
      if (design && key==='color') design={...design,front:{...design.front,baseColor:value as Garment['color']}};
      if (design && key==='pattern') design={...design,front:{...design.front,pattern:value as Garment['pattern'],patternColors:[],patternScale:value==='무지'?'none':'unknown'}};
      if (design && key==='backUri') design={...design,back:null,backStatus:'unknown'};
      if (design && key==='category' && designCategory(design.garmentType) && designCategory(design.garmentType)!==value) design={...design,garmentType:'unknown'};
      return {...g,[key]:value,design,referenceProductId:['category','color','pattern','name'].includes(key)?undefined:g.referenceProductId};
    });
    if (['category','color','pattern','design','backUri','audience'].includes(key)) setConfirmed(false);
    if (['backUri','cutoutUri','category','color','pattern'].includes(key)) setPhotoRevision(v=>v+1);
  } }
  function updateDesign(design:DesignProfile) { if (!operation.current) {setDraft(g=>({...g,referenceProductId:undefined,design,category:designCategory(design.garmentType)||g.category,color:legacyDesignColour(design.front.baseColor),pattern:design.front.pattern}));setConfirmed(false);setPhotoRevision(v=>v+1);} }
  async function pick(back = false, camera = false) {
    if (operation.current || extracting) return;
    operation.current = true; setBusy(true);
    try {
      const uri = await selectPhoto(camera);
      if (!uri) return;
      const imageHash = back ? undefined : await fingerprintPhoto(uri);
      setDraft(g => back ? { ...g, backUri: uri,design:g.design?{...g.design,back:null,backStatus:'unknown'}:undefined,isExample: false } : { ...g, sourceProductId: undefined, referenceProductId: undefined, audience: 'unknown', imageUri: uri, imageHash, cutoutUri: undefined, demoAsset: undefined,design:{version:1,front:{...emptyView(g.color,g.pattern),uncertain:['baseColor','pattern','patternColors','patternScale','contrast','accents']},back:null,backStatus:'unknown'},isExample: false });
      setConfirmed(false);setPhotoRevision(v=>v+1);
      setError('');
    } catch (e) { setError((e as Error).message); }
    finally { operation.current = false; setBusy(false); }
  }
  async function submit() {
    if (operation.current || extracting) return;
    if (draft.role === 'candidate' && draft.category !== '하의') { setError('현재 구매 후보 비교는 하의를 지원해요. 상의는 보유 옷으로 등록해주세요.'); return; }
    if (!draft.design?.garmentType || !['tshirt','shirt','knit','jacket','jeans','trousers','shorts'].includes(draft.design.garmentType)) { setError('코디 검색에 사용할 세부 종류를 선택해주세요.'); return; }
    if (!draft.name.trim()) { setError('옷 이름을 입력해주세요.'); return; }
    if (!draft.imageUri && !draft.demoAsset) { setError('앞면 사진을 선택해주세요.'); return; }
    if (!draft.demoAsset && !confirmed) { setError('사진과 종류·색상·무늬를 비교한 뒤 확인해주세요.'); return; }
    if (duplicate) { setError('동일 사진이 등록되어 있어요. 아래에서 기존 의류를 확인해주세요.'); return; }
    operation.current = true;
    setBusy(true); setError('');
    try {
      const id = draft.id || `garment-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const imageUri = await persistPhoto(draft.imageUri), backUri = await persistPhoto(draft.backUri), cutoutUri = await persistPhoto(draft.cutoutUri);
      await store.save({ ...draft, id, name: draft.name.trim(), imageUri, backUri, cutoutUri });
      router.replace(`/garment/${id}`);
    } catch (e) { setError((e as Error).message); }
    finally { operation.current = false; setBusy(false); }
  }
  if (params.id && !existing) return <Shell title="의류를 찾을 수 없어요" back><Empty title="옷장을 확인해주세요" description="제외한 옷을 복원하거나 수정할 의류를 다시 선택해주세요." action={<Button onPress={() => router.replace('/')}>내 옷장으로</Button>} /></Shell>;
  return <Shell title={existing ? '의류 정보 수정' : '새 의류 등록'} subtitle="사진과 의류 정보를 확인하고 옷장에 보관하세요." back>
    <View style={[st.layout, width < 760 && { flexDirection: 'column' }]}>
      <View style={[st.photoPanel, width < 760 && { width: '100%' }]}><View style={st.panelHeading}><Text style={st.panelTitle}>의류 사진</Text><Text style={st.required}>앞면 필수</Text></View>
        {draft.imageUri || draft.demoAsset ? <GarmentPhoto garment={draft} height={280} /> : <Pressable accessibilityRole="button" accessibilityLabel="앞면 사진 선택" onPress={() => pick()} style={st.photoEmpty}><Icon name="camera-outline" size={36} color={C.blue} /><Text style={st.photoTitle}>옷 사진을 선택해주세요</Text><Text style={st.photoCopy}>옷 전체가 보이는 사진이 좋아요.</Text></Pressable>}
        <View style={st.photoButtons}><Button secondary icon="images-outline" disabled={busy || extracting} onPress={() => pick()} style={{ flex: 1 }}>사진 선택</Button>{Platform.OS !== 'web' && <Button secondary icon="camera-outline" disabled={busy || extracting} onPress={() => pick(false, true)} style={{ flex: 1 }}>촬영</Button>}</View>
        <View style={st.backRow}>{draft.backUri ? <Image accessibilityLabel="뒷면 사진" source={{ uri: draft.backUri }} resizeMode="contain" style={st.backImage} /> : <View style={st.backPlaceholder}><Icon name="image-outline" color={C.muted} /></View>}<View style={{ flex: 1 }}><Text style={st.backTitle}>뒷면 사진</Text><Text style={st.backCopy}>선택 사항 · 같은 옷으로 저장</Text></View><Pressable accessibilityRole="button" accessibilityLabel="뒷면 사진 선택" onPress={() => pick(true)} style={st.smallAction}><Icon name="add" color={C.blue} size={21} /></Pressable>{draft.backUri && <Pressable accessibilityRole="button" accessibilityLabel="뒷면 선택 취소" onPress={() => change('backUri', undefined)} style={st.smallAction}><Icon name="close" color={C.muted} size={20} /></Pressable>}</View>
        {draft.imageUri && <PhotoCutout key={draft.imageUri} uri={draft.imageUri} disabled={busy} applied={Boolean(draft.cutoutUri)} onBusy={setExtracting} onApply={value => change('cutoutUri', value)} onRestore={() => change('cutoutUri', undefined)} />}
        {!existing && <Pressable accessibilityRole="button" disabled={busy || extracting} onPress={() => { setDraft(g => ({ ...g, name: g.name || (g.role === 'candidate' ? '베이지 바지' : '화이트 셔츠'), audience: 'mens', sourceProductId: undefined, demoAsset: g.role === 'candidate' ? 'beige-pants' : 'white-shirt', imageUri: undefined, cutoutUri: undefined, imageHash: undefined,backUri:undefined,referenceProductId:undefined,design:{version:1,garmentType:g.role === 'candidate'?'trousers':'shirt',front:emptyView(g.role === 'candidate'?'베이지':'화이트','무지'),back:null,backStatus:'unknown'}, category: g.role === 'candidate' ? '하의' : '상의', color: g.role === 'candidate' ? '베이지' : '화이트', pattern: '무지', isExample: true }));setPhotoRevision(v=>v+1); setConfirmed(true); setError(''); }} style={st.sampleButton}><Text style={st.sampleCopy}>사진 없이 둘러보기 · 예시 사진 사용</Text></Pressable>}
      </View>
      <View style={st.formPanel}><Text style={st.panelTitle}>의류 정보</Text><Text style={st.formIntro}>사진의 종류·색상·무늬를 직접 확인해주세요. 확인한 정보로 코디 근거를 검색합니다.</Text>
        <View style={st.field}><Text style={st.label}>의류 구분</Text>{draft.sourceProductId ? <Notice>{`원본 상품 표기로 확인: ${draft.audience === 'mens' ? '남성복' : '공용'} · 사진의 무늬·로고는 아래에서 확인해주세요.`}</Notice> : <Choices options={['남성복','공용','확인 필요']} value={draft.audience === 'mens' ? '남성복' : draft.audience === 'unisex' ? '공용' : '확인 필요'} onChange={label => change('audience', label === '남성복' ? 'mens' : label === '공용' ? 'unisex' : 'unknown')} />}<Text style={st.formIntro}>상품 설명이나 의류 표기를 기준으로 선택해주세요. 남성복·공용으로 확인한 의류끼리 활용도를 비교합니다.</Text></View>
        {draft.imageUri && <PhotoAnalysis key={photoRevision} uri={draft.cutoutUri || draft.imageUri} backUri={draft.backUri} processed={Boolean(draft.cutoutUri)} disabled={busy || extracting} onApply={value => { if (!operation.current) { setDraft(g => ({ ...g, ...value, referenceProductId:undefined })); setConfirmed(false); } }} />}
        <View style={st.field}><Text style={st.label}>등록 위치</Text><Choices options={['보유 옷', '구매 후보']} value={draft.role === 'owned' ? '보유 옷' : '구매 후보'} onChange={v => change('role', v === '보유 옷' ? 'owned' : 'candidate')} /></View>
        <View style={st.field}><Text style={st.label}>옷 이름 <Text style={st.required}>*</Text></Text><TextInput accessibilityLabel="옷 이름" value={draft.name} onChangeText={v => change('name', v)} maxLength={60} placeholder="예: 베이지 코튼 팬츠" placeholderTextColor="#A3A7B6" style={st.input} /></View>
        <View style={st.field}><Text style={st.label}>종류</Text><Choices options={draft.role === 'candidate' ? ['하의'] : categories} value={draft.category} onChange={v => change('category', v)} /></View>
        <View style={st.field}><Text style={st.label}>세부 종류</Text><Choices options={(draft.category === '상의' ? ['tshirt','shirt','knit','jacket'] as const : ['jeans','trousers','shorts'] as const).map(k=>garmentTypeLabels[k])} value={garmentTypeLabels[draft.design?.garmentType || 'unknown']} onChange={label=>{const garmentType=(['tshirt','shirt','knit','jacket','jeans','trousers','shorts'] as const).find(k=>garmentTypeLabels[k]===label)!;updateDesign({...draft.design,version:1,garmentType,front:draft.design?.front || emptyView(draft.color,draft.pattern),back:draft.design?.back || null,backStatus:draft.design?.backStatus || 'unknown'});}} /></View>
        <View style={st.field}><Text style={st.label}>주요 색상</Text><Choices options={colors} value={draft.color} onChange={v => change('color', v)} /></View>
        <View style={st.field}><Text style={st.label}>무늬</Text><Choices options={patterns} value={draft.pattern} onChange={v => change('pattern', v)} /></View>
        <View style={st.field}><Text style={st.label}>주로 입는 상황</Text><Choices options={occasions} value={draft.occasion} onChange={v => change('occasion', v)} /></View>
        {(draft.imageUri || draft.demoAsset) && <DesignEditor value={draft.design} baseColor={draft.color} pattern={draft.pattern} onChange={updateDesign} />}
        {draft.imageUri && <View style={{ gap: 10, marginBottom: 14 }}><Notice>{confirmed ? '종류와 앞·뒤 디자인을 확인했어요. 미확인으로 남긴 항목은 평가에 표시합니다.' : '앞·뒤 사진이 같은 옷인지, 바탕색·무늬색·로고의 크기와 위치가 맞는지 확인해주세요. 사진을 바꾸면 이전 AI 결과는 적용하지 않습니다.'}</Notice>{!confirmed && <Button secondary disabled={busy || extracting} onPress={() => { setDraft(g=>({...g,design:g.design?{...g.design,front:{...g.design.front,uncertain:[]},back:g.design.back?{...g.design.back,uncertain:[]}:null}:undefined}));setConfirmed(true); setError(''); }}>입력한 종류·디자인을 직접 확인했어요</Button>}</View>}
        <Notice>사진은 기기에 보관합니다. 배경 분리는 분석 서버에서 처리하고, AI 속성 분석을 선택할 때만 NVIDIA로 전송해요. 같은 원본 사진의 중복을 확인하며, 재촬영한 같은 옷은 구분하지 못해요. 점수는 확인한 속성으로 계산합니다.</Notice>
        {duplicate && <View style={{ gap: 10, marginTop: 14 }}><Notice>{`동일 사진이 '${duplicate.name}'에 등록되어 있어요. 기존 정보를 수정하면 점수 기준도 유지됩니다.`}</Notice><Button secondary disabled={busy} onPress={async () => {
          try { if (store.archived.some(g => g.id === duplicate.id)) await store.restore(duplicate.id); router.replace({ pathname: '/register', params: { id: duplicate.id } }); }
          catch (e) { setError((e as Error).message); }
        }}>기존 의류 확인·수정</Button></View>}
        {Boolean(error) && <Text accessibilityRole="alert" style={st.error}>{error}</Text>}
        <Button icon="checkmark" loading={busy} disabled={extracting || (!draft.demoAsset && !confirmed) || !store.ready || Boolean(store.error) || Boolean(duplicate)} onPress={submit} style={{ marginTop: 22 }}>{existing ? '수정 내용 저장' : '옷장에 등록하기'}</Button>
      </View>
    </View>
  </Shell>;
}
const st = StyleSheet.create({ layout: { flexDirection: 'row', gap: 24, alignItems: 'flex-start' }, photoPanel: { width: 310, backgroundColor: C.white, borderRadius: 20, padding: 22, borderWidth: 1, borderColor: C.line }, panelHeading: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }, panelTitle: { fontFamily: font, color: C.ink, fontSize: 17, fontWeight: '700' }, required: { fontSize: 10, color: C.blue, fontFamily: font }, photoEmpty: { height: 280, borderWidth: 1, borderStyle: 'dashed', borderColor: '#D0CFED', backgroundColor: '#F8F7FF', borderRadius: 14, alignItems: 'center', justifyContent: 'center', gap: 13 }, photoTitle: { fontFamily: font, fontSize: 13, fontWeight: '600', color: C.ink }, photoCopy: { fontFamily: font, fontSize: 11, color: C.muted }, photoButtons: { flexDirection: 'row', gap: 8, marginTop: 14 }, backRow: { flexDirection: 'row', alignItems: 'center', gap: 10, marginTop: 24 }, backImage: { height: 52, width: 45, backgroundColor: C.pale, borderRadius: 8 }, backPlaceholder: { height: 52, width: 45, backgroundColor: C.pale, borderRadius: 8, alignItems: 'center', justifyContent: 'center' }, backTitle: { fontFamily: font, color: C.ink, fontSize: 12, fontWeight: '700' }, backCopy: { fontFamily: font, color: C.muted, fontSize: 9, marginTop: 6 }, smallAction: { padding: 5 }, sampleButton: { marginTop: 22, paddingVertical: 12, borderTopWidth: 1, borderTopColor: C.line, alignItems: 'center' }, sampleCopy: { fontFamily: font, color: C.muted, fontSize: 10 }, formPanel: { flex: 1, width: '100%', padding: 26, borderRadius: 20, borderWidth: 1, borderColor: C.line, backgroundColor: C.white }, formIntro: { fontFamily: font, color: C.muted, fontSize: 11, lineHeight: 18, marginTop: 10, marginBottom: 8 }, field: { marginVertical: 13 }, label: { fontFamily: font, fontSize: 12, color: C.ink, fontWeight: '700', marginBottom: 10 }, input: { paddingHorizontal: 14, height: 46, borderRadius: 10, borderWidth: 1, borderColor: C.line, color: C.ink, fontFamily: font, fontSize: 13, backgroundColor: C.bg }, error: { color: '#B54848', marginTop: 16, fontFamily: font, lineHeight: 22, fontSize: 12 } });


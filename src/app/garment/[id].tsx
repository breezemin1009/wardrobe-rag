import { useState } from 'react';
import { router, useLocalSearchParams } from 'expo-router';
import { Image, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { Shell } from '@/components/Shell';
import { GarmentPhoto } from '@/components/GarmentPhoto';
import { Button, Empty, Loading, Sheet, Tag } from '@/components/ui';
import { C, font } from '@/components/theme';
import { useWardrobe } from '@/state/WardrobeProvider';
import { DesignSummary } from '@/components/DesignSummary';
import { catalog } from '@/domain/demo';
import { audienceLabels } from '@/domain/audience';

export default function GarmentScreen() {
  const { id } = useLocalSearchParams<{ id: string }>(), store = useWardrobe(), { width } = useWindowDimensions();
  const garment = store.garments.find(g => g.id === id) || catalog.find(g => g.id === id);
  const registered = store.garments.some(g => g.id === id);
  const [archiveOpen, setArchiveOpen] = useState(false), [error, setError] = useState(''), [busy, setBusy] = useState(false);
  async function action(fn: () => Promise<void>) { setBusy(true); try { await fn(); setError(''); } catch (e) { setError((e as Error).message); } finally { setBusy(false); } }
  if (!store.ready) return <Shell title="의류 정보 불러오는 중" back><Loading /></Shell>;
  if (!garment) return <Shell title="의류를 찾을 수 없어요" back><Empty title="옷장을 확인해주세요" description="제외한 의류는 옷장에서 복원할 수 있습니다." action={<Button onPress={() => router.replace('/')}>내 옷장으로</Button>} /></Shell>;
  return <Shell title="의류 상세" subtitle="등록한 사진과 속성을 확인하세요." back>
    <View style={[st.layout, width < 760 && { flexDirection: 'column' }]}>
      <View style={[st.photoPanel, width < 760 && { width: '100%' }]}><GarmentPhoto garment={garment} height={340} />{garment.backUri && <View style={st.back}><Text style={st.backLabel}>뒷면 사진</Text><Image source={{ uri: garment.backUri }} accessibilityLabel="등록한 뒷면" resizeMode="contain" style={{ height: 160, width: '100%' }} /></View>}</View>
      <View style={st.infoPanel}><View style={st.tags}><Tag blue>{garment.role === 'owned' ? '보유 의류' : '구매 후보'}</Tag>{garment.isExample && <Tag>예시 의류</Tag>}</View><Text style={st.name}>{garment.name}</Text>
        {[['의류 구분', audienceLabels[garment.audience || 'unknown']], ['종류', garment.category], ['색상', garment.color], ['무늬', garment.pattern], ['상황', garment.occasion]].map(([label, value]) => <View key={label} style={st.attribute}><Text style={st.attrLabel}>{label}</Text><Text style={st.attrValue}>{value}</Text></View>)}
        <DesignSummary design={garment.design} /><View style={st.actions}>
          {garment.role === 'candidate' && <Button icon="analytics-outline" onPress={() => router.push(`/analysis/${id}`)}>활용도 평가 보기</Button>}
          {!registered && <Button secondary loading={busy} icon="add" onPress={() => action(() => store.save({ ...garment, createdAt: Date.now() }))}>구매 후보에 추가</Button>}
          {registered && <Button secondary icon="create-outline" onPress={() => router.push({ pathname: '/register', params: { id } })}>정보 수정</Button>}
          {registered && garment.role === 'candidate' && <Button secondary loading={busy} icon="shirt-outline" onPress={() => action(() => store.save({ ...garment, role: 'owned' }))}>보유 옷으로 이동</Button>}
          {registered && <Button secondary icon="archive-outline" onPress={() => setArchiveOpen(true)}>옷장에서 제외</Button>}
        </View>{Boolean(error) && <Text style={st.error}>{error}</Text>}
      </View>
    </View>
    <Sheet title="옷장에서 제외할까요?" visible={archiveOpen} onClose={() => setArchiveOpen(false)}><Text style={st.dialogCopy}>목록에서 숨겨집니다. 필요하면 내 옷장에서 다시 복원할 수 있어요.</Text><Button loading={busy} onPress={() => action(async () => { await store.archive(id); setArchiveOpen(false); router.replace('/'); })}>제외하기</Button></Sheet>
  </Shell>;
}
const st = StyleSheet.create({ layout: { flexDirection: 'row', alignItems: 'flex-start', gap: 24 }, photoPanel: { width: 365, borderRadius: 20, padding: 22, backgroundColor: C.white, borderWidth: 1, borderColor: C.line }, infoPanel: { flex: 1, width: '100%', backgroundColor: C.white, borderRadius: 20, padding: 28, borderWidth: 1, borderColor: C.line }, tags: { flexDirection: 'row', gap: 8 }, name: { fontFamily: font, fontSize: 26, color: C.ink, fontWeight: '800', lineHeight: 36, marginTop: 22, marginBottom: 20 }, attribute: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 18, borderBottomWidth: 1, borderBottomColor: C.line }, attrLabel: { color: C.muted, fontFamily: font, fontSize: 13 }, attrValue: { color: C.ink, fontFamily: font, fontSize: 13, fontWeight: '600' }, actions: { marginTop: 28, gap: 10 }, back: { paddingTop: 22, marginTop: 22, borderTopWidth: 1, borderTopColor: C.line }, backLabel: { fontFamily: font, color: C.muted, fontSize: 12, marginBottom: 12 }, error: { color: '#B54848', marginTop: 15 }, dialogCopy: { fontFamily: font, color: C.muted, lineHeight: 23, fontSize: 13, marginBottom: 23 } });


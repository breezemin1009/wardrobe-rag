import { useState } from 'react';
import { router } from 'expo-router';
import { StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { Shell } from '@/components/Shell';
import { GarmentCard } from '@/components/GarmentCard';
import { Button, Empty } from '@/components/ui';
import { useWardrobe } from '@/state/WardrobeProvider';
import { catalog } from '@/domain/demo';

export default function SavedScreen() {
  const store = useWardrobe(), { width } = useWindowDimensions(), [error, setError] = useState('');
  const all = [...store.garments, ...catalog.filter(c => !store.garments.some(g => g.id === c.id))];
  const items = store.savedIds.map(id => all.find(g => g.id === id)).filter(g => g !== undefined);
  return <Shell title="저장한 구매 후보" subtitle="마음에 드는 후보를 모아두고, 한 벌씩 비교해보세요.">
    {Boolean(error) && <Text style={{ color: '#B54848', marginBottom: 20 }}>{error}</Text>}
    {items.length ? <View style={st.grid}>{items.map(g => <GarmentCard key={g.id} garment={g} width={width >= 1200 ? '31.7%' : '48%'} saved onSave={() => store.toggleSaved(g.id).catch(e => setError(e.message))} />)}</View> : <Empty title="아직 저장한 후보가 없어요" description="구매 추천에서 북마크를 눌러 마음에 드는 옷을 모아보세요." action={<Button icon="sparkles-outline" onPress={() => router.replace('/recommendations')}>구매 추천 보기</Button>} />}
  </Shell>;
}
const st = StyleSheet.create({ grid: { flexDirection: 'row', flexWrap: 'wrap', gap: 18, justifyContent: 'space-between' } });

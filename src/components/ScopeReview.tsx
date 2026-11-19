import { router } from 'expo-router';
import { Text, View } from 'react-native';
import type { Garment } from '@/domain/model';
import { scopeReview } from '@/domain/audience';
import { Button, Notice } from './ui';
import { C, font } from './theme';

export function ScopeReview({ garments }: { garments: Garment[] }) {
  const review = scopeReview(garments);
  if (!review.length) return null;
  return <View style={{ gap: 12, marginBottom: 20 }}>
    <Notice>{`의류 구분 확인이 필요한 옷이 ${review.length}벌 있어요. 남성복·공용 여부를 확인하거나 보관함으로 옮기면 현재 옷장으로 평가할 수 있어요.`}</Notice>
    {review.slice(0, 3).map(g => <View key={g.id} style={{ backgroundColor: C.white, padding: 16, borderRadius: 14, gap: 12 }}>
      <Text style={{ fontFamily: font, color: C.ink, fontSize: 14 }}>{g.name}</Text>
      <Button secondary onPress={() => router.push({ pathname: '/register', params: { id: g.id } })}>의류 구분 확인</Button>
    </View>)}
  </View>;
}

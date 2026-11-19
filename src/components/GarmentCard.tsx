import { router } from 'expo-router';
import { Pressable, StyleSheet, Text, View } from 'react-native';
import type { Garment } from '@/domain/model';
import { GarmentPhoto } from './GarmentPhoto';
import { C, font } from './theme';
import { Icon, Tag } from './ui';
import { audienceLabels } from '@/domain/audience';

export function GarmentCard({ garment, width, saved, onSave }: { garment: Garment; width: `${number}%` | number; saved?: boolean; onSave?: () => void }) {
  return <View style={[st.card, { width }]}><Pressable accessibilityRole="button" accessibilityLabel={garment.name} onPress={() => router.push(`/garment/${garment.id}`)}>
    <GarmentPhoto garment={garment} /><View style={st.info}><Text style={st.name} numberOfLines={1}>{garment.name}</Text><View style={st.tags}><Tag>{audienceLabels[garment.audience || 'unknown']}</Tag><Tag>{garment.category}</Tag><Tag>{garment.color}</Tag><Tag>{garment.pattern}</Tag></View></View>
  </Pressable>{onSave && <Pressable accessibilityRole="button" accessibilityLabel={`${garment.name} ${saved ? '저장 해제' : '저장'}`} accessibilityState={{ selected: saved }} onPress={onSave} style={st.bookmark}><Icon name={saved ? 'bookmark' : 'bookmark-outline'} size={17} color={saved ? C.blue : C.muted} /></Pressable>}</View>;
}
const st = StyleSheet.create({ card: { backgroundColor: C.white, borderWidth: 1, borderColor: C.line, borderRadius: 3, padding: 8 }, info: { paddingHorizontal: 4, paddingTop: 13, paddingBottom: 7 }, name: { color: C.ink, fontFamily: font, fontSize: 14, fontWeight: '700' }, tags: { marginTop: 9, flexDirection: 'row', gap: 4, flexWrap: 'wrap' }, bookmark: { position: 'absolute', right: 18, top: 17, padding: 6, borderRadius: 9, backgroundColor: '#FFFFFFE8' } });

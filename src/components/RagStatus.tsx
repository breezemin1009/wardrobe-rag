import { Pressable, StyleSheet, Text, View } from 'react-native';
import { useRag } from '@/state/RagProvider';
import { C, font } from './theme';
import { Icon } from './ui';
export function RagStatus() {
  const { health, checking, refresh } = useRag();
  return <View style={st.row}><View style={[st.dot, { backgroundColor: health ? C.green : C.muted }]} /><View style={{ flex: 1 }}><Text style={st.title}>{checking ? '근거 자료 연결 확인 중' : health ? '근거 검색 연결됨' : '분석 서버 연결 필요'}</Text><Text style={st.copy}>{health ? '공개 상품·코디 검색과 옷장 비교를 사용할 수 있어요.' : '로컬 서버를 실행하면 검색과 평가를 이용할 수 있어요.'}</Text></View><Pressable accessibilityRole="button" accessibilityLabel="분석 서버 연결 다시 확인" onPress={refresh} disabled={checking} style={{ padding: 8 }}><Icon name="refresh-outline" size={19} color={C.blue} /></Pressable></View>;
}
const st = StyleSheet.create({ row: { backgroundColor: C.white, borderWidth: 1, borderColor: C.line, padding: 16, borderRadius: 14, flexDirection: 'row', gap: 11, alignItems: 'center', marginBottom: 20 }, dot: { width: 7, height: 7, borderRadius: 5 }, title: { fontFamily: font, fontWeight: '700', fontSize: 12, color: C.ink }, copy: { fontFamily: font, fontSize: 11, lineHeight: 18, color: C.muted, marginTop: 4 } });

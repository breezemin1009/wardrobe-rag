import { StyleSheet, Text, View } from 'react-native';
import type { Assessment } from '@/services/rag';
import { C, font } from './theme';

export function PublicCaseDetails({ result }: { result: Assessment }) {
  return <View style={s.panel}>
    <Text style={s.kicker}>검색에서 비교까지</Text>
    <Text style={s.title}>내 옷을 기준으로, 공개 코디를 참고해요.</Text>
    <Text style={s.copy}>{result.message}</Text>
    <View style={s.steps}>{['확인한 의류 정보', '유사 의류·코디 검색', '기존 옷과 후보 비교'].map((text, i) => <View key={text} style={s.step}><Text style={s.number}>0{i + 1}</Text><Text style={s.label}>{text}</Text></View>)}</View>
    <Text style={s.copy}>조합 활용도는 내 상의와의 전체 지원, 추가 코디 기여는 기존 하의의 지원을 제외하고 새로 더하는 부분이에요.</Text>
    <Text style={s.note}>유사 상품은 같은 옷으로 확정하지 않아요. 현재 점수에는 종류·색상·무늬를 사용하며, 착용 상황과 로고의 세부 위치는 반영하지 않아요.</Text>
  </View>;
}
const s = StyleSheet.create({
  panel: { backgroundColor: C.white, padding: 24, borderWidth: 1, borderColor: C.line, marginTop: 18 },
  kicker: { color: C.blue, fontFamily: font, fontSize: 11, fontWeight: '700', letterSpacing: 1 },
  title: { color: C.ink, fontFamily: font, fontSize: 20, lineHeight: 29, marginTop: 12, fontWeight: '700' },
  copy: { color: C.ink, fontFamily: font, fontSize: 13, lineHeight: 23, marginTop: 12 },
  note: { color: C.muted, fontFamily: font, fontSize: 12, lineHeight: 21, marginTop: 14 },
  steps: { flexDirection: 'row', gap: 12, marginVertical: 20, flexWrap: 'wrap' },
  step: { flex: 1, minWidth: 100, borderTopWidth: 1, borderColor: C.line, paddingTop: 10 },
  number: { fontFamily: font, color: C.blue, fontSize: 11 }, label: { fontFamily: font, color: C.ink, fontSize: 12, lineHeight: 21, marginTop: 6 },
});

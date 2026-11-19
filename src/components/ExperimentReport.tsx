import { useEffect, useState } from 'react';
import { ActivityIndicator, StyleSheet, Text, View } from 'react-native';
import { Button, Notice } from './ui';
import { C, font } from './theme';
import { api } from '@/services/rag';
import { useRag } from '@/state/RagProvider';

type Report = { version: string; scope: string; corpusSha256: string; checkedAt: string;
  data: { acceptedOutfits: number; apparelItems: number; observedPairs: number };
  catalog: { accepted: number; audiences: { mens: number; unisex: number } };
  consistency: { cases: number; scoredCases: number; repeatInvariantPassed: number; renameOrderInvariantPassed: number; exactOwnedComparable: number; exactOwnedZeroGain: number; positiveGainCases: number };
  retrieval: { mode: string; returned: number; conditionMatches: number; inScope: boolean }[];
};
const modes = { keyword: '키워드 검색', vector: '의미 검색', hybrid: '결합 검색', conditioned: '속성 조건 결합' };
export function ExperimentReport() {
  const connection = useRag(), [attempt, setAttempt] = useState(0);
  const key = connection.revision + ':' + attempt;
  const [state, setState] = useState<{ key: string; report: Report | null; error: string }>({ key: '', report: null, error: '' });
  useEffect(() => {
    if (!connection.health) return;
    const controller = new AbortController();
    api<Report>('/experiments', undefined, controller.signal)
      .then(report => { if (!controller.signal.aborted) setState({ key, report, error: '' }); })
      .catch(e => { if (!controller.signal.aborted) setState({ key, report: null, error: e.message }); });
    return () => controller.abort();
  }, [key, connection.health]);
  const current = state.key === key, report = current ? state.report : null;
  if (!current || !connection.health) return <View style={st.loading}><ActivityIndicator color={C.blue} /><Text style={st.copy}>현재 자료의 검사 결과를 확인하고 있어요.</Text></View>;
  if (!report) return <View style={st.panel}><Text style={st.copy}>{state.error}</Text><Button secondary onPress={() => setAttempt(v => v + 1)}>다시 확인</Button></View>;
  const c = report.consistency;
  return <View style={{ gap: 22 }}>
    <View style={st.panel}><Text style={st.title}>남성복 데이터와 추천 경로 검증</Text><Text style={st.copy}>현재 앱과 같은 검색 자료·계산 버전으로 실행한 결과입니다.</Text><View style={st.stats}>{[['사진 대조 상품', report.catalog.accepted], ['실제 코디', report.data.acceptedOutfits], ['상·하의 연결', report.data.observedPairs]].map(([label, value]) => <View key={String(label)}><Text style={st.number}>{Number(value).toLocaleString()}</Text><Text style={st.copy}>{label}</Text></View>)}</View><Text style={st.copy}>상품 분류: 남성복 {report.catalog.audiences.mens.toLocaleString()}개 · 공용 {report.catalog.audiences.unisex}개</Text></View>
    <View style={st.panel}><Text style={st.title}>같은 옷장은 같은 결과로</Text><Text style={st.copy}>기본 7색·무지 의류로 보유 상의, 구매 하의, 기존 하의 유무를 바꾼 {c.cases}가지 조건을 검사했습니다.</Text>{[['중복 의류를 추가해도 동일', c.repeatInvariantPassed, c.cases], ['이름·등록 순서를 바꿔도 동일', c.renameOrderInvariantPassed, c.cases], ['이미 보유한 후보의 추가 기여는 0', c.exactOwnedZeroGain, c.exactOwnedComparable]].map(([label, passed, total]) => <View key={String(label)} style={st.row}><Text style={st.copy}>{label}</Text><Text style={st.result}>{passed} / {total}</Text></View>)}<Text style={st.copy}>근거가 확보돼 점수를 계산한 조건 {c.scoredCases}개 · 기존 옷보다 추가 지원이 확인된 조건 {c.positiveGainCases}개</Text></View>
    <View style={st.panel}><Text style={st.title}>검색한 근거가 입력 조건에 맞는가?</Text><Text style={st.copy}>검색어 6개에 대해 최대 5개씩 검색했습니다. 아래 비율은 검색 결과 중 요청한 색상·무늬·상황이 맞는 근거의 비율입니다.</Text>{Object.entries(modes).map(([mode, label]) => {
      const rows = report.retrieval.filter(r => r.mode === mode), total = rows.reduce((s, r) => s + r.returned, 0), matched = rows.reduce((s, r) => s + r.conditionMatches, 0);
      return <View style={st.row} key={mode}><Text style={st.copy}>{label}</Text><Text style={st.result}>{total ? Math.round(matched / total * 100) : 0}% · {matched}/{total}</Text></View>;
    })}<Notice>색상·무늬·상황을 명시한 검색은 먼저 조건이 맞는 근거로 범위를 좁힙니다. 검색 방식에 따라 그 안의 순서를 정합니다.</Notice></View>
    <View style={st.panel}><Text style={st.title}>검사 결과 읽는 방법</Text><Text style={st.copy}>위 수치는 자료 선별, 검색 조건, 계산의 일관성을 확인한 결과입니다. 조합 점수는 코디 근거의 지원 수준이고, 추가 기여는 같은 옷장에서 기존 대안보다 늘어난 지원 수준입니다.</Text><Text style={st.copy}>확인 시각 {new Date(report.checkedAt).toLocaleString('ko-KR')} · {report.version}</Text></View>
  </View>;
}
const st = StyleSheet.create({
  panel: { padding: 24, borderWidth: 1, borderColor: C.line, borderRadius: 18, backgroundColor: C.white, gap: 14 },
  title: { fontFamily: font, color: C.ink, fontSize: 18, fontWeight: '700', lineHeight: 27 },
  copy: { fontFamily: font, color: C.muted, fontSize: 13, lineHeight: 23, flexShrink: 1 },
  stats: { flexDirection: 'row', flexWrap: 'wrap', gap: 30, paddingVertical: 10 }, number: { fontFamily: font, fontWeight: '800', fontSize: 30, color: C.blue },
  row: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', gap: 15, borderTopWidth: 1, borderTopColor: C.line, paddingTop: 12 },
  result: { fontFamily: font, color: C.blue, fontSize: 14, fontWeight: '700' }, loading: { alignItems: 'center', padding: 35, gap: 18 },
});

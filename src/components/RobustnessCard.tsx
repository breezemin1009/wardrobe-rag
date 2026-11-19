import { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { api, type Assessment, type RobustnessReport } from '@/services/rag';
import { Button, Notice, Tag } from './ui';
import { C, font } from './theme';

export function RobustnessCard({result,onRefresh}:{result:Assessment;onRefresh:()=>void}) {
  const [report,setReport]=useState<RobustnessReport|null>(null);
  const [busy,setBusy]=useState(false),[error,setError]=useState('');
  const [sourcesOpen,setSourcesOpen]=useState(false),[parametersOpen,setParametersOpen]=useState(false);
  const active=useRef<AbortController|null>(null);
  useEffect(()=>()=>active.current?.abort(),[]);
  async function inspect() {
    if (active.current) return;
    const controller=new AbortController();active.current=controller;setBusy(true);setError('');
    try {
      const next=await api<RobustnessReport>(`/runs/${result.runId}/robustness`,undefined,controller.signal);
      if (!controller.signal.aborted) setReport(next);
    } catch (e) {if (!controller.signal.aborted) setError((e as Error).message);}
    finally {if (!controller.signal.aborted) {setBusy(false);active.current=null;}}
  }
  return <View style={st.card}>
    <Text style={st.title}>근거에 따른 점수 변화</Text>
    <Text style={st.copy}>옷장과 상황을 고정하고, 출처를 하나씩 제외했을 때 평가가 얼마나 달라지는지 확인해요.</Text>
    {!report && <Button secondary icon="analytics-outline" loading={busy} onPress={inspect}>근거 의존성 확인</Button>}
    {Boolean(error) && <><Text accessibilityRole="alert" style={st.error}>{error}</Text><Button secondary onPress={onRefresh}>현재 옷장으로 다시 평가</Button></>}
    {report && <>
      <View style={st.stats}>{[
        ['같은 범위로 점수 비교',`${report.summary.comparableRemovals}/${report.summary.familiesTested}조건`],
        ['비교 범위가 달라짐',`${report.summary.scopeChangedRemovals}조건`],
        [report.base.score===null?'제외 후에도 점수 보류':'제외 후 점수 보류',`${report.summary.withheldRemovals}조건`],
      ].map(([label,value])=><View key={label} style={st.stat}><Text style={st.small}>{label}</Text><Text style={st.value}>{value}</Text></View>)}</View>
      <Text style={st.copy}>{report.summary.maxAbsoluteComparableDelta===null
        ? '같은 범위에서 계산된 종합점수 쌍이 없어 점수 차이를 제시하지 않아요.'
        : `같은 범위에서 비교할 수 있는 조건의 최대 차이는 ${report.summary.maxAbsoluteComparableDelta.toFixed(1)}점이에요.`}</Text>
      <Text style={st.small}>출처를 제외해 비교 범위가 바뀌거나 점수가 보류되면 점수 차이에 포함하지 않아요. 차이가 작아도 미적 정확도가 검증된 것은 아니에요.</Text>
      <Button secondary onPress={()=>setSourcesOpen(v=>!v)}>{sourcesOpen?'출처별 변화 접기':'출처별 변화 보기'}</Button>
      {sourcesOpen && <View style={st.list}>{report.sourceRemoval.map(row=><View key={row.removedFamily} style={st.row}>
        <Text style={st.source}>{row.removedFamily} 제외</Text>
        <Tag blue>{row.score===null?'점수 보류':!row.sameComputedScope?'비교 범위 변경':`${row.score.toFixed(1)}점`}</Tag>
        <Text style={st.small}>기존 의류까지 비교 {row.gainCoverage}%{row.scoreDelta!==null?` · 기준 대비 ${row.scoreDelta>0?'+':''}${row.scoreDelta.toFixed(1)}점`:''}</Text>
      </View>)}</View>}
      <Button secondary onPress={()=>setParametersOpen(v=>!v)}>{parametersOpen?'계산 기준 비교 접기':'계산 기준 9조건 보기'}</Button>
      {parametersOpen && <View style={st.list}>
        <Text style={st.copy}>조합 활용도의 비중과 필요한 근거 범위를 바꿔본 개발 비교예요. 현재 점수의 설정을 변경하지 않아요.</Text>
        {report.parameters.map(row=><Text key={`${row.alpha}:${row.minimumCoverage}`} style={st.copy}>
          조합 비중 {Math.round(row.alpha*100)}% · 근거 기준 {Math.round(row.minimumCoverage*100)}% → {row.score===null?'보류':`${row.score.toFixed(1)}점`}{!row.sameComputedScope?' · 비교 범위 변경':''}
        </Text>)}
      </View>}
      <Notice>현재 개발 지수의 민감도 검사예요. 어울림 확률·신뢰구간·실제 구매 만족도를 뜻하지 않아요.</Notice>
    </>}
  </View>;
}
const st=StyleSheet.create({
  card:{marginTop:24,padding:22,borderRadius:18,borderWidth:1,borderColor:C.line,backgroundColor:C.white,gap:14},
  title:{fontFamily:font,color:C.ink,fontSize:16,fontWeight:'700'},
  copy:{fontFamily:font,color:C.muted,fontSize:12,lineHeight:22},
  small:{fontFamily:font,color:C.muted,fontSize:11,lineHeight:20},
  stats:{flexDirection:'row',flexWrap:'wrap',gap:12},
  stat:{flexGrow:1,flexBasis:170,borderRadius:12,backgroundColor:C.pale,padding:14,gap:8},
  value:{fontFamily:font,color:C.ink,fontSize:19,fontWeight:'700'},
  list:{gap:12},row:{padding:14,borderRadius:12,borderWidth:1,borderColor:C.line,gap:8,alignItems:'flex-start'},
  source:{fontFamily:font,color:C.ink,fontSize:12,fontWeight:'700'},
  error:{fontFamily:font,color:'#B54848',fontSize:12,lineHeight:21},
});

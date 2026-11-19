import { useState } from 'react';
import { router } from 'expo-router';
import { Text,View } from 'react-native';
import type { Garment } from '@/domain/model';
import type { Assessment } from '@/services/rag';
import { Button } from './ui';
import { C,font } from './theme';

const labels:Record<string,string>={design:'로고·무늬와 앞뒤 디자인',back:'뒷면의 디자인',garmentType:'티셔츠·청바지 등 세부 종류',baseColor:'바탕색',pattern:'무늬 종류',patternColors:'무늬색',patternScale:'무늬 크기·줄 폭',contrast:'무늬 대비',accents:'로고·문자·그래픽의 크기와 색'};
function fieldLabel(field:string) {
  const [side,key]=field.split('.');
  return key ? `${side==='back'?'뒷면':'앞면'} ${labels[key] || '디자인'}` : labels[field] || '의류 정보';
}
export function InputReview({result,candidate,garments}:{result:Assessment;candidate:Garment;garments:Garment[]}) {
  const [expanded,setExpanded]=useState(false);
  const reviews=(result.inputReviews || []).map(r=>({...r,garment:garments.find(g=>g.id===r.garmentId) || (candidate.id===r.garmentId?candidate:undefined)}))
    .filter(r=>r.garment).sort((a,b)=>Number(b.garmentId===candidate.id)-Number(a.garmentId===candidate.id) || a.garmentId.localeCompare(b.garmentId));
  if (!reviews.length) return null;
  const text={fontFamily:font,fontSize:12,lineHeight:21,color:C.muted} as const;
  return <View style={{marginTop:18,padding:20,borderWidth:1,borderColor:C.line,borderRadius:16,backgroundColor:C.white,gap:14}}>
    <Text style={{...text,fontSize:15,fontWeight:'700',color:C.ink}}>추가로 확인할 의류 정보</Text>
    <Text style={text}>아래 옷의 정보를 확인하면 같은 옷장으로 다시 평가할 수 있어요. 확인 후에도 조합 근거가 부족한 경우는 따로 안내해요.</Text>
    {(expanded?reviews:reviews.slice(0,3)).map(r=><View key={r.garmentId} style={{padding:14,borderRadius:12,backgroundColor:C.pale,gap:10}}>
      <Text style={{...text,fontWeight:'700',color:C.ink}}>{r.garment!.name}</Text><Text style={text}>{[...new Set(r.fields.map(fieldLabel))].join(' · ')}</Text>
      <Button secondary icon="create-outline" accessibilityLabel={`${r.garment!.name}의 누락 정보 확인`} onPress={()=>router.push({pathname:garments.some(g=>g.id===r.garmentId)?'/register':'/garment/[id]',params:{id:r.garmentId}})}>의류 정보 확인·수정</Button>
    </View>)}
    {reviews.length>3 && <Button secondary onPress={()=>setExpanded(v=>!v)}>{expanded?'간단히 보기':`전체 ${reviews.length}벌 확인하기`}</Button>}
  </View>;
}

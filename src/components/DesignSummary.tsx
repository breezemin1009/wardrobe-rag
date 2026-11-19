import { Text,View } from 'react-native';
import { backLabels,garmentTypeLabels,viewSummary,type DesignProfile } from '@/domain/design';
import { C,font } from './theme';
export function DesignSummary({design}:{design?:DesignProfile}) {
  const text={fontFamily:font,fontSize:11,lineHeight:20,color:C.muted} as const;
  if (!design) return <Text style={{...text,marginTop:15}}>기본 색상·무늬만 등록되어 있어요. 정보 수정에서 로고·앞뒤 디자인을 기록할 수 있어요. 현재 코디 점수에는 종류·색상·무늬를 사용합니다.</Text>;
  return <View style={{marginTop:15,padding:14,backgroundColor:C.pale,borderRadius:12,gap:8}}><Text style={{...text,fontWeight:'700',color:C.ink}}>확인한 디자인 · {garmentTypeLabels[design.garmentType || 'unknown']}</Text><Text style={text}>앞면 · {viewSummary(design.front)}</Text><Text style={text}>뒷면 · {design.back ? viewSummary(design.back) : backLabels[design.backStatus]}</Text></View>;
}

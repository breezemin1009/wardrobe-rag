import { useState } from 'react';
import { Pressable, Text, View } from 'react-native';
import { Choices } from './Choices';
import { Button, Notice } from './ui';
import { C, font } from './theme';
import { accentKinds,accentSizes,placements,designColors,designPatterns,scales,contrasts,kindLabels,sizeLabels,placementLabels,scaleLabels,contrastLabels,backLabels,garmentTypes,garmentTypeLabels,emptyView,viewSummary,maxDesignColors,maxDesignAccents,type DesignProfile,type ViewDesign,type DesignColour } from '@/domain/design';

const caption = { fontFamily:font,fontSize:11,color:C.muted,lineHeight:19 } as const;
function Palette({ value, onChange,exclude }: { value: DesignColour[]; onChange: (value: DesignColour[]) => void;exclude?:DesignColour }) {
  return <View style={{gap:8}}><View style={{ flexDirection:'row',flexWrap:'wrap',gap:6 }}>{designColors.filter(c=>c!==exclude).map(c => {
    const limited=value.length>=maxDesignColors && !value.includes(c);
    return <Pressable key={c} disabled={limited} accessibilityRole="checkbox" accessibilityLabel={c} accessibilityState={{ checked:value.includes(c),disabled:limited }} aria-checked={value.includes(c)} aria-disabled={limited} onPress={() => onChange(value.includes(c) ? value.filter(v => v !== c) : [...value,c])} style={{ padding:8,borderRadius:8,opacity:limited?.4:1,backgroundColor:value.includes(c)?C.light:C.pale,borderWidth:1,borderColor:value.includes(c)?C.blue:C.line }}><Text style={caption}>{c}</Text></Pressable>;
  })}</View><Text style={caption}>선택 {value.length}/{maxDesignColors} · 색을 구분하기 어려우면 기타로 표시해주세요.</Text></View>;
}
function ViewEditor({ value, onChange }: { value: ViewDesign; onChange:(v:ViewDesign)=>void }) {
  const change = (patch:Partial<ViewDesign>) => onChange({ ...value,...patch,uncertain:value.uncertain.filter(k=>!(k in patch)) });
  return <View style={{ gap:12 }}>
    <Text style={caption}>옷의 바탕색</Text><Choices options={designColors} value={value.baseColor} onChange={baseColor => change({baseColor,patternColors:value.patternColors.filter(c=>c!==baseColor)})} />
    <Text style={caption}>천 전체의 무늬 · 한 곳의 로고는 아래에 추가</Text><Choices options={designPatterns} value={value.pattern} onChange={pattern => change({pattern,patternColors:[],patternScale:pattern==='무지'?'none':'unknown'})} />
    {value.pattern !== '무지' && <><Text style={caption}>바탕색을 제외한 무늬색 · 최대 {maxDesignColors}개</Text><Palette value={value.patternColors} exclude={value.baseColor} onChange={patternColors => change({patternColors})} /><Text style={caption}>줄 폭·무늬 크기</Text><Choices options={scales.filter(k=>k!=='none').map(k=>scaleLabels[k])} value={scaleLabels[value.patternScale]} onChange={v=>change({patternScale:scales.find(k=>scaleLabels[k]===v)!})} /><Text style={caption}>바탕과 무늬의 대비</Text><Choices options={contrasts.map(k=>contrastLabels[k])} value={contrastLabels[value.contrast]} onChange={v=>change({contrast:contrasts.find(k=>contrastLabels[k]===v)!})} /></>}
    {value.accents.map((a,i)=><View key={i} style={{ padding:12,gap:10,borderRadius:12,backgroundColor:C.pale }}>
      <Text style={{...caption,fontWeight:'700'}}>로고·장식 {i+1}</Text>
      <Choices options={accentKinds.map(k=>kindLabels[k])} value={kindLabels[a.kind]} onChange={v=>change({accents:value.accents.map((a,j)=>j===i?{...a,kind:accentKinds.find(k=>kindLabels[k]===v)!}:a)})} />
      <Text style={caption}>보이는 옷 면적에서의 크기 · 작음 약 5% 미만, 보통 약 5~20%, 큼 약 20% 초과</Text>
      <Choices options={accentSizes.map(k=>sizeLabels[k])} value={sizeLabels[a.size]} onChange={v=>change({accents:value.accents.map((a,j)=>j===i?{...a,size:accentSizes.find(k=>sizeLabels[k]===v)!}:a)})} />
      <Choices options={placements.map(k=>placementLabels[k])} value={placementLabels[a.placement]} onChange={v=>change({accents:value.accents.map((a,j)=>j===i?{...a,placement:placements.find(k=>placementLabels[k]===v)!}:a)})} />
      <Text style={caption}>로고·장식 색 · 최대 {maxDesignColors}개</Text><Palette value={a.colors} onChange={colors=>change({accents:value.accents.map((a,j)=>j===i?{...a,colors}:a)})} />
      <Button secondary onPress={()=>change({accents:value.accents.filter((_,j)=>j!==i)})}>이 장식 삭제</Button>
    </View>)}
    {value.accents.length < maxDesignAccents && <Button secondary icon="add" onPress={()=>change({accents:[...value.accents,{kind:'logo',size:'unknown',placement:'chest',colors:[]}]})}>로고·문자·그래픽 추가</Button>}
    {!value.accents.length && <><Text style={caption}>현재 장식 없음으로 입력되어 있어요. 로고나 문자가 있으면 추가해주세요.</Text>{value.uncertain.includes('accents') && <Button secondary onPress={()=>change({accents:[]})}>로고·장식 없음을 확인</Button>}</>}
  </View>;
}
export function DesignEditor({ value, baseColor, pattern, onChange }: { value?:DesignProfile; baseColor:DesignColour; pattern:ViewDesign['pattern']; onChange:(v:DesignProfile)=>void }) {
  const [open,setOpen]=useState(false),[side,setSide]=useState<'front'|'back'>('front');
  const draft=value || {version:1 as const,front:emptyView(baseColor,pattern),back:null,backStatus:'unknown' as const};
  return <View style={{ gap:12,marginVertical:18,padding:16,borderWidth:1,borderColor:C.line,borderRadius:14 }}>
    <Text style={{...caption,fontSize:14,fontWeight:'700',color:C.ink}}>앞·뒤 디자인 확인</Text>
    {value ? <><Text style={caption}>앞면 · {viewSummary(value.front)}</Text><Text style={caption}>뒷면 · {value.back ? viewSummary(value.back) : backLabels[value.backStatus]}</Text></> : <Text style={caption}>바탕색과 무늬색, 로고의 크기·위치를 추가하면 서로 다른 디자인을 구분할 수 있어요.</Text>}
    <Button secondary onPress={()=>setOpen(v=>!v)}>{open?'디자인 편집 접기':'디자인 직접 확인·수정'}</Button>
    {open && <>
      <Notice>AI가 읽은 값과 기본 입력값을 사진과 비교해주세요. 뒷면을 보지 못했다면 미확인으로 남길 수 있어요.</Notice>
      <Text style={caption}>세부 종류 · 파란색이라고 청바지로 판단하지 않아요</Text><Choices options={garmentTypes.filter(k=>k!=='skirt').map(k=>garmentTypeLabels[k])} value={garmentTypeLabels[draft.garmentType || 'unknown']} onChange={v=>onChange({...draft,garmentType:garmentTypes.find(k=>garmentTypeLabels[k]===v)!})} />
      <Choices options={['앞면','뒷면']} value={side==='front'?'앞면':'뒷면'} onChange={v=>setSide(v==='앞면'?'front':'back')} />
      {side==='front' ? <ViewEditor value={draft.front} onChange={front=>onChange({...draft,front})} /> : <>
        <Text style={caption}>뒷면의 확인 상태</Text><Choices options={Object.values(backLabels)} value={backLabels[draft.backStatus]} onChange={v=>{const backStatus=(Object.keys(backLabels) as DesignProfile['backStatus'][]).find(k=>backLabels[k]===v)!;onChange({...draft,backStatus,back:backStatus==='observed'?(draft.back||emptyView(draft.front.baseColor)):null});}} />
        {draft.backStatus==='observed' && <ViewEditor value={draft.back!} onChange={back=>onChange({...draft,back})} />}
      </>}
    </>}
  </View>;
}

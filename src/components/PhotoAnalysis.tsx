import { useEffect, useRef, useState } from 'react';
import { Text, View } from 'react-native';
import { analyseDesign, type DesignPhotoProposal } from '@/services/photo-analysis';
import type { Garment } from '@/domain/model';
import { viewSummary } from '@/domain/design';
import { Button, Notice } from './ui';
import { C, font } from './theme';

export function PhotoAnalysis({ uri, backUri, disabled, processed = false, onApply }: { uri: string; backUri?:string; disabled: boolean; processed?: boolean; onApply: (value: Pick<Garment, 'category' | 'color' | 'pattern' | 'design'>) => void }) {
  const [result, setResult] = useState<DesignPhotoProposal | null>(null), [busy, setBusy] = useState(false), [error, setError] = useState(''), [applied, setApplied] = useState(false);
  const active = useRef<AbortController | null>(null);
  useEffect(() => () => active.current?.abort(), []);
  async function analyse() {
    if (active.current) return;
    const controller = new AbortController(); active.current = controller;
    setBusy(true); setError(''); setResult(null); setApplied(false);
    try { const next = await analyseDesign(uri, backUri, controller.signal); if (!controller.signal.aborted) setResult(next); }
    catch (e) { if (!controller.signal.aborted) setError((e as Error).message); }
    finally { if (!controller.signal.aborted) { active.current = null; setBusy(false); } }
  }
  return <View style={{ marginTop: 22, gap: 12 }}>
    <Text style={{ fontFamily: font, fontSize: 13, fontWeight: '700', color: C.ink }}>사진 속성 확인 보조</Text>
    <Text style={{ fontFamily: font, fontSize: 11, lineHeight: 19, color: C.muted }}>{processed ? '배경을 분리한 앞면 사진' : '선택한 앞면 사진'}{backUri ? '과 뒷면 사진을' : '을'} NVIDIA에 보내 바탕색·무늬색·로고의 크기와 위치를 각각 읽습니다. 인물·민감한 정보가 없는 옷 단독 사진을 사용해주세요. 사진은 서버 기록에 저장하지 않습니다.</Text>
    <Button secondary icon="sparkles-outline" disabled={disabled} loading={busy} onPress={analyse}>사진을 보내 AI 분석</Button>
    {result && <><Notice>{`${result.proposal.category}\n앞면 · ${viewSummary(result.proposal.front)}\n뒷면 · ${result.proposal.back ? viewSummary(result.proposal.back) : '미확인'}\n${result.proposal.uncertain.length ? '확인이 필요한 항목: '+result.proposal.uncertain.join(', ') : 'AI 추정값입니다. 사진과 직접 비교해주세요.'}${result.proposal.pairCheck === 'different' ? '\n앞뒤가 다른 옷으로 보입니다. 같은 옷 사진으로 바꾸거나 직접 확인·입력해주세요.' : result.proposal.pairCheck === 'uncertain' ? '\n앞뒤가 같은 옷인지 직접 확인해주세요.' : ''}`}</Notice>
      <Button secondary disabled={disabled || applied || result.proposal.pairCheck === 'different'} onPress={() => { onApply({ ...result.legacy,design:{version:1,garmentType:result.proposal.garmentType,front:result.proposal.front,back:result.proposal.back,backStatus:result.proposal.back?'observed':'unknown'} }); setApplied(true); }}>{applied ? '의류 정보에 반영했어요' : '앞·뒤 디자인 확인·반영'}</Button></>}
    {Boolean(error) && <Text accessibilityRole="alert" style={{ color: '#B54848', fontFamily: font, fontSize: 11, lineHeight: 20 }}>{error}</Text>}
  </View>;
}

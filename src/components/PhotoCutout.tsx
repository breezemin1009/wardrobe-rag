import { useEffect, useRef, useState } from 'react';
import { Image, Text, View } from 'react-native';
import { extractPhoto, type CutoutResult } from '@/services/photo-analysis';
import { Button, Notice } from './ui';
import { C, font } from './theme';

export function PhotoCutout({ uri, disabled, applied, onApply, onRestore, onBusy }: { uri: string; disabled: boolean; applied: boolean; onApply: (value: string) => void; onRestore: () => void; onBusy: (value: boolean) => void }) {
  const [result, setResult] = useState<CutoutResult | null>(null), [busy, setBusy] = useState(false), [error, setError] = useState('');
  const active = useRef<AbortController | null>(null);
  useEffect(() => () => { active.current?.abort(); onBusy(false); }, [onBusy]);
  async function extract() {
    if (active.current) return;
    const controller = new AbortController(); active.current = controller;
    setBusy(true); onBusy(true); setError(''); setResult(null);
    try { const next = await extractPhoto(uri, controller.signal); if (!controller.signal.aborted) setResult(next); }
    catch (e) { if (!controller.signal.aborted) setError((e as Error).message); }
    finally { if (!controller.signal.aborted) { active.current = null; setBusy(false); onBusy(false); } }
  }
  return <View style={{ gap: 12, marginTop: 20 }}>
    <Text style={{ fontFamily: font, fontSize: 13, fontWeight: '700', color: C.ink }}>사진 배경 분리</Text>
    <Text style={{ fontFamily: font, fontSize: 11, lineHeight: 19, color: C.muted }}>원본을 보관하면서 사진 배경을 분석 서버에서 분리해요. 옷 단독 사진은 의류 윤곽을, 모델 착용 사진은 인물과 의류를 함께 분리합니다. 흰 옷은 어두운 배경, 검은 옷은 밝은 배경에서 경계가 더 잘 보여요.</Text>
    <Button secondary icon="cut-outline" disabled={disabled} loading={busy} onPress={extract}>배경 분리 미리보기</Button>
    {result && <>
      {result.cutout && <View style={{ flexDirection: 'row', gap: 8 }}>
        <View style={{ flex: 1, gap: 6 }}><Text style={{ color: C.muted, fontFamily: font, fontSize: 10 }}>원본</Text><Image accessibilityLabel="배경 분리 전 원본" source={{ uri }} resizeMode="contain" style={{ width: '100%', height: 180, borderRadius: 10, backgroundColor: C.pale }} /></View>
        <View style={{ flex: 1, gap: 6 }}><Text style={{ color: C.muted, fontFamily: font, fontSize: 10 }}>분리 결과</Text><Image accessibilityLabel="배경에서 분리한 의류" source={{ uri: result.cutout }} resizeMode="contain" style={{ width: '100%', height: 180, borderRadius: 10, backgroundColor: '#90969E' }} /></View>
      </View>}
      <Notice>{[result.message, ...result.quality.warnings].join('\n')}</Notice>
      {result.usable && result.cutout && <Button secondary disabled={disabled || applied} onPress={() => onApply(result.cutout!)}>{applied ? '분리 사진을 적용했어요' : '윤곽 확인·분리 사진 적용'}</Button>}
    </>}
    {applied && <Button secondary disabled={disabled || busy} onPress={onRestore}>원본 사진으로 되돌리기</Button>}
    {Boolean(error) && <Text accessibilityRole="alert" style={{ color: '#B54848', fontFamily: font, fontSize: 11, lineHeight: 19 }}>{error}</Text>}
  </View>;
}

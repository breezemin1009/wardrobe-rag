import { useState } from 'react';
import { Linking, Pressable, StyleSheet, Text, View } from 'react-native';
import { C, font } from './theme';
import { Icon, Tag } from './ui';
import type { Evidence } from '@/services/rag';

export function EvidenceCard({ evidence, compact = false }: { evidence: Evidence; compact?: boolean }) {
  const [expanded, setExpanded] = useState(false), [error, setError] = useState('');
  const publicOutfit = evidence.kind === 'outfit' && evidence.publicPair;
  const label = publicOutfit ? '공개 코디 기록' : evidence.label === 'recommend' ? '지원 사례' : evidence.label === 'conditional' ? '조건부 적용' : evidence.label === 'conflict' ? '명시적 충돌' : evidence.kind === 'paper' ? '논문 원문' : evidence.scope === 'observed-outfit' ? '공개 코디 기록' : '연구 기록';
  async function openSource() {
    if (!/^https:\/\//.test(evidence.url)) return;
    try { await Linking.openURL(evidence.url); } catch { setError('원문 링크를 열지 못했어요.'); }
  }
  return <View style={st.card}><View style={st.top}><Tag blue>{label}</Tag><Text style={st.locator}>{evidence.locator}</Text></View><Text style={st.title}>{evidence.title}</Text><Text style={st.copy} numberOfLines={expanded ? undefined : compact ? 3 : 5}>{evidence.text}</Text>
    {publicOutfit ? <><Text style={st.id}>{evidence.sourceId} · 실제로 함께 코디된 상의와 하의</Text><Text style={st.copy}>상의 상품번호 {publicOutfit.top.id}{'\n'}하의 상품번호 {publicOutfit.bottom.id}</Text>{expanded && <Text style={st.copy}>원자료의 상의 이름: {publicOutfit.top.title}{'\n'}원자료의 하의 이름: {publicOutfit.bottom.title}</Text>}</> : evidence.collection === 'style' && <Text style={st.id}>{[evidence.publisher, evidence.requiresDesign ? '의류 종류·디자인 공동 조건의 제한적 적용' : evidence.scope === 'observed-outfit' ? '항목별로 확인한 남성복 코디' : evidence.scope === 'direct_pair' ? '구체적 조합 사례' : evidence.scope === 'partial_outfit' ? '전체 착장의 부분 조합' : evidence.scope === 'capsule_palette' ? '캡슐 팔레트의 조건부 적용' : '색상 관계의 조건부 적용', evidence.checked ? `확인 ${evidence.checked}` : undefined].filter(Boolean).join(' · ')}</Text>}
    {Boolean(evidence.caution) && <View style={st.caution}><Icon name="information-circle-outline" color={C.blue} size={15} /><Text style={st.cautionText}>{evidence.caution}</Text></View>}
    <View style={st.bottom}><Pressable accessibilityRole="button" accessibilityLabel={`${evidence.title} 발췌 ${expanded ? '접기' : '펼치기'}`} onPress={() => setExpanded(!expanded)}><Text style={st.link}>{expanded ? '발췌 접기' : '발췌 펼치기'}</Text></Pressable>{Boolean(evidence.url) && <Pressable accessibilityRole="link" accessibilityLabel={`${evidence.title} 원문 보기`} onPress={openSource} style={st.source}><Text style={st.link}>원문 보기</Text><Icon name="open-outline" color={C.blue} size={14} /></Pressable>}</View><Text style={st.id}>근거 ID {evidence.id}</Text>{Boolean(error) && <Text style={st.error}>{error}</Text>}</View>;
}
const st = StyleSheet.create({ card: { padding: 20, borderWidth: 1, borderColor: C.line, borderRadius: 16, backgroundColor: C.white, marginBottom: 12 }, top: { flexDirection: 'row', alignItems: 'center', flexWrap: 'wrap', gap: 10, marginBottom: 14 }, locator: { fontFamily: font, fontSize: 10, color: C.muted }, title: { color: C.ink, fontFamily: font, fontSize: 14, lineHeight: 22, fontWeight: '700' }, copy: { fontFamily: font, color: '#687187', fontSize: 12, lineHeight: 22, marginTop: 12 }, bottom: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginTop: 17 }, link: { color: C.blue, fontFamily: font, fontSize: 11, fontWeight: '700', paddingVertical: 4 }, source: { flexDirection: 'row', gap: 5, alignItems: 'center' }, caution: { flexDirection: 'row', gap: 7, marginTop: 13, backgroundColor: C.light, padding: 11, borderRadius: 10 }, cautionText: { flex: 1, fontFamily: font, fontSize: 11, lineHeight: 19, color: '#64638D' }, id: { fontFamily: font, color: '#9CA3B2', fontSize: 9, marginTop: 14 }, error: { color: '#B54848', marginTop: 8 } });

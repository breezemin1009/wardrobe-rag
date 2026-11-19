import { router, usePathname } from 'expo-router';
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import type { ReactNode } from 'react';
import { C, font } from './theme';
import { Icon, type IconName } from './ui';
import { useWardrobe } from '@/state/WardrobeProvider';

const destinations: { path: '/' | '/recommendations' | '/saved' | '/knowledge'; label: string; icon: IconName }[] = [
  { path: '/', label: '내 옷장', icon: 'shirt-outline' },
  { path: '/recommendations', label: '구매 추천', icon: 'sparkles-outline' },
  { path: '/saved', label: '저장한 후보', icon: 'bookmark-outline' },
  { path: '/knowledge', label: '자료·근거', icon: 'library-outline' },
];
export function Shell({ children, title, subtitle, action, back = false }: { children: ReactNode; title: string; subtitle?: string; action?: ReactNode; back?: boolean }) {
  const { width } = useWindowDimensions(), desktop = width >= 1000, inset = useSafeAreaInsets(), path = usePathname();
  const store = useWardrobe();
  function goBack() { if (router.canGoBack()) router.back(); else router.replace('/'); }
  function nav(vertical: boolean) { return destinations.map(d => {
    const active = d.path === '/' ? !['/recommendations', '/saved', '/knowledge', '/analysis', '/catalog-comparison'].some(p => path.startsWith(p)) : path.startsWith(d.path) || (d.path === '/recommendations' && path.startsWith('/analysis')) || (d.path === '/knowledge' && path.startsWith('/catalog-comparison'));
    return <Pressable key={d.path} accessibilityRole="button" accessibilityLabel={d.label} accessibilityState={{ selected: active }} onPress={() => router.replace(d.path)} style={[st.navItem, vertical ? st.verticalItem : st.horizontalItem, active && (vertical ? st.active : undefined)]}>
      <Icon name={d.icon} color={active ? C.blue : C.muted} size={vertical ? 20 : 23} /><Text style={[st.navText, active && { color: C.blue, fontWeight: '700' }, !vertical && { fontSize: 10 }]}>{d.label}</Text>
      {vertical && d.path === '/saved' && <Text style={st.savedCount}>{store.savedIds.length}</Text>}
    </Pressable>;
  }); }
  return <View style={[st.root, { paddingTop: inset.top }]}>
    <View style={st.main}><View style={[st.topbar, !desktop && {paddingHorizontal:20}]}><View><Text style={st.brandName}>wardrobe / 옷장</Text><Text style={st.brandSub}>이미 가진 옷에서 시작하는 선택</Text></View>{desktop && <View style={{flexDirection:'row',width:480}}>{nav(false)}</View>}</View>
      <ScrollView key={path} style={{ flex: 1 }} contentContainerStyle={[st.scroll, !desktop && { paddingHorizontal: 20, paddingTop: 26 }]} keyboardShouldPersistTaps="handled">
        <View style={st.content}>
          <View style={st.heading}><View style={{ flex: 1 }}>{back && <Pressable accessibilityRole="button" accessibilityLabel="뒤로" onPress={goBack} style={st.back}><Icon name="arrow-back" size={18} color={C.muted} /><Text style={st.backText}>돌아가기</Text></Pressable>}
            <Text style={[st.title, !desktop && { fontSize: 29 }]}>{title}</Text>{subtitle && <Text style={st.subtitle}>{subtitle}</Text>}</View>{action}</View>
          {!store.ready ? <View style={{ padding: 80 }}><ActivityIndicator color={C.blue} /></View> : <>{store.error && <Text style={st.error}>{store.error}</Text>}{children}</>}
        </View>
      </ScrollView>
      {!desktop && <View style={[st.bottomNav, { paddingBottom: Math.max(inset.bottom, 10) }]}>{nav(false)}</View>}
    </View>
  </View>;
}
const st = StyleSheet.create({
  root: { flex: 1, backgroundColor: C.bg, flexDirection: 'row' }, sidebar: { width: 220, backgroundColor: C.white, borderRightWidth: 1, borderRightColor: C.line, paddingHorizontal: 22, paddingTop: 34 }, brand: { flexDirection: 'row', alignItems: 'center', gap: 9 }, brandIcon: { width: 34, height: 34, borderRadius: 11, backgroundColor: C.blue, alignItems: 'center', justifyContent: 'center' }, brandName: { fontFamily: font, fontSize: 22, color: C.ink, fontWeight: '800', letterSpacing: -.6 }, brandSub: { fontFamily: font, fontSize: 10, color: C.muted, marginTop: 12 }, nav: { gap: 9, marginTop: 52 }, navItem: { flexDirection: 'row', alignItems: 'center', gap: 12 }, verticalItem: { paddingHorizontal: 14, paddingVertical: 16, borderRadius: 12 }, horizontalItem: { flex: 1, flexDirection: 'column', gap: 5, paddingTop: 10 }, navText: { fontFamily: font, fontSize: 13, color: C.muted }, active: { backgroundColor: C.light }, savedCount: { marginLeft: 'auto', color: C.muted, fontSize: 11 }, main: { flex: 1, minWidth: 0 }, topbar: { height: 94, backgroundColor: C.bg, borderBottomWidth: 1, borderBottomColor: C.line, paddingHorizontal: 38, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' }, topbarText: { fontFamily: font, fontSize: 10, color: C.muted, letterSpacing: 1 }, profile: { height: 31, width: 31, borderRadius: 16, backgroundColor: C.light, alignItems: 'center', justifyContent: 'center' }, profileText: { color: C.blue, fontFamily: font, fontSize: 11, fontWeight: '800' }, scroll: { flexGrow: 1, padding: 36, paddingBottom: 45 }, content: { width: '100%', maxWidth: 1160, alignSelf: 'center' }, heading: { flexDirection: 'row', alignItems: 'center', gap: 12, marginBottom: 28 }, title: { fontFamily: font, fontSize: 32, fontWeight: '800', color: C.ink, letterSpacing: -1 }, subtitle: { fontFamily: font, color: C.muted, fontSize: 13, marginTop: 10, lineHeight: 21 }, back: { flexDirection: 'row', gap: 7, alignItems: 'center', marginBottom: 18, alignSelf: 'flex-start', paddingVertical: 3 }, backText: { color: C.muted, fontFamily: font, fontSize: 12 }, bottomNav: { backgroundColor: C.white, borderTopWidth: 1, borderTopColor: C.line, flexDirection: 'row' }, railBottom: { position: 'absolute', bottom: 28, left: 30, flexDirection: 'row', gap: 7, alignItems: 'center' }, sampleDot: { height: 6, width: 6, borderRadius: 3, backgroundColor: C.blue }, railCopy: { fontFamily: font, color: C.muted, fontSize: 10 }, error: { color: '#B54848', backgroundColor: '#FFF0F0', padding: 16, borderRadius: 12, marginBottom: 18 },
});

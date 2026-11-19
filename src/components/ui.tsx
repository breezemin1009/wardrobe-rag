import Ionicons from '@expo/vector-icons/Ionicons';
import { ActivityIndicator, Modal, Pressable, StyleSheet, Text, View, type StyleProp, type ViewStyle } from 'react-native';
import type { ReactNode } from 'react';
import { C, font } from './theme';

export type IconName = React.ComponentProps<typeof Ionicons>['name'];
export function Icon({ name, size = 21, color = C.ink }: { name: IconName; size?: number; color?: string }) { return <Ionicons name={name} size={size} color={color} />; }
export function Button({ children, onPress, secondary, disabled, icon, loading, style,accessibilityLabel }: { children: string; onPress: () => void; secondary?: boolean; disabled?: boolean; icon?: IconName; loading?: boolean; style?: StyleProp<ViewStyle>;accessibilityLabel?:string }) {
  return <Pressable accessibilityRole="button" accessibilityLabel={accessibilityLabel || children} disabled={disabled || loading} onPress={onPress} style={({ pressed }) => [s.button, secondary && s.secondary, (disabled || loading) && { opacity: .5 }, pressed && { opacity: .8 }, style]}>
    {loading ? <ActivityIndicator color={secondary ? C.blue : C.white} /> : icon && <Icon name={icon} color={secondary ? C.blue : C.white} size={19} />}
    <Text style={[s.buttonText, secondary && { color: C.blue }]}>{children}</Text>
  </Pressable>;
}
export function Tag({ children, blue = false }: { children: ReactNode; blue?: boolean }) { return <View style={[s.tag, blue && { backgroundColor: C.light }]}><Text style={[s.tagText, blue && { color: C.blue }]}>{children}</Text></View>; }
export function Empty({ title, description, action }: { title: string; description: string; action?: ReactNode }) { return <View style={s.empty}><View style={s.emptyIcon}><Icon name="shirt-outline" size={34} color={C.blue} /></View><Text style={s.emptyTitle}>{title}</Text><Text style={s.emptyCopy}>{description}</Text>{action}</View>; }
export function Loading({message='저장한 의류 정보를 불러오고 있어요.'}:{message?:string}) { return <View style={s.empty} accessibilityRole="progressbar" accessibilityLabel={message}><ActivityIndicator color={C.blue} /><Text style={s.emptyCopy}>{message}</Text></View>; }
export function Sheet({ title, visible, onClose, children }: { title: string; visible: boolean; onClose: () => void; children: ReactNode }) { return <Modal visible={visible} transparent animationType="fade" onRequestClose={onClose}><View style={s.overlay}><Pressable accessibilityLabel="창 닫기" onPress={onClose} style={StyleSheet.absoluteFill} /><View style={s.dialog}><View style={s.dialogHeader}><Text style={s.dialogTitle}>{title}</Text><Pressable accessibilityRole="button" accessibilityLabel="닫기" onPress={onClose} style={s.close}><Icon name="close" /></Pressable></View>{children}</View></View></Modal>; }
export function Notice({ children }: { children: string }) { return <View style={s.notice}><Icon name="information-circle-outline" color={C.blue} size={18} /><Text style={s.noticeText}>{children}</Text></View>; }
export const s = StyleSheet.create({
  button: { minHeight: 48, paddingVertical: 13, paddingHorizontal: 21, borderRadius: 12, backgroundColor: C.blue, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
  secondary: { backgroundColor: C.light }, buttonText: { fontFamily: font, fontSize: 14, fontWeight: '700', color: C.white },
  tag: { paddingHorizontal: 9, paddingVertical: 5, borderRadius: 6, backgroundColor: C.pale }, tagText: { fontFamily: font, color: C.muted, fontSize: 11, fontWeight: '600' },
  empty: { alignItems: 'center', padding: 36, gap: 14 }, emptyIcon: { backgroundColor: C.light, borderRadius: 22, padding: 20 }, emptyTitle: { color: C.ink, fontFamily: font, fontSize: 19, fontWeight: '700' }, emptyCopy: { color: C.muted, fontFamily: font, textAlign: 'center', lineHeight: 23, fontSize: 14 },
  overlay: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24, backgroundColor: '#121B3C66' }, dialog: { width: '100%', maxWidth: 520, maxHeight: '85%', padding: 24, borderRadius: 22, backgroundColor: C.white }, dialogHeader: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }, dialogTitle: { fontFamily: font, fontSize: 20, fontWeight: '700', color: C.ink }, close: { padding: 8 },
  notice: { backgroundColor: C.light, borderRadius: 12, padding: 14, flexDirection: 'row', alignItems: 'flex-start', gap: 8 }, noticeText: { fontFamily: font, color: '#64638D', fontSize: 12, lineHeight: 19, flex: 1 },
});

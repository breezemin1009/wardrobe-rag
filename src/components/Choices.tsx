import { Pressable, StyleSheet, Text, View } from 'react-native';
import { C, font } from './theme';
export function Choices<T extends string>({ options, value, onChange }: { options: readonly T[]; value: T; onChange: (value: T) => void }) {
  return <View style={st.row}>{options.map(option => <Pressable key={option} accessibilityRole="button" accessibilityLabel={option} accessibilityState={{ selected: value === option }} aria-selected={value === option} onPress={() => onChange(option)} style={[st.option, value === option && st.selected]}><Text style={[st.copy, value === option && st.active]}>{option}</Text></Pressable>)}</View>;
}
const st = StyleSheet.create({ row: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 }, option: { paddingHorizontal: 14, paddingVertical: 11, borderRadius: 10, borderWidth: 1, borderColor: C.line, backgroundColor: C.white }, selected: { backgroundColor: C.light, borderColor: '#DCD9FF' }, copy: { fontFamily: font, fontSize: 12, color: C.muted }, active: { color: C.blue, fontWeight: '700' } });

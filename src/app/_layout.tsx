import { Stack } from 'expo-router';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { WardrobeProvider } from '@/state/WardrobeProvider';
import { RagProvider } from '@/state/RagProvider';
export default function RootLayout() {
  return <SafeAreaProvider><WardrobeProvider><RagProvider><StatusBar style="dark" /><Stack screenOptions={{ headerShown: false, animation: 'fade', contentStyle: { backgroundColor: '#F7F8FB' } }} /></RagProvider></WardrobeProvider></SafeAreaProvider>;
}

import { useState } from 'react';
import { Image, View, StyleSheet } from 'react-native';
import type { Garment } from '@/domain/model';
import { Icon } from './ui';
import { C } from './theme';

const sheets = { studio: require('../../assets/demo/garments-studio-v2.png'), patterns: require('../../assets/demo/garments-patterns-v2.png') };
const crops: Record<string, { sheet: keyof typeof sheets; bounds: [number, number, number, number] }> = {
  'white-shirt': { sheet: 'studio', bounds: [0, 0, 418, 384] },
  'cream-tee': { sheet: 'studio', bounds: [418, 0, 836, 384] },
  'blue-shirt': { sheet: 'studio', bounds: [836, 0, 1254, 384] },
  'black-knit': { sheet: 'studio', bounds: [0, 384, 418, 746] },
  'grey-sweatshirt': { sheet: 'studio', bounds: [418, 384, 836, 746] },
  'olive-shirt': { sheet: 'studio', bounds: [836, 384, 1254, 746] },
  'black-pants': { sheet: 'studio', bounds: [0, 746, 418, 1254] },
  denim: { sheet: 'studio', bounds: [418, 746, 836, 1254] },
  'beige-pants': { sheet: 'studio', bounds: [836, 746, 1254, 1254] },
  'blue-stripe': { sheet: 'patterns', bounds: [0, 0, 418, 397] },
  'breton-tee': { sheet: 'patterns', bounds: [418, 0, 836, 397] },
  'olive-check': { sheet: 'patterns', bounds: [836, 0, 1254, 397] },
  'beige-knit': { sheet: 'patterns', bounds: [0, 397, 418, 782] },
  'cream-knit': { sheet: 'patterns', bounds: [418, 397, 836, 782] },
  'navy-polo': { sheet: 'patterns', bounds: [836, 397, 1254, 782] },
  'grey-pants': { sheet: 'patterns', bounds: [0, 782, 418, 1254] },
  'olive-pants': { sheet: 'patterns', bounds: [418, 782, 836, 1254] },
  'cream-pants': { sheet: 'patterns', bounds: [836, 782, 1254, 1254] },
};
/** Clips the original generated demo sheet at render time; never modifies source pixels. */
export function GarmentPhoto({ garment, height = 176 }: { garment: Pick<Garment, 'imageUri' | 'cutoutUri' | 'demoAsset' | 'name'>; height?: number }) {
  const [width, setWidth] = useState(0);
  const [failedSources, setFailedSources] = useState<string[]>([]);
  const uri = [garment.cutoutUri, garment.imageUri].find(value => value && !failedSources.includes(value));
  const sourceKey = uri || garment.demoAsset || 'empty';
  const failed = failedSources.includes(sourceKey);
  const fail = () => setFailedSources(values => values.includes(sourceKey) ? values : [...values, sourceKey]);
  const crop = garment.demoAsset ? crops[garment.demoAsset] : undefined;
  if (uri && !failed) return <Image accessibilityLabel={garment.name} source={{ uri }} style={{ width: '100%', height, borderRadius: 12, backgroundColor: uri === garment.cutoutUri ? '#90969E' : C.pale }} resizeMode="contain" onError={fail} />;
  if (!crop || failed) return <View style={[st.placeholder, { height }]}><Icon name="shirt-outline" size={40} color={C.muted} /></View>;
  const [x1, y1, x2, y2] = crop.bounds, scale = Math.min(width / (x2 - x1), height / (y2 - y1));
  const sheetWidth = 1254, sheetHeight = 1254;
  return <View accessibilityLabel={garment.name} onLayout={e => setWidth(e.nativeEvent.layout.width)} style={[st.crop, { height }]}>{width > 0 && <View style={{ width: (x2 - x1) * scale, height: (y2 - y1) * scale, overflow: 'hidden' }}><Image source={sheets[crop.sheet]} resizeMode="stretch" style={{ position: 'absolute', width: sheetWidth * scale, height: sheetHeight * scale, left: -x1 * scale, top: -y1 * scale }} onError={fail} /></View>}</View>;
}
const st = StyleSheet.create({ crop: { width: '100%', borderRadius: 12, overflow: 'hidden', backgroundColor: '#F7F7F5', alignItems: 'center', justifyContent: 'center' }, placeholder: { width: '100%', alignItems: 'center', justifyContent: 'center', backgroundColor: C.pale, borderRadius: 12 } });

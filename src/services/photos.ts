import * as ImagePicker from 'expo-image-picker';
import { Platform } from 'react-native';
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';

export async function selectPhoto(camera = false): Promise<string | null> {
  if (camera) {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) throw new Error('카메라 권한이 필요해요. 사진 선택으로도 등록할 수 있습니다.');
  }
  const options: ImagePicker.ImagePickerOptions = { mediaTypes: ['images'], allowsEditing: false, quality: 0.75, base64: Platform.OS === 'web' };
  const result = await (camera ? ImagePicker.launchCameraAsync(options) : ImagePicker.launchImageLibraryAsync(options));
  if (result.canceled || !result.assets?.length) return null;
  const asset = result.assets[0];
  if (Platform.OS === 'web') {
    if (!asset.base64) throw new Error('사진을 읽지 못했어요. 다른 이미지로 다시 시도해주세요.');
    if (asset.base64.length > 2_000_000) {
      // The web picker ignores quality. Optimise large photos on the device,
      // preserving aspect ratio; small existing photos keep their original hash.
      const context = ImageManipulator.manipulate(asset.uri);
      try {
        const edge = Math.max(asset.width, asset.height);
        if (edge > 1280) context.resize(asset.width >= asset.height ? { width: 1280, height: null } : { width: null, height: 1280 });
        const rendered = await context.renderAsync();
        try {
          let output = await rendered.saveAsync({ format: SaveFormat.JPEG, compress: .82, base64: true });
          if (!output.base64 || output.base64.length > 2_000_000) output = await rendered.saveAsync({ format: SaveFormat.JPEG, compress: .6, base64: true });
          if (!output.base64 || output.base64.length > 2_000_000) throw new Error('사진 용량을 줄이지 못했어요. 다른 사진을 선택해주세요.');
          return `data:image/jpeg;base64,${output.base64}`;
        } finally { rendered.release(); }
      } finally { context.release(); }
    }
    return `data:${asset.mimeType || 'image/jpeg'};base64,${asset.base64}`;
  }
  return asset.uri;
}

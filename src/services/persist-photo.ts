import { Platform } from 'react-native';

export async function persistPhoto(uri: string | undefined): Promise<string | undefined> {
  if (!uri || Platform.OS === 'web') return uri;
  const { Directory, File, Paths } = await import('expo-file-system');
  const directory = new Directory(Paths.document, 'wardrobe-photos');
  await directory.create({ idempotent: true, intermediates: true });
  if (uri.startsWith(directory.uri)) return uri;
  const data = /^data:image\/(png|jpeg);base64,(.+)$/.exec(uri);
  if (data) {
    const destination = new File(directory, `${Date.now()}-${Math.random().toString(36).slice(2, 8)}.${data[1] === 'png' ? 'png' : 'jpg'}`);
    destination.create();
    destination.write(data[2], { encoding: 'base64' });
    return destination.uri;
  }
  const extension = uri.split('.').pop()?.split('?')[0] || 'jpg';
  const destination = new File(directory, `${Date.now()}-${Math.random().toString(36).slice(2, 8)}.${extension}`);
  await new File(uri).copy(destination);
  return destination.uri;
}

import { Platform } from 'react-native';
import * as Crypto from 'expo-crypto';

/** Exact file bytes only: retakes, crops and changed lighting are not identities. */
export async function fingerprintPhoto(uri: string): Promise<string> {
  let bytes: Uint8Array;
  if (Platform.OS === 'web') {
    const response = await fetch(uri);
    if (!response.ok) throw new Error('선택한 사진을 읽지 못했어요. 다시 선택해주세요.');
    bytes = new Uint8Array(await response.arrayBuffer());
  } else {
    const { File } = await import('expo-file-system');
    bytes = await new File(uri).bytes();
  }
  const digest = await Crypto.digest(Crypto.CryptoDigestAlgorithm.SHA256, new Uint8Array(bytes));
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('');
}

import { Platform } from 'react-native';
import { api } from './rag';
import type { Garment } from '@/domain/model';
import type { ViewDesign } from '@/domain/design';

export type PhotoProposal = { proposal: Pick<Garment, 'category' | 'color' | 'pattern'> & { uncertain: string[] }; model: string; reviewRequired: true; message: string };
export type DesignPhotoProposal = { proposal: { category:Garment['category']; garmentType:NonNullable<Garment['design']>['garmentType']; garmentPresent:boolean; singleGarment:boolean; front:ViewDesign; back:ViewDesign|null; pairCheck:'single'|'consistent'|'different'|'uncertain'; uncertain:string[] }; legacy:Pick<Garment,'category'|'color'|'pattern'>; model:string; reviewRequired:true; message:string };
export type CutoutResult = { usable: boolean; cutout: string | null; quality: { needsReview: boolean; warnings: string[] }; model: string; message: string; reviewRequired: true };
async function photoData(uri: string): Promise<string> {
  let image = uri;
  if (Platform.OS !== 'web' && !uri.startsWith('data:image/')) {
    const { File } = await import('expo-file-system');
    const file = new File(uri);
    if (file.size > 3_000_000) throw new Error('AI 분석은 3MB 이하 사진을 사용해주세요. 직접 입력도 가능해요.');
    image = `data:image/${uri.toLowerCase().endsWith('.png') ? 'png' : 'jpeg'};base64,${await file.base64()}`;
  }
  if (!image.startsWith('data:image/')) throw new Error('사진을 다시 선택해주세요.');
  return image;
}
export async function analysePhoto(uri: string, signal: AbortSignal): Promise<PhotoProposal> {
  const image = await photoData(uri);
  return api<PhotoProposal>('/photo-attributes', { image, consent: true }, signal);
}
export async function extractPhoto(uri: string, signal: AbortSignal): Promise<CutoutResult> {
  const image = await photoData(uri);
  return api<CutoutResult>('/photo-cutout', { image, consent: true }, signal);
}
export async function analyseDesign(uri: string, backUri: string | undefined, signal: AbortSignal): Promise<DesignPhotoProposal> {
  const [front,back] = await Promise.all([photoData(uri),backUri ? photoData(backUri) : Promise.resolve(undefined)]);
  return api<DesignPhotoProposal>('/photo-design',{front,back,consent:true},signal);
}

import type { DesignProfile } from './design.ts';
export const categories = ['상의', '하의'] as const;
export const colors = ['화이트', '크림', '블루', '블랙', '베이지', '그레이', '올리브', '기타'] as const;
export const patterns = ['무지', '스트라이프', '체크', '프린트', '기타'] as const;
export const occasions = ['데일리', '출근', '나들이'] as const;
export const demoAssets = ['white-shirt', 'cream-tee', 'blue-shirt', 'black-knit', 'black-pants', 'denim', 'beige-pants', 'grey-pants', 'olive-pants', 'grey-sweatshirt', 'olive-shirt', 'blue-stripe', 'breton-tee', 'olive-check', 'beige-knit', 'cream-knit', 'navy-polo', 'cream-pants'] as const;
export type Category = (typeof categories)[number];
export type Audience = 'mens' | 'unisex' | 'unknown' | 'womens';
export type Garment = {
  id: string;
  name: string;
  category: Category;
  color: (typeof colors)[number];
  pattern: (typeof patterns)[number];
  occasion: (typeof occasions)[number];
  role: 'owned' | 'candidate';
  audience?: Audience; // Missing fields in old local records require user review.
  sourceProductId?: string;
  referenceProductId?: string;
  demoAsset?: (typeof demoAssets)[number];
  imageUri?: string;
  cutoutUri?: string;
  imageHash?: string;
  backUri?: string;
  design?: DesignProfile;
  isExample: boolean;
  createdAt: number;
};
export type AppState = { version: 1; garments: Garment[]; archived: Garment[]; savedIds: string[] };
export type Evaluation = {
  status: 'example' | 'unavailable';
  score?: number;
  combination?: number;
  contribution?: number;
  message: string;
  pairs: { garmentId: string; reason: string }[];
};

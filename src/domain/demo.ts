import type { AppState, Garment } from './model.ts';

const base = { audience: 'mens', pattern: '무지', occasion: '데일리', isExample: true, createdAt: 1790812800000 } as const;
export const exampleWardrobe: Garment[] = [
  { ...base, id: 'owned-white', name: '화이트 옥스퍼드 셔츠', category: '상의', color: '화이트', role: 'owned', demoAsset: 'white-shirt' },
  { ...base, id: 'owned-cream', name: '크림 코튼 티셔츠', category: '상의', color: '크림', role: 'owned', demoAsset: 'cream-tee' },
  { ...base, id: 'owned-blue', name: '블루 스트레이트 셔츠', category: '상의', color: '블루', role: 'owned', demoAsset: 'blue-shirt' },
  { ...base, id: 'owned-knit', name: '블랙 라운드 니트', category: '상의', color: '블랙', role: 'owned', demoAsset: 'black-knit' },
  { ...base, id: 'owned-black', name: '블랙 스트레이트 팬츠', category: '하의', color: '블랙', role: 'owned', demoAsset: 'black-pants' },
  { ...base, id: 'owned-denim', name: '미드 블루 데님', category: '하의', color: '블루', role: 'owned', demoAsset: 'denim' },
];
export const catalog: Garment[] = [
  { ...base, id: 'catalog-beige', name: '베이지 코튼 팬츠', category: '하의', color: '베이지', role: 'candidate', demoAsset: 'beige-pants' },
  { ...base, id: 'catalog-grey', name: '그레이 테일러드 팬츠', category: '하의', color: '그레이', role: 'candidate', demoAsset: 'grey-pants' },
  { ...base, id: 'catalog-olive', name: '올리브 와이드 팬츠', category: '하의', color: '올리브', role: 'candidate', demoAsset: 'olive-pants' },
  { ...base, id: 'catalog-shirt', name: '라이트 블루 셔츠', category: '상의', color: '블루', role: 'candidate', demoAsset: 'blue-shirt' },
  { ...base, id: 'catalog-cream-pants', name: '크림 스트레이트 진', category: '하의', color: '크림', role: 'candidate', demoAsset: 'cream-pants' },
  { ...base, id: 'catalog-denim', name: '미드 블루 데님', category: '하의', color: '블루', role: 'candidate', demoAsset: 'denim' },
  { ...base, id: 'catalog-black-pants', name: '블랙 테일러드 팬츠', category: '하의', color: '블랙', role: 'candidate', demoAsset: 'black-pants' },
  { ...base, id: 'catalog-white', name: '화이트 옥스퍼드 셔츠', category: '상의', color: '화이트', role: 'candidate', demoAsset: 'white-shirt' },
  { ...base, id: 'catalog-cream-tee', name: '크림 코튼 티셔츠', category: '상의', color: '크림', role: 'candidate', demoAsset: 'cream-tee' },
  { ...base, id: 'catalog-black-knit', name: '블랙 라운드 니트', category: '상의', color: '블랙', role: 'candidate', demoAsset: 'black-knit' },
  { ...base, id: 'catalog-grey-top', name: '그레이 크루넥 스웨트셔츠', category: '상의', color: '그레이', role: 'candidate', demoAsset: 'grey-sweatshirt' },
  { ...base, id: 'catalog-olive-top', name: '올리브 코튼 오버셔츠', category: '상의', color: '올리브', role: 'candidate', demoAsset: 'olive-shirt' },
  { ...base, id: 'catalog-blue-stripe', name: '블루 스트라이프 셔츠', category: '상의', color: '블루', pattern: '스트라이프', role: 'candidate', demoAsset: 'blue-stripe' },
  { ...base, id: 'catalog-breton', name: '화이트 브레톤 티셔츠', category: '상의', color: '화이트', pattern: '스트라이프', role: 'candidate', demoAsset: 'breton-tee' },
  { ...base, id: 'catalog-check', name: '올리브 깅엄 체크 셔츠', category: '상의', color: '올리브', pattern: '체크', role: 'candidate', demoAsset: 'olive-check' },
  { ...base, id: 'catalog-beige-knit', name: '베이지 크루넥 니트', category: '상의', color: '베이지', role: 'candidate', demoAsset: 'beige-knit' },
  { ...base, id: 'catalog-cream-knit', name: '크림 케이블 니트', category: '상의', color: '크림', role: 'candidate', demoAsset: 'cream-knit' },
  { ...base, id: 'catalog-navy-polo', name: '네이비 코튼 폴로', category: '상의', color: '블루', role: 'candidate', demoAsset: 'navy-polo' },
];
export const exampleScores: Record<string, { score: number; combination: number; contribution: number }> = {
  'catalog-beige': { score: 70, combination: 75, contribution: 60 },
  'catalog-grey': { score: 64, combination: 72, contribution: 48 },
  'catalog-olive': { score: 56, combination: 63, contribution: 42 },
};
export function initialState(): AppState {
  return { version: 1, garments: [...exampleWardrobe.map(g => ({ ...g })), { ...catalog[0] }], archived: [], savedIds: [] };
}

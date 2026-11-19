import type { Garment } from './model.ts';

export const audienceLabels = { mens: '남성복', unisex: '공용', unknown: '의류 구분 확인 필요', womens: '현재 추천 범위 밖' } as const;
export function inMenswearScope(g: Pick<Garment, 'audience'>): boolean {
  return g.audience === 'mens' || g.audience === 'unisex';
}
export function scopeReview(garments: Garment[]): Garment[] {
  return garments.filter(g => !inMenswearScope(g));
}

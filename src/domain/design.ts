export const designColors = ['화이트','크림','블루','블랙','베이지','그레이','올리브','레드','핑크','브라운','옐로','오렌지','퍼플','그린','기타'] as const;
export const designPatterns = ['무지','스트라이프','체크','프린트','기타'] as const;
export const scales = ['none','fine','medium','bold','unknown'] as const;
export const contrasts = ['low','medium','high','unknown'] as const;
export const accentKinds = ['logo','text','graphic','patch','other'] as const;
export const accentSizes = ['small','medium','large','unknown'] as const;
export const placements = ['chest','center','upper','lower','side','all-over','other'] as const;
export const maxDesignColors = 6;
export const maxDesignAccents = 6;
export const garmentTypes = ['tshirt','shirt','knit','jacket','jeans','trousers','shorts','skirt','other','unknown'] as const;
export const garmentTypeLabels = { tshirt:'티셔츠',shirt:'셔츠',knit:'니트',jacket:'재킷',jeans:'청바지',trousers:'일반 바지',shorts:'반바지',skirt:'치마',other:'기타',unknown:'미확인' } as const;
export type DesignColour = typeof designColors[number];
export type Accent = { kind: typeof accentKinds[number]; size: typeof accentSizes[number]; placement: typeof placements[number]; colors: DesignColour[] };
export type ViewDesign = { baseColor: DesignColour; pattern: typeof designPatterns[number]; patternColors: DesignColour[]; patternScale: typeof scales[number]; contrast: typeof contrasts[number]; accents: Accent[]; uncertain: string[] };
export type DesignProfile = { version: 1; garmentType?: typeof garmentTypes[number]; front: ViewDesign; back: ViewDesign | null; backStatus: 'unknown' | 'observed' | 'same-as-front' | 'plain-confirmed' };
export const scaleLabels = { none:'없음', fine:'가는·작은', medium:'보통', bold:'굵은·큰', unknown:'미확인' } as const;
export const contrastLabels = { low:'낮음', medium:'보통', high:'높음', unknown:'미확인' } as const;
export const kindLabels = { logo:'로고', text:'문자', graphic:'그래픽', patch:'패치', other:'기타 장식' } as const;
export const sizeLabels = { small:'작음', medium:'보통', large:'큼', unknown:'미확인' } as const;
export const placementLabels = { chest:'가슴', center:'중앙', upper:'상단', lower:'하단', side:'측면', 'all-over':'전체', other:'기타 위치' } as const;
export const backLabels = { unknown:'뒷면 미확인', observed:'뒷면 별도 확인', 'same-as-front':'앞면과 동일', 'plain-confirmed':'같은 바탕색·무지·장식 없음' } as const;
export function emptyView(baseColor: DesignColour = '화이트', pattern: ViewDesign['pattern'] = '무지'): ViewDesign {
  return { baseColor, pattern, patternColors: [], patternScale: pattern === '무지' ? 'none' : 'unknown', contrast: 'unknown', accents: [], uncertain: ['accents'] };
}
export function legacyDesignColour(value: DesignColour) { return (['화이트','크림','블루','블랙','베이지','그레이','올리브'] as string[]).includes(value) ? value as '화이트'|'크림'|'블루'|'블랙'|'베이지'|'그레이'|'올리브' : '기타'; }
export function designCategory(type: DesignProfile['garmentType']) {
  return ['tshirt','shirt','knit','jacket'].includes(type || '') ? '상의' as const : ['jeans','trousers','shorts','skirt'].includes(type || '') ? '하의' as const : undefined;
}
export function viewSummary(view: ViewDesign) {
  return [`${view.baseColor} 바탕 · ${view.patternColors.join('·') || '추가 무늬색 없음'} · ${view.pattern}${view.pattern === '무지' ? '' : ` (${scaleLabels[view.patternScale]})`}`,
    ...view.accents.map(a => `${placementLabels[a.placement]} ${sizeLabels[a.size]} ${kindLabels[a.kind]} · ${a.colors.join('·') || '색 미확인'}`),
    ...(view.uncertain.length ? ['미확인 항목 있음 · 사진과 비교해주세요'] : [])].join('\n');
}
export function validDesign(value: unknown): value is DesignProfile {
  if (!value || typeof value !== 'object') return false;
  const d = value as DesignProfile;
  const palette = (v: unknown, max: number) => Array.isArray(v) && v.length <= max && v.every(c => designColors.includes(c));
  const view = (v: ViewDesign | null) => Boolean(v && designColors.includes(v.baseColor) && designPatterns.includes(v.pattern)
    && scales.includes(v.patternScale) && contrasts.includes(v.contrast) && palette(v.patternColors,maxDesignColors)
    && Array.isArray(v.uncertain) && v.uncertain.length <= 6 && v.uncertain.every(k => ['baseColor','pattern','patternColors','patternScale','contrast','accents'].includes(k))
    && Array.isArray(v.accents) && v.accents.length <= maxDesignAccents && v.accents.every(a => a && accentKinds.includes(a.kind) && accentSizes.includes(a.size) && placements.includes(a.placement) && palette(a.colors,maxDesignColors)));
  return d.version === 1 && (d.garmentType === undefined || garmentTypes.includes(d.garmentType)) && view(d.front) && Object.keys(backLabels).includes(d.backStatus)
    && (d.backStatus === 'observed' ? view(d.back) : d.back === null);
}

import { categories, colors, patterns, occasions, demoAssets } from './model.ts';
import type { AppState, Evaluation, Garment } from './model.ts';
import { validDesign, legacyDesignColour,designCategory } from './design.ts';
import { exampleScores, exampleWardrobe, catalog } from './demo.ts';

function signature(g: Garment): string {
  return JSON.stringify([g.id, g.name, g.category, g.color, g.pattern, g.occasion, g.role, g.demoAsset, g.imageUri, g.backUri, g.isExample,g.design,g.audience,g.sourceProductId]);
}
export function isExampleWardrobe(garments: Garment[]): boolean {
  const owned = garments.filter(g => g.role === 'owned');
  return owned.length === exampleWardrobe.length && owned.every(g => exampleWardrobe.some(seed => signature(seed) === signature(g)));
}
/** Fixed preview fixtures. Never scores uploaded photos or a changed wardrobe. */
export function previewEvaluation(candidate: Garment, garments: Garment[]): Evaluation {
  const seed = catalog.find(g => signature(g) === signature(candidate));
  const fixture = seed && exampleScores[seed.id];
  if (!garments.some(g => g.role === 'owned')) {
    return { status: 'unavailable', message: '먼저 보유 의류를 등록해주세요. 구매 후보를 비교할 옷장이 필요해요.', pairs: [] };
  }
  if (!isExampleWardrobe(garments) || !fixture) {
    return { status: 'unavailable', message: '의류 정보는 저장됐어요. 등록한 옷의 실제 평가는 분석 기능을 연결한 후 제공됩니다.', pairs: [] };
  }
  return {
    status: 'example', ...fixture,
    message: '예시 옷장에 대한 화면 시연용 점수입니다. 실제 AI 분석 결과나 만족 확률이 아닙니다.',
    pairs: [
      { garmentId: 'owned-white', reason: '화이트 셔츠와의 조합을 확인하는 화면 예시' },
      { garmentId: 'owned-blue', reason: '블루 셔츠와의 조합을 확인하는 화면 예시' },
    ],
  };
}
function isGarment(value: unknown): value is Garment {
  if (!value || typeof value !== 'object') return false;
  const g = value as Garment;
  return typeof g.id === 'string' && g.id.length > 0 && typeof g.name === 'string' && g.name.trim().length > 0 && g.name.length <= 60
    && categories.includes(g.category) && colors.includes(g.color) && patterns.includes(g.pattern) && occasions.includes(g.occasion)
    && ['owned', 'candidate'].includes(g.role) && typeof g.isExample === 'boolean' && Number.isFinite(g.createdAt)
    && (g.audience === undefined || ['mens','unisex','unknown','womens'].includes(g.audience))
    && (g.sourceProductId === undefined || typeof g.sourceProductId === 'string' && /^\d{1,10}$/.test(g.sourceProductId))
    && (g.demoAsset === undefined || demoAssets.includes(g.demoAsset))
    && (g.imageUri === undefined || typeof g.imageUri === 'string') && (g.backUri === undefined || typeof g.backUri === 'string')
    && (g.cutoutUri === undefined || typeof g.cutoutUri === 'string' && Boolean(g.imageUri))
    && (g.design === undefined || validDesign(g.design) && legacyDesignColour(g.design.front.baseColor) === g.color && g.design.front.pattern === g.pattern && (!designCategory(g.design.garmentType) || designCategory(g.design.garmentType) === g.category))
    && (g.imageHash === undefined || typeof g.imageHash === 'string' && /^[a-f0-9]{64}$/.test(g.imageHash))
    && Boolean(g.demoAsset || g.imageUri);
}
export function parseState(raw: string): AppState {
  const value: unknown = JSON.parse(raw);
  if (!value || typeof value !== 'object') throw new Error('저장 형식이 올바르지 않습니다.');
  const s = value as AppState;
  if (s.version !== 1 || !Array.isArray(s.garments) || !s.garments.every(isGarment)
    || !Array.isArray(s.archived) || !s.archived.every(isGarment)
    || !Array.isArray(s.savedIds) || !s.savedIds.every(id => typeof id === 'string')) throw new Error('저장 데이터를 확인할 수 없습니다.');
  const ids = [...s.garments, ...s.archived].map(g => g.id);
  if (new Set(ids).size !== ids.length) throw new Error('의류 ID가 중복되었습니다.');
  const migrate = (g: Garment): Garment => {
    if (g.audience) return g;
    // Recognize only unchanged built-in examples. A user photo, even carrying
    // the same ID or isExample flag, never inherits an audience classification.
    const seed = [...exampleWardrobe, ...catalog].find(v => signature({ ...v, audience: undefined }) === signature(g));
    return { ...g, audience: seed ? seed.audience : 'unknown' };
  };
  return { ...s, garments: s.garments.map(migrate), archived: s.archived.map(migrate), savedIds: [...new Set(s.savedIds)] };
}
export function archiveGarment(s: AppState, id: string): AppState {
  const garment = s.garments.find(g => g.id === id);
  if (!garment) return s;
  return { ...s, garments: s.garments.filter(g => g.id !== id), archived: [...s.archived, garment], savedIds: s.savedIds.filter(v => v !== id) };
}
export function duplicatePhoto(s: AppState, g: Garment): Garment | undefined {
  return [...s.garments, ...s.archived].find(v => v.id !== g.id && Boolean(
    g.sourceProductId && v.sourceProductId === g.sourceProductId || g.imageHash && v.imageHash === g.imageHash || g.imageUri && v.imageUri === g.imageUri));
}
export function saveGarment(s: AppState, g: Garment): AppState {
  if (!isGarment(g)) throw new Error('사진과 의류의 기본 정보·앞뒤 디자인을 함께 확인해주세요.');
  if (g.role === 'owned' && s.garments.filter(v => v.role === 'owned' && v.id !== g.id).length >= 64)
    throw new Error('한 번에 분석할 보유 옷은 최대 64벌이에요. 보관함으로 옮긴 후 다시 등록해주세요.');
  const duplicate = duplicatePhoto(s, g);
  if (duplicate) throw new Error(`같은 사진이 '${duplicate.name}'에 등록되어 있어요. 기존 의류를 수정하거나 복원해주세요.`);
  if (s.archived.some(v => v.id === g.id)) throw new Error('보관함의 의류를 먼저 복원해주세요.');
  return { ...s, garments: s.garments.some(v => v.id === g.id) ? s.garments.map(v => v.id === g.id ? g : v) : [...s.garments, g] };
}
export function restoreGarment(s: AppState, id: string): AppState {
  const garment = s.archived.find(g => g.id === id);
  if (!garment) return s;
  if (garment.role === 'owned' && s.garments.filter(g => g.role === 'owned').length >= 64)
    throw new Error('한 번에 분석할 보유 옷은 최대 64벌이에요. 다른 옷을 보관함으로 옮긴 후 복원해주세요.');
  return { ...s, garments: [...s.garments, garment], archived: s.archived.filter(g => g.id !== id) };
}

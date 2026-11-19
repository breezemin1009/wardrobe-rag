import type { Garment } from '@/domain/model';
import { assessmentDesign } from '@/domain/assessment-garment';

export type SearchMode = 'keyword' | 'vector' | 'hybrid' | 'conditioned';
export type Evidence = { id: string; sourceId: string; title: string; text: string; locator: string; url: string;
  collection: 'research' | 'style'; kind: string; currentScope: boolean; label?: string; caution?: string;
  scope?: string; publisher?: string; sourceFamily?: string; checked?: string; annotationVersion?: string;
  publicPair?: { top: {id:string;title:string}; bottom: {id:string;title:string}; sourceOutfits: string[] };
  requiresDesign?: boolean;
  retrieval: { bm25: number; cosine: number; rrf: number } };
export type Health = { status: string; sources: number; chunks: number; styleEvidence: number; encoder: string; revision: string; generation: string; version: string; corpusSha256: string };
export type Source = { id: string; title: string; kind: string; url: string; collection: string; chunks: number; pages?: number; slides?: number; currentScope?: boolean };
export type Assessment = { candidateId: string; evaluationFingerprint: string; counts: { totalGroups: number; supportedGroups: number; comparableGroups: number; baselineGroups: number; uniqueFamilies: number }; status: 'scored' | 'partial' | 'insufficient'; score: number | null; combination: number | null; contribution: number | null;
  inputReviews?:{garmentId:string;fields:string[]}[];
  assessmentScope:'basic-attributes'|'front-back-design'|'public-case'; partialAssessment:{frontSupportedGroups:number;baseSupportedGroups:number;totalGroups:number;frontCombination:number|null;label:string};
  evidenceQuality: { directGroups: number; ruleOnlyGroups: number; level: string; label: string };
  gaps: { groupId: string; label: string; missingDesign:boolean; missingCandidate: boolean; missingBaselineGroups: string[] }[];
  coverage: number; gainCoverage: number; message: string; evidence: Evidence[]; limitations: string[]; runId: string;
  groups: { groupId: string; garmentIds: string[]; label: string; support: number | null; existingSupport: number | null; gain: number | null; evidenceIds: string[];
    gainProof?:{type:'known-maximum'|'covered-by-existing-alternative'|'unknown';knownExistingLowerBound:number};
    designAudit?:{scope:'front-only'|'front-back';missingBack:boolean;legacyInputs:string[];front:DesignViewAssessment;back:DesignViewAssessment|null;aggregation:string};
    grounds: { familyCount: number; directFamilyCount: number; families: { family: string; support: number; evidenceIds: string[] }[] } }[];
  method: { version: string; alpha: number; minimumCoverage: number; grouping: string; formula: string; retrieval: string; aggregation: string }; rank?: number | null };
export type GeneratedAnswer = { mode: string; text: string; citations: string[]; warning?: string; reason?: string; model?: string; cached?: boolean };
export type SensitivityRow = { score:number|null;status:string;coverage:number;gainCoverage:number;comparableGroups:number;sameComputedScope:boolean;comparableToBase:boolean;scoreDelta:number|null };
export type RobustnessReport = {runId:string;version:string;scope:string;evaluationFingerprint:string;corpusSha256:string;
  base:{score:number|null;status:string;coverage:number;gainCoverage:number;comparableGroups:number};
  sourceRemoval:(SensitivityRow & {removedFamily:string})[];
  parameters:(SensitivityRow & {alpha:number;minimumCoverage:number})[];
  summary:{familiesTested:number;comparableRemovals:number;scopeChangedRemovals:number;withheldRemovals:number;newlyWithheldRemovals:number;maxAbsoluteComparableDelta:number|null};limitations:string[]};
export type DesignViewAssessment = {support:number|null;baseSupport:number|null;summary:string[];reasons:string[];ruleIds:string[];evidenceIds:string[]};
export type SearchResult = { query: string; mode: string; collection: string; evidence: Evidence[]; elapsedMs: number;
  answer: GeneratedAnswer; runId: string };
export type Run = { id: string; created: number; kind: string; candidateId?: string; candidateName?: string; query?: string; status?: string; score?: number; version: string };

export const API_URL = process.env.EXPO_PUBLIC_RAG_API_URL || 'http://127.0.0.1:8787';
export async function api<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) controller.abort();
  const timer = setTimeout(abort, 45000);
  try {
    const response = await fetch(`${API_URL}${path}`, { method: body === undefined ? 'GET' : 'POST',
      headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body), signal: controller.signal });
    const json = await response.json();
    if (!response.ok) {
      const messages: string[] = Array.isArray(json.detail) ? json.detail.map((v: { msg?: unknown }) => typeof v.msg === 'string' ? v.msg.replace(/^Value error, /, '') : '').filter((v: string) => /[가-힣]/.test(v)) : [];
      throw new Error(typeof json.detail === 'string' ? json.detail : messages.length ? [...new Set(messages)].join('\n') : '입력 정보를 확인해주세요. 보유 옷은 최대 64벌까지 분석할 수 있어요.');
    }
    return json as T;
  } catch (e) {
    if (e instanceof Error && e.name === 'AbortError') throw new Error('응답 대기 시간이 지났어요. 연결을 확인하고 다시 시도해주세요.');
    if (e instanceof TypeError) throw new Error('분석 서버에 연결할 수 없어요. 서버를 실행한 후 다시 시도해주세요.');
    throw e;
  } finally { clearTimeout(timer); signal?.removeEventListener('abort', abort); }
}
/** Photos remain on the device. Only confirmed text attributes are sent locally. */
export function attributes(g: Garment) {
  return { id: g.id, name: g.name, category: g.category, color: g.color, pattern: g.pattern, occasion: g.occasion, role: g.role, design: assessmentDesign(g), audience: g.audience || 'unknown', sourceProductId: g.sourceProductId, referenceProductId: g.referenceProductId };
}
export function assessmentInput(candidate: Garment, wardrobe: Garment[], occasion: Garment['occasion'], mode: Exclude<SearchMode, 'conditioned'> = 'hybrid') {
  return { candidate: attributes(candidate), wardrobe: wardrobe.filter(g => g.role === 'owned').map(attributes), occasion, mode };
}

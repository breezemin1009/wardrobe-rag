import { test } from 'node:test';
import assert from 'node:assert/strict';
import { initialState, catalog } from '../src/domain/demo.ts';
import { archiveGarment, duplicatePhoto, parseState, previewEvaluation, restoreGarment, saveGarment } from '../src/domain/rules.ts';
import { emptyView,validDesign } from '../src/domain/design.ts';
import { inMenswearScope, scopeReview } from '../src/domain/audience.ts';

test('legacy user photos remain preserved and require explicit audience review', () => {
  const seed = { ...initialState().garments[0], id: 'legacy-photo', imageUri: 'file:///my-shirt.jpg', isExample: false, audience: undefined };
  const state = { ...initialState(), garments: [seed], archived: [{ ...seed, id: 'archived-photo' }] };
  const migrated = parseState(JSON.stringify(state));
  assert.equal(migrated.garments[0].imageUri, seed.imageUri);
  assert.equal(migrated.garments[0].audience, 'unknown');
  assert.equal(migrated.archived[0].audience, 'unknown');
  assert.equal(scopeReview(migrated.garments).length, 1);
});

test('only unchanged built-in examples inherit their intended menswear scope', () => {
  const state = initialState();
  state.garments.forEach(g => delete g.audience);
  assert.ok(parseState(JSON.stringify(state)).garments.every(g => g.audience === 'mens'));
  state.garments[0].imageUri = 'file:///different-garment.jpg';
  assert.equal(parseState(JSON.stringify(state)).garments[0].audience, 'unknown');
});

test('missing or unknown audience is never treated as unisex', () => {
  for (const audience of [undefined, 'unknown', 'womens']) assert.equal(inMenswearScope({ audience }), false);
  for (const audience of ['mens', 'unisex']) assert.equal(inMenswearScope({ audience }), true);
  const state = initialState();
  state.garments[0].audience = 'invalid';
  assert.throws(() => parseState(JSON.stringify(state)));
});

test('same verified product cannot duplicate by changing photo resolution', () => {
  const g = { ...initialState().garments[0], sourceProductId: '1163', imageHash: 'a'.repeat(64) };
  const state = { ...initialState(), garments: [g] };
  assert.equal(duplicatePhoto(state, { ...g, id: 'copy', imageHash: 'b'.repeat(64) }).id, g.id);
  assert.throws(() => saveGarment(state, { ...g, id: 'copy', imageHash: 'b'.repeat(64) }));
});

test('an uploaded garment never receives a fixture score', () => {
  const candidate = { ...catalog[0], id: 'uploaded', isExample: false, imageUri: 'data:image/jpeg;base64,abc', demoAsset: undefined };
  const result = previewEvaluation(candidate, initialState().garments);
  assert.equal(result.status, 'unavailable');
  assert.equal(result.score, undefined);
});
test('changes to the wardrobe invalidate preview scores', () => {
  const state = initialState();
  assert.equal(previewEvaluation(catalog[0], state.garments).score, 70);
  state.garments[0].color = '블랙';
  assert.equal(previewEvaluation(catalog[0], state.garments).status, 'unavailable');
});
test('additional owned garments invalidate fixtures; bookmarking does not', () => {
  const state = initialState();
  state.savedIds.push(catalog[0].id);
  assert.equal(previewEvaluation(catalog[0], state.garments).status, 'example');
  state.garments.push({ ...state.garments[0], id: 'new-owned' });
  assert.equal(previewEvaluation(catalog[0], state.garments).status, 'unavailable');
});
test('archiving is reversible and restores the demo context', () => {
  const initial = initialState();
  const archived = archiveGarment(initial, 'owned-white');
  assert.equal(archived.garments.length, initial.garments.length - 1);
  assert.equal(previewEvaluation(catalog[0], archived.garments).status, 'unavailable');
  const restored = restoreGarment(archived, 'owned-white');
  assert.equal(previewEvaluation(catalog[0], restored.garments).status, 'example');
  assert.equal(restored.archived.length, 0);
});
test('invalid persisted data is rejected rather than overwritten', () => {
  assert.throws(() => parseState('{'));
  assert.throws(() => parseState(JSON.stringify({ version: 1, garments: [], archived: [] })));
  const duplicate = initialState();
  duplicate.garments.push({ ...duplicate.garments[0] });
  assert.throws(() => parseState(JSON.stringify(duplicate)));
  assert.deepEqual(parseState(JSON.stringify(initialState())), initialState());
});

test('same photo under another filename or role cannot duplicate a registered garment', () => {
  const original = { ...initialState().garments[0], imageHash: 'a'.repeat(64), imageUri: 'file:///photo-one.jpg' };
  const state = { ...initialState(), garments: [original] };
  const copy = { ...original, id: 'copy', role: 'candidate', imageUri: 'file:///photo-two.jpg' };
  assert.equal(duplicatePhoto(state, copy).id, original.id);
  assert.throws(() => saveGarment(state, copy), /같은 사진/);
  assert.equal(saveGarment(state, { ...original, name: 'edited' }).garments[0].name, 'edited');
  const archived = archiveGarment(state, original.id);
  assert.throws(() => saveGarment(archived, copy), /같은 사진/);
  assert.equal(restoreGarment(archived, original.id).garments.length, 1);
});

test('old wardrobe persists without a hash and invalid hashes are rejected', () => {
  const state = initialState();
  assert.deepEqual(parseState(JSON.stringify(state)), state);
  state.garments[0].imageHash = 'invalid';
  assert.throws(() => parseState(JSON.stringify(state)));
});

test('cutout storage keeps the original photo and cannot bypass duplicate detection', () => {
  const original = { ...initialState().garments[0], imageUri: 'file:///original.jpg', imageHash: 'b'.repeat(64), cutoutUri: 'file:///cutout.png' };
  const state = { ...initialState(), garments: [original] };
  const restored = parseState(JSON.stringify(state));
  assert.equal(restored.garments[0].imageUri, original.imageUri);
  assert.equal(restored.garments[0].cutoutUri, original.cutoutUri);
  assert.throws(() => saveGarment(restored, { ...original, id: 'copy', cutoutUri: 'file:///another-cutout.png' }), /같은 사진/);
  assert.throws(() => parseState(JSON.stringify({ ...state, garments: [{ ...original, cutoutUri: 42 }] })));
});

test('owned wardrobe limit prevents registering an unanalysable 65th garment', () => {
  const seed = initialState().garments[0];
  const state = { ...initialState(), garments: Array.from({ length: 64 }, (_, i) => ({ ...seed, id: `g-${i}` })) };
  assert.throws(() => saveGarment(state, { ...seed, id: 'new' }), /64/);
  assert.equal(saveGarment(state, { ...state.garments[0], name: 'edited' }).garments.length, 64);
});

test('restoring an archived garment respects the analysis limit and preserves recovery', () => {
  const seed = initialState().garments[0];
  const owned = { ...seed, id: 'archived-owned' };
  const candidate = { ...seed, id: 'archived-candidate', role: 'candidate' };
  const state = { ...initialState(), garments: Array.from({ length: 64 }, (_, i) => ({ ...seed, id: `g-${i}` })), archived: [owned, candidate] };
  assert.throws(() => restoreGarment(state, owned.id), /64/);
  assert.equal(state.archived.length, 2);
  const candidateRestored = restoreGarment(state, candidate.id);
  assert.equal(candidateRestored.garments.filter(g => g.role === 'owned').length, 64);
  const freed = archiveGarment(candidateRestored, 'g-0');
  const restored = restoreGarment(freed, owned.id);
  assert.equal(restored.garments.filter(g => g.role === 'owned').length, 64);
  assert.ok(restored.garments.some(g => g.id === owned.id));
  assert.ok(!restored.archived.some(g => g.id === owned.id));
});

test('detailed front and back designs persist without inventing an unseen back', () => {
  const state=initialState(),g={...state.garments[0],design:{version:1,garmentType:'shirt',front:emptyView('화이트'),back:null,backStatus:'unknown'}};
  const saved=saveGarment(state,g),roundTrip=parseState(JSON.stringify(saved));
  assert.equal(roundTrip.garments[0].design.back,null);
  assert.equal(roundTrip.garments[0].design.backStatus,'unknown');
  assert.equal(validDesign({...g.design,back:g.design.front}),false);
  assert.equal(validDesign({...g.design,backStatus:'observed'}),false);
});

test('mismatched basic fields or garment types cannot corrupt saved wardrobe', () => {
  const state=initialState(),g={...state.garments[0],design:{version:1,garmentType:'shirt',front:emptyView('블랙'),back:null,backStatus:'unknown'}};
  assert.throws(()=>saveGarment(state,g),/기본 정보/);
  g.design.front.baseColor='화이트';g.design.garmentType='jeans';
  assert.throws(()=>saveGarment(state,g),/기본 정보/);
  assert.equal(parseState(JSON.stringify(state)).garments.length,state.garments.length);
});

test('invalid design values and oversized decoration lists are rejected', () => {
  const d={version:1,front:emptyView(),back:null,backStatus:'unknown'};
  assert.equal(validDesign(d),true);
  assert.equal(validDesign({...d,front:{...d.front,accents:Array(7).fill({kind:'logo',size:'large',placement:'center',colors:['레드']})}}),false);
  assert.equal(validDesign({...d,front:{...d.front,accents:Array(4).fill({kind:'patch',size:'small',placement:'side',colors:['레드','그린','옐로','화이트']})}}),true);
  assert.equal(validDesign({...d,front:{...d.front,baseColor:'untrusted'}}),false);
  assert.equal(validDesign({...d,garmentType:'invented'}),false);
});

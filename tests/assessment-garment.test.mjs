import test from 'node:test';
import assert from 'node:assert/strict';
import { assessmentDesign } from '../src/domain/assessment-garment.ts';
import { exampleWardrobe } from '../src/domain/demo.ts';

test('known example garment types survive renaming without using names as model input', () => {
  for (const garment of exampleWardrobe) {
    const first=assessmentDesign(garment), second=assessmentDesign({...garment,name:'이름 변경'});
    assert.ok(first.garmentType);
    assert.deepEqual(first,second);
  }
  assert.equal(assessmentDesign(exampleWardrobe.find(g=>g.id==='owned-cream')).garmentType,'tshirt');
  assert.equal(assessmentDesign(exampleWardrobe.find(g=>g.id==='owned-denim')).garmentType,'jeans');
});

test('uploaded photos and a conflicting category never inherit a built-in type', () => {
  const garment=exampleWardrobe[0];
  assert.equal(assessmentDesign({...garment,imageUri:'data:image/jpeg;base64,abc'}),undefined);
  assert.equal(assessmentDesign({...garment,category:'하의'}),undefined);
  const confirmed={version:1,garmentType:'knit',front:{},back:null,backStatus:'unknown'};
  assert.equal(assessmentDesign({...garment,design:confirmed}),confirmed);
});

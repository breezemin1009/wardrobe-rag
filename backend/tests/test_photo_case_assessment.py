import unittest
import numpy as np
from backend.photo_case_assessment import PhotoCaseAssessment,outfit_evidence_id

class DescriptorTests(unittest.TestCase):
 def test_all_confirmed_colour_names_have_a_source_language_descriptor(self):
  from types import SimpleNamespace
  from typing import get_args
  from backend.design import Colour
  from backend.photo_case_assessment import COLOURS
  u=PhotoCaseAssessment(SimpleNamespace(engine=None))
  self.assertEqual(set(COLOURS),set(get_args(Colour))-{'기타'})
  for colour in get_args(Colour):
   if colour=='기타':continue
   g=dict(name='등록 의류',category='상의',color='기타',pattern='무지',design={'garmentType':'shirt','front':{'baseColor':colour,'pattern':'무지'}})
   kind,query=u.descriptor(g)
   self.assertEqual(kind,'shirt')
   self.assertIn(COLOURS[colour],query)
   self.assertNotIn(colour,query)

 def test_same_source_outfit_found_for_two_contexts_counts_once(self):
  r=dict(source='https://example.org/source',top={'id':'top'},bottom={'id':'bottom'},sourceOutfits=['b','a'],contextId='query-one')
  other={**r,'contextId':'query-two','sourceOutfits':['a','b','a']}
  self.assertEqual(outfit_evidence_id(r),outfit_evidence_id(other))
  self.assertNotEqual(outfit_evidence_id(r),outfit_evidence_id({**r,'bottom':{'id':'other-bottom'}}))
 def test_verified_descriptors_and_side(self):
  from types import SimpleNamespace
  u=PhotoCaseAssessment(SimpleNamespace(engine=None))
  kind,q=u.descriptor(dict(name='내 바지',category='하의',color='블랙',pattern='체크',design={'garmentType':'jeans'}))
  self.assertEqual(kind,'jeans');self.assertIn('黑色',q);self.assertIn('格子',q)
  tee=dict(name='크림 코튼 티셔츠',category='상의',color='크림',pattern='무지',design={'garmentType':'tshirt'})
  self.assertEqual(u.descriptor(tee)[0],'tee')
  self.assertEqual(u.descriptor(tee),u.descriptor({**tee,'name':'이름을 바꾼 옷'}))
  with self.assertRaises(ValueError):u.descriptor({**tee,'design':None})
  with self.assertRaises(ValueError):u.descriptor(dict(name='상의',category='상의',color='화이트',pattern='무지',design={'garmentType':'jeans'}))
 def test_gain_properties(self):
  rng=np.random.default_rng(2810)
  for _ in range(500):
   s=rng.random(8);h=rng.random(8);new=np.maximum(h,rng.random(8))
   m=np.maximum(s-h,0).mean()
   self.assertTrue(0<=m<=s.mean())
   self.assertLessEqual(np.maximum(s-new,0).mean(),m+1e-12)
   self.assertEqual(float(np.maximum(s-s,0).mean()),0)
   binary=rng.integers(0,2,8)
   np.testing.assert_allclose(np.maximum(s-binary,0),s*(1-binary))

if __name__=='__main__':unittest.main()

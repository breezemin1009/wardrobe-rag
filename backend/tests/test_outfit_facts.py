import unittest
from backend.outfit_facts import garment_facts,record_facts,validate_plan,render_plan
from backend.generation import validate_assessment_answer

class OutfitFactsTests(unittest.TestCase):
    def test_source_sentences_use_particles_for_the_actual_garment_words(self):
        make=lambda top,bottom:{'top':{'kind':top,'colors':[],'patterns':[]},'bottom':{'kind':bottom,'colors':[],'patterns':[]}}
        self.assertEqual(render_plan(make('jacket','trousers')),'참고한 유사 상품의 공개 코디에는 재킷과 바지가 함께 등장합니다.')
        self.assertEqual(render_plan(make('shirt','jeans')),'참고한 유사 상품의 공개 코디에는 셔츠와 청바지가 함께 등장합니다.')
        self.assertEqual(render_plan(make('top','bottom')),'참고한 유사 상품의 공개 코디에는 상의와 하의가 함께 등장합니다.')

    def test_unstated_patterns_are_not_invented_and_brand_is_not_a_colour(self):
        self.assertEqual(garment_facts('Off-White shirt','top'),{'kind':'shirt','colors':[],'patterns':[]})
        self.assertEqual(garment_facts('米白色纯色衬衫','top'),{'kind':'shirt','colors':['크림'],'patterns':['무지']})
        self.assertEqual(garment_facts('军绿色短裤','bottom'),{'kind':'shorts','colors':['올리브'],'patterns':[]})

    def test_crossed_source_and_fabricated_attributes_are_rejected(self):
        r={'source':'E1','sourceOutfits':['original-1'],'top':{'title':'white striped shirt'},'bottom':{'title':'black jeans'}}
        plan={'source':'E1',**record_facts(r)}
        self.assertEqual(validate_plan(plan,[r],'original-1'),plan)
        self.assertIn('화이트 스트라이프 셔츠',render_plan(plan))
        for changed in ({**plan,'source':'E2'}, {**plan,'top':{**plan['top'],'colors':['블루']}}):
            with self.assertRaises(ValueError):validate_plan(changed,[r],'original-1')
        with self.assertRaises(ValueError):validate_plan(plan,[r],'unrelated-original')

    def test_app_appends_checked_fact_without_altering_fixed_comparison(self):
        record={'source':'E1','sourceOutfits':['one'],'top':{'title':'white shirt'},'bottom':{'title':'black jeans'}}
        facts={'candidate':{'category':'하의'},'candidateLabel':'바지','publicCase':True,
               'outfitRecords':[record],'targetOutfit':'one',
               'groups':[{'key':'G1','relation':'higher','보유옷':'셔츠','citations':['E1']} ]}
        plan={'candidate':facts['candidate'],'groups':[{'key':'G1','relation':'higher','textKey':'brief','citations':['E1']}],
              'summary':{'source':'E1',**record_facts(record)}}
        answer=validate_assessment_answer(plan,facts)
        self.assertIn('추가 활용이 있는 조합',answer['text'])
        self.assertIn('화이트 셔츠와 블랙 청바지',answer['text'])
        self.assertEqual(answer['citations'],['E1'])

if __name__=='__main__':unittest.main()

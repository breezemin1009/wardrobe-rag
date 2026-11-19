import io,json,unittest
from types import SimpleNamespace
from unittest.mock import patch
from backend.generation import validate_assessment_answer
from backend.menswear_case_explanation import explain_case

class ExplanationPlanTests(unittest.TestCase):
    def test_contradictory_legacy_prose_is_never_rendered(self):
        facts={'candidate':{'category':'하의','color':'크림','pattern':'무지'},'평가범위':'색상·무늬','groups':[
            {'key':'G1','relation':'higher','보유옷':'블루 상의','citations':['E1']}]}
        for attack in ['기존 대안보다 근거가 더 적습니다.','기존 블랙 후보가 추천됩니다.','기존 대안보다 빨간 체크 바지가 좋습니다.']:
            answer={'candidate':facts['candidate'],'groups':[{'key':'G1','relation':'higher','text':attack,'citations':['E1']}]}
            with self.subTest(attack=attack):
                text=validate_assessment_answer(answer,facts)['text']
                self.assertNotIn(attack,text);self.assertIn('조합 근거가 더 있습니다',text);self.assertNotIn('빨간',text)

    def test_structured_plan_renders_confirmed_attributes(self):
        facts={'candidate':{'category':'하의','color':'크림','pattern':'무지'},'평가범위':'색상·무늬','groups':[
            {'key':'G1','relation':'covered','보유옷':'화이트 상의','citations':['E2']}]}
        answer={'candidate':facts['candidate'],'groups':[{'key':'G1','relation':'covered','textKey':'brief','citations':['E2']}]}
        out=validate_assessment_answer(answer,facts)
        self.assertIn('크림',out['text']);self.assertIn('기존 대안도 같은 수준',out['text'])

    def engine(self):
        return SimpleNamespace(items={'b':{'title':'蓝色纯色长裤'}},identified=lambda ids:ids,
            assess=lambda *args,**kwargs:dict(results=[dict(id='b',score=.3,rank=1,evidence=[dict(top={'title':'白色衬衫'},bottom={'title':'蓝色长裤'})])],evaluationFingerprint='fixed'))

    def call(self,data):
        response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(data)}}]}
        with patch('backend.menswear_case_explanation.NVIDIA_API_KEY','test'),patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
            return explain_case(self.engine(),SimpleNamespace(candidates=['b'],contexts=['q'],owned=['w']))

    def test_case_plan_uses_only_server_fact_catalog(self):
        comparison='후보 A는 기존 하의와 비교해 추가 조합 근거가 가장 크게 나타납니다.'
        out=self.call(dict(candidateLabel='A',comparison=comparison,detailKey='candidate-and-source',citations=['E1']))
        self.assertEqual(out['mode'],'nvidia');self.assertIn('블루·무지',out['text']);self.assertNotIn('빨간',out['text'])

    def test_case_wrong_colour_or_opposite_meaning_cannot_enter_plan(self):
        comparison='후보 A는 기존 하의와 비교해 추가 조합 근거가 가장 크게 나타납니다.'
        for attack in ['기존 하의보다 조합 근거가 적습니다.','빨간 체크 바지입니다.']:
            out=self.call(dict(candidateLabel='A',comparison=comparison,detail=attack,citations=['E1']))
            self.assertEqual(out['mode'],'extractive');self.assertNotIn(attack,out['text'])

    def test_weighted_winner_is_not_described_as_largest_addition(self):
        engine=SimpleNamespace(identified=lambda ids:ids,items={'a':{'title':'黑色长裤'},'b':{'title':'蓝色长裤'}},
            assess=lambda *a,**k:dict(method={'alpha':.5},results=[dict(id='a',score=60,additionalContribution=20,evidence=[]),dict(id='b',score=55,additionalContribution=40,evidence=[])],evaluationFingerprint='weighted'))
        with patch('backend.menswear_case_explanation.NVIDIA_API_KEY',''):
            out=explain_case(engine,SimpleNamespace(candidates=['a','b'],contexts=['q'],owned=['w']))
        self.assertIn('추천 지수가 가장 높',out['text']);self.assertNotIn('추가 조합 근거가 가장',out['text'])

    def test_owned_candidate_cannot_become_explanation_winner(self):
        engine=SimpleNamespace(identified=lambda ids:ids,items={'a':{'title':'黑色长裤'},'b':{'title':'蓝色长裤'}},
            assess=lambda *a,**k:dict(method={'alpha':.5},results=[dict(id='a',score=90,additionalContribution=0,alreadyOwned=True,evidence=[]),dict(id='b',score=50,additionalContribution=20,evidence=[])],evaluationFingerprint='owned'))
        with patch('backend.menswear_case_explanation.NVIDIA_API_KEY',''):
            out=explain_case(engine,SimpleNamespace(candidates=['a','b'],contexts=['q'],owned=['a']))
        self.assertIn('후보 B',out['text']);self.assertNotIn('후보 A',out['text'])

    def test_zero_estimated_increment_does_not_assert_all_real_combos_covered(self):
        engine=self.engine();engine.assess=lambda *a,**k:dict(results=[dict(id='b',score=0,additionalContribution=0,evidence=[])],evaluationFingerprint='zero')
        with patch('backend.menswear_case_explanation.NVIDIA_API_KEY',''):
            out=explain_case(engine,SimpleNamespace(candidates=['b'],contexts=['q'],owned=['w']))
        self.assertIn('늘어나는 조합 근거가 나타나지',out['text']);self.assertNotIn('이미 지원',out['text'])

if __name__=='__main__':unittest.main()

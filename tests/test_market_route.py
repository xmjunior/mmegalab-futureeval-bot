"""Only the Market Pulse transport boundary is exercised; no sockets or credentials."""
import hashlib
import os
import socket
import time
import unittest
from unittest.mock import AsyncMock, Mock, patch


class MarketRouteTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.enterClassContext(patch.object(socket.socket, 'connect', side_effect=AssertionError('network disabled')))
        import market_pulse
        cls.m = market_pulse

    def setUp(self):
        self.enterContext(patch.dict(os.environ, {'GROQ_API_KEY':'test-only', 'GROQ_FREE_ACCOUNT_CONFIRMED':'true'}, clear=True))
        self.row = {'choices':[{'width':1.25}]}

    async def test_frozen_review_prompt_and_named_groq_transport(self):
        # Exact payload hashes from the previously sealed historical review.
        hashes = {46212:'4d702a46f2615ade2b77c316feaf553c8064ae1bbe99aaebaa70a02f36fbbd8c',
                  46213:'f31a0dc3e0a526ffca3aaec5e80d879a6531e3a5c3cc35a73d19626708f5fd0c',
                  46709:'f51037ad3daec5a20f4ca66ca1106af7d86d3cec8f661f61191a6f66f2b7c648'}
        route = self.m.groq_free
        self.assertEqual(self.m.configured_routes(), [route.ROUTE])
        post = Mock(status_code=200)
        post.json.return_value = {'model':route.MODEL, 'choices':[{'message':{'content':'[1.25]'}, 'finish_reason':'stop'}],
                                  'usage':{'prompt_tokens':10, 'completion_tokens':4}}
        free = Mock(status_code=403)
        free.json.return_value = {'error':{'code':'free_tier_not_supported'}}
        with patch.object(route, '_request', side_effect=[free,post]*3) as request:
            for case in [r for r in self.m.prices.HISTORY if r['question_id'] in hashes]:
                spot = case['spot']
                est = self.m.prices.model(46010,spot-self.m.timedelta(seconds=1),spot,case['resolve'])
                row = self.m.options_for(est,[1.25],spot,case['resolve'])
                self.assertEqual(await self.m.review([route.ROUTE],[row]),(route.ROUTE,[1.25]))
                plan, completion = request.call_args_list[-2:]
                self.assertEqual(plan.args[:2],('GET','/batches?limit=1'))
                self.assertEqual(completion.args[:2],('POST','/chat/completions'))
                payload = completion.kwargs['json']
                self.assertEqual((payload['model'],payload['max_completion_tokens'],payload['tools']),(route.MODEL,512,[]))
                self.assertEqual(hashlib.sha256(payload['messages'][0]['content'].encode()).hexdigest(),hashes[case['question_id']])

    async def test_no_paid_or_ambiguous_plan_reaches_inference(self):
        route=self.m.groq_free
        for status in (200,402,403,429):
            resp=Mock(status_code=status,headers={})
            resp.json.return_value={'error':{'message':'permission denied'}}
            with self.subTest(status=status), patch.object(route,'_request',return_value=resp) as request:
                with self.assertRaises(self.m.ReviewerUnavailable):
                    await self.m.review([route.ROUTE],[self.row])
                self.assertEqual([c.args[0] for c in request.call_args_list],['GET'])
        os.environ['GROQ_FREE_ACCOUNT_CONFIRMED']='false'
        with patch.object(route,'_request') as request:
            self.assertEqual(self.m.configured_routes(),[])
            with self.assertRaises(self.m.ReviewerUnavailable):
                await self.m.review([route.ROUTE],[self.row])
            request.assert_not_called()

    async def test_whole_cohort_restarts_with_one_identity_after_outage(self):
        os.environ['GEMINI_API_KEY']='test-only'
        os.environ['OPENROUTER_API_KEY']='test-only'
        names=self.m.MODEL_NAMES['GEMINI_API_KEY']
        calls=[]
        async def reply(routes,rows):
            marker=rows[0]['i']; calls.append((tuple(routes),marker))
            if routes[0].startswith('openrouter/') or (routes[0]==names[1] and marker==2):
                raise self.m.ReviewerUnavailable()
            if routes[0].startswith('gemini/'):
                return names[1],[1.25]
            return self.m.groq_free.ROUTE,[1.25]
        with patch.object(self.m,'review',side_effect=reply):
            result=await self.m.qualify_reviewers([{'i':0},{'i':1},{'i':2}])
        groq=self.m.groq_free.ROUTE
        self.assertEqual(result[-1],(groq,[1.25]*3))
        self.assertEqual(calls[-3:],[((groq,),0),((groq,),1),((groq,),2)])
        self.assertIn(((names[1],),1),calls)
        self.assertIn(((names[1],),2),calls)

    async def test_transport_cooldowns_rotate_to_groq_after_primary_outages(self):
        gemini=self.m.MODEL_NAMES['GEMINI_API_KEY'][0]
        routers=self.m.MODEL_NAMES['OPENROUTER_API_KEY']
        groq=self.m.groq_free.ROUTE
        llm=Mock()
        attempted=[]
        async def primary(prompt):
            name=next(iter(llm.available_at))
            attempted.append(name)
            llm.available_at[name]=time.monotonic()+3600
            raise RuntimeError("Partner unavailable after backoff")
        llm.invoke=primary
        with patch.object(self.m,'FreeForecastLlm',return_value=llm), patch.object(self.m,'GeneralLlm'), \
             patch.object(self.m.groq_free.GroqFreeRoute,'complete',return_value=('[null]',{})) as complete:
            result=await self.m.review([groq,gemini,*routers],[self.row])
            self.assertEqual(attempted,[gemini,routers[0]])
            self.assertEqual(result,(groq,[None]))
            complete.assert_called_once()

    async def test_available_reviewer_abstention_and_invalid_answer_block_groq(self):
        os.environ['GEMINI_API_KEY']='test-only'
        for answers in [[(self.m.MODEL_NAMES['GEMINI_API_KEY'][0],[None])],
                        [(self.m.MODEL_NAMES['GEMINI_API_KEY'][0],[1.25])],
                        [ValueError('invalid output')],
                        [(self.m.MODEL_NAMES['GEMINI_API_KEY'][0],[None]),self.m.ReviewerUnavailable()]]:
            with self.subTest(answers=answers), patch.object(self.m,'review',side_effect=answers) as review:
                results=await self.m.qualify_reviewers([self.row]*len(answers))
                self.assertEqual(len(results),1)
                self.assertTrue(all(self.m.groq_free.ROUTE not in c.args[0] for c in review.call_args_list))

    async def test_invalid_primary_output_and_live_proof_do_not_authorize_fallback(self):
        route=self.m.groq_free
        primary=self.m.MODEL_NAMES['GEMINI_API_KEY'][0]
        llm=Mock()
        llm.invoke=AsyncMock(return_value='[true]')
        with patch.object(self.m,'FreeForecastLlm',return_value=llm), patch.object(self.m,'GeneralLlm'), patch.object(route,'_request') as transport:
            with self.assertRaises(RuntimeError):
                await self.m.review([route.ROUTE,primary],[self.row])
            transport.assert_not_called()
        os.environ['MARKET_CODE_SHA']='expected'
        with self.assertRaises(ValueError):
            self.m.checked_proof({'code_sha':'old','project':self.m.PROJECT,'dry':True,'routes':[route.ROUTE]})
        with self.assertRaises(RuntimeError):
            self.m.checked_proof({'code_sha':'expected','project':self.m.PROJECT,'dry':True,'routes':[primary]})

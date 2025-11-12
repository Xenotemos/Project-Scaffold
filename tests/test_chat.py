"""Tests for chat heuristics and endpoints."""

from __future__ import annotations

import asyncio
import json
import unittest

try:  # pragma: no cover - optional dependency
    import httpx  # type: ignore[unused-ignore]
except ModuleNotFoundError as exc:  # pragma: no cover
    httpx = None  # type: ignore[assignment]
    HTTPX_IMPORT_ERROR = exc
else:  # pragma: no cover
    HTTPX_IMPORT_ERROR = None

import main
from brain.intent_router import predict_intent
from brain.reinforcement import score_response


class StubStreamEngine:
    """Emulate a streaming local llama engine."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.stream_queue: list[tuple[list[str], dict[str, object]]] = []

    async def stream_reply(
        self,
        prompt: str,
        context: dict[str, object],
        *,
        sampling: dict[str, object] | None = None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "context": context,
                "sampling": dict(sampling or {}),
            }
        )
        if not self.stream_queue:
            raise AssertionError("stream_queue is empty")
        tokens, payload = self.stream_queue.pop(0)

        async def iterator():
            for token in tokens:
                yield {"type": "token", "text": token}
            yield {"type": "done", "text": "".join(tokens), "payload": payload}

        return iterator()

    async def generate_reply(
        self,
        prompt: str,
        context: dict[str, object],
        *,
        sampling: dict[str, object] | None = None,
    ):
        raise AssertionError("generate_reply should not be called in streaming tests")


@unittest.skipIf(httpx is None, f"httpx unavailable: {HTTPX_IMPORT_ERROR}")
class ChatEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Validate the /chat endpoint responses."""

    async def asyncSetUp(self) -> None:
        self._original_endpoint = main.LLM_ENDPOINT
        self._original_client = main.llm_client
        self._original_local_engine = main.local_llama_engine
        main.LLM_ENDPOINT = ""
        main.llm_client = None
        main.local_llama_engine = None
        self._transport = httpx.ASGITransport(app=main.app)
        self._client = httpx.AsyncClient(transport=self._transport, base_url="http://testserver")

    async def asyncTearDown(self) -> None:
        await self._client.aclose()
        await self._transport.aclose()
        main.LLM_ENDPOINT = self._original_endpoint
        main.llm_client = self._original_client
        main.local_llama_engine = self._original_local_engine

    async def test_chat_returns_reply(self) -> None:
        response = await self._client.post("/chat", json={"message": "hello there"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("reply", payload)
        self.assertIn("state", payload)
        self.assertIn("source", payload)
        self.assertEqual(payload["source"], "heuristic")
        self.assertIsInstance(payload["reply"], str)
        self.assertIn("mood", payload["state"])
        self.assertIn("intent", payload)
        self.assertIn("reinforcement", payload)
        self.assertIn("voice_guard", payload)
        self.assertIn("flagged", payload["voice_guard"])
        self.assertNotIn("llm", payload)
        self.assertNotIn("User:", payload["reply"])
        self.assertNotIn("Assistant:", payload["reply"])

    async def test_invalid_stimulus_returns_error(self) -> None:
        response = await self._client.post("/chat", json={"message": "hi", "stimulus": "unknown"})
        self.assertEqual(response.status_code, 400)

    async def test_chat_uses_llm_when_client_available(self) -> None:
        class StubClient:
            async def generate_reply(self, prompt, context):
                payload = {
                    "usage": {"total_tokens": 9},
                    "timings": {"predicted_ms": 42},
                }
                return "llm says hi", payload

        main.llm_client = StubClient()
        main.LLM_ENDPOINT = "http://fake-llm"

        response = await self._client.post("/chat", json={"message": "checking remote"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["source"], "llm")
        self.assertEqual(payload["reply"], "llm says hi")
        self.assertIn("llm", payload)
        self.assertEqual(payload["llm"]["usage"]["total_tokens"], 9)
        self.assertIn("intent", payload)
        self.assertIn("length_plan", payload)
        self.assertIn("reinforcement", payload)
        self.assertIn("voice_guard", payload)
        self.assertIn("severity", payload["voice_guard"])

    async def test_state_endpoint_reports_snapshot(self) -> None:
        response = await self._client.get("/state")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("mood", payload)
        self.assertIn("hormones", payload)
        self.assertIsInstance(payload["hormones"], dict)
        self.assertIn("timestamp", payload)

    async def test_events_endpoint_accepts_payload(self) -> None:
        event_payload = {"content": "integration test", "strength": 0.6, "stimulus": "reward"}
        response = await self._client.post("/events", json=event_payload)
        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload.get("status"), "accepted")
        self.assertIn("mood", payload)

    async def test_events_endpoint_rejects_invalid_stimulus(self) -> None:
        event_payload = {"content": "bad stimulus", "strength": 0.4, "stimulus": "unknown"}
        response = await self._client.post("/events", json=event_payload)
        self.assertEqual(response.status_code, 400)

    async def test_admin_caps_reports_limits(self) -> None:
        response = await self._client.get("/admin/caps")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("llama_completion_tokens", payload)
        self.assertIn("length_overrides", payload)
        self.assertIn("timeouts", payload)

    async def test_admin_reload_endpoint(self) -> None:
        original_configure = main._configure_clients

        async def noop_configure() -> None:
            return None

        main._configure_clients = noop_configure  # type: ignore[assignment]
        try:
            response = await self._client.post("/admin/reload")
        finally:
            main._configure_clients = original_configure  # type: ignore[assignment]
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("llm_endpoint", payload)
        self.assertIn("llama", payload)


@unittest.skipIf(httpx is None, f"httpx unavailable: {HTTPX_IMPORT_ERROR}")
class StreamingEndpointTests(unittest.IsolatedAsyncioTestCase):
    """Validate the streaming /chat/stream endpoint."""

    async def asyncSetUp(self) -> None:
        self.stub = StubStreamEngine()
        self._original_endpoint = main.LLM_ENDPOINT
        self._original_client = main.llm_client
        self._original_engine = main.local_llama_engine
        main.LLM_ENDPOINT = ""
        main.llm_client = None
        main.local_llama_engine = self.stub
        self._transport = httpx.ASGITransport(app=main.app)
        self._client = httpx.AsyncClient(transport=self._transport, base_url="http://testserver")

    async def asyncTearDown(self) -> None:
        await self._client.aclose()
        await self._transport.aclose()
        main.LLM_ENDPOINT = self._original_endpoint
        main.llm_client = self._original_client
        main.local_llama_engine = self._original_engine

    @staticmethod
    async def _collect_events(response: httpx.Response) -> list[tuple[str, dict[str, object]]]:
        events: list[tuple[str, dict[str, object]]] = []
        buffer = ""
        async for chunk in response.aiter_text():
            buffer += chunk
            while "\n\n" in buffer:
                raw, buffer = buffer.split("\n\n", 1)
                if not raw.strip():
                    continue
                event_type = "message"
                data_payload = ""
                for line in raw.split("\n"):
                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_payload += line[5:].strip()
                payload = json.loads(data_payload) if data_payload else {}
                events.append((event_type, payload))
        return events

    async def test_streaming_emits_tokens_and_complete_event(self) -> None:
        self.stub.stream_queue.append(
            (
                ["Hi", " there!"],
                {
                    "choices": [{"text": "Hi there!", "finish_reason": "stop"}],
                    "usage": {"total_tokens": 4},
                    "timings": {"predicted_ms": 123},
                },
            )
        )
        async with self._client.stream("POST", "/chat/stream", json={"message": "hello there"}) as response:
            self.assertEqual(response.status_code, 200)
            events = await self._collect_events(response)
        event_types = [etype for etype, _ in events]
        self.assertEqual(event_types[0], "init")
        self.assertIn("complete", event_types)
        self.assertGreaterEqual(event_types.count("token"), 1)
        final_payload = events[-1][1]
        self.assertEqual(final_payload["source"], "local")
        self.assertEqual(final_payload["reply"], "Hi there!")
        self.assertEqual(final_payload["length_plan"]["label"], "brief")
        self.assertEqual(self.stub.calls[0]["sampling"].get("max_tokens"), main.LENGTH_SAMPLING_OVERRIDES["brief"]["max_tokens"])

    async def test_streaming_alternates_length_overrides(self) -> None:
        self.stub.stream_queue.append(
            (
                ["Sure."],
                {"choices": [{"text": "Sure.", "finish_reason": "stop"}]},
            )
        )
        async with self._client.stream("POST", "/chat/stream", json={"message": "hi"}) as response:
            self.assertEqual(response.status_code, 200)
            await self._collect_events(response)

        self.stub.stream_queue.append(
            (
                ["Here ", "is ", "the ", "full ", "walkthrough."],
                {"choices": [{"text": "Here is the full walkthrough.", "finish_reason": "stop"}]},
            )
        )
        long_prompt = "Can you explain every step in configuring the telemetry pipeline from scratch?"
        async with self._client.stream("POST", "/chat/stream", json={"message": long_prompt}) as response:
            self.assertEqual(response.status_code, 200)
            events = await self._collect_events(response)

        self.assertGreaterEqual(len(self.stub.calls), 2)
        brief_sampling = self.stub.calls[0]["sampling"]
        detailed_sampling = self.stub.calls[1]["sampling"]
        self.assertEqual(brief_sampling.get("max_tokens"), main.LENGTH_SAMPLING_OVERRIDES["brief"]["max_tokens"])
        self.assertEqual(detailed_sampling.get("max_tokens"), main.LENGTH_SAMPLING_OVERRIDES["detailed"]["max_tokens"])
        self.assertEqual(events[-1][1]["length_plan"]["label"], "detailed")

class ChatReplyTests(unittest.IsolatedAsyncioTestCase):
    """Ensure the heuristic reply references state elements."""

    async def test_reply_mentions_user_content(self) -> None:
        context = main._build_chat_context()
        length_plan = {"label": "concise", "prompt": "Provide a clear answer in roughly two to three sentences.", "hint": "Balancing substance with brevity.", "target_range": (2, 3)}
        reply = main._compose_heuristic_reply("testing the system", context=context, intent="analytical", length_plan=length_plan)
        self.assertIn("testing the system", reply)
        self.assertNotIn("hormone", reply.lower())
        self.assertIn("i hear", reply.lower())


class SamplingParamTests(unittest.TestCase):
    """Validate sampling adjustments derived from hormone state."""

    def setUp(self) -> None:
        self.baseline = {
            "dopamine": 50.0,
            "serotonin": 50.0,
            "cortisol": 30.0,
            "oxytocin": 40.0,
            "noradrenaline": 45.0,
        }

    def test_dopamine_surging_increases_temperature(self) -> None:
        hormones = dict(self.baseline)
        hormones["dopamine"] = 90.0  # triggers surging status
        params, _ = main._sampling_params_from_hormones(hormones)
        self.assertGreater(params["temperature"], main.BASE_TEMPERATURE)
        self.assertGreater(params["top_p"], main.BASE_TOP_P)

    def test_low_noradrenaline_reduces_frequency_penalty(self) -> None:
        hormones = dict(self.baseline)
        hormones["noradrenaline"] = 20.0  # triggers crashing status
        params, _ = main._sampling_params_from_hormones(hormones)
        self.assertLess(params["frequency_penalty"], main.BASE_FREQUENCY_PENALTY)

    def test_intent_sampling_override_narrative(self) -> None:
        sampling, _ = main._sampling_params_from_hormones(self.baseline)
        adjusted = main._apply_intent_sampling(sampling, "narrative")
        self.assertGreater(adjusted["temperature"], sampling["temperature"])
        self.assertGreater(adjusted["top_p"], sampling["top_p"])




class ResponseLengthPlannerTests(unittest.TestCase):
    """Ensure the length planner selects sensible profiles."""

    def test_greeting_maps_to_brief(self) -> None:
        plan = main._plan_response_length("hello there", "emotional")
        self.assertEqual(plan["label"], "brief")

    def test_long_request_maps_to_detailed(self) -> None:
        plan = main._plan_response_length("Can you explain all the steps involved in configuring the monitoring pipeline?", "analytical")
        self.assertEqual(plan["label"], "detailed")

    def test_default_is_concise(self) -> None:
        plan = main._plan_response_length("What is the status?", "analytical")
        self.assertEqual(plan["label"], "concise")

class IntentRouterTests(unittest.TestCase):
    """Ensure heuristics classify basic intent types."""

    def test_emotional_detection(self) -> None:
        prediction = predict_intent("I feel overwhelmed and sad")
        self.assertEqual(prediction.intent, "emotional")

    def test_analytical_question_detection(self) -> None:
        prediction = predict_intent("Why is the sky blue?")
        self.assertEqual(prediction.intent, "analytical")



class ReinforcementHeuristicsTests(unittest.TestCase):
    """Score response heuristics should behave consistently."""

    def test_score_response_valence_increase(self) -> None:
        scores = score_response("I am sad", "I am hopeful and calm")
        self.assertGreater(scores["valence_delta"], 0)

    def test_score_response_length_ratio(self) -> None:
        scores = score_response("short question?", "Providing a more detailed answer with several points.")
        self.assertGreater(scores["length_score"], 1.0)

    def test_score_response_engagement_entropy_bounds(self) -> None:
        scores = score_response("Tell me", "word word word word")
        self.assertGreaterEqual(scores["engagement_score"], 0.0)
        self.assertLessEqual(scores["engagement_score"], 1.0)
if __name__ == "__main__":
    unittest.main()



import asyncio
import unittest
from typing import Mapping, Sequence

from brain.engine_router import EngineRouter, RoutingContext
from brain.engines import Engine, EngineProfile


class DummyEngine:
    def __init__(self, name: str, size: str, roles: Sequence[str]) -> None:
        self.profile = EngineProfile(
            name=name,
            model_size=size,
            parameters_billion=float(size.rstrip("B")) if size.endswith("B") else 0.0,
            roles=tuple(roles),
        )

    async def generate(self, *, messages: Sequence[Mapping[str, str]], settings: Mapping[str, object] | None = None) -> Mapping[str, object]:
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


class EngineRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = EngineRouter()
        self.router.register(DummyEngine(name="idle-engine", size="2B", roles=("idle",)))
        self.router.register(DummyEngine(name="chat-engine", size="13B", roles=("chat", "linguistic")))
        self.router.register(DummyEngine(name="reasoning-engine", size="24B", roles=("reasoning", "chat")))

    def test_route_defaults_to_first_registered(self) -> None:
        decision = self.router.route(RoutingContext(task="fallback"))
        self.assertEqual(decision.engine_name, "idle-engine")

    def test_route_filters_by_role(self) -> None:
        decision = self.router.route(RoutingContext(task="conversation", required_roles=("chat",)))
        self.assertIn(decision.engine_name, {"chat-engine", "reasoning-engine"})

    def test_route_prefers_size_when_available(self) -> None:
        decision = self.router.route(
            RoutingContext(task="deep_reasoning", required_roles=("reasoning",), preferred_size="24B")
        )
        self.assertEqual(decision.engine_name, "reasoning-engine")

    def test_registered_profiles(self) -> None:
        profiles = {profile.name: profile for profile in self.router.registered_profiles()}
        self.assertIn("chat-engine", profiles)
        self.assertTrue(profiles["chat-engine"].supports_role("linguistic"))

    def test_generate_signature_matches_protocol(self) -> None:
        engine = DummyEngine(name="proto", size="7B", roles=("chat",))
        result = asyncio.run(engine.generate(messages=[{"role": "user", "content": "Hello"}], settings=None))
        self.assertIn("choices", result)


if __name__ == "__main__":
    unittest.main()

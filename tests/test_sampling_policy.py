import unittest

from brain.policy import SamplingPolicy, derive_policy
from state_engine.affect import TraitSnapshot


class SamplingPolicyTests(unittest.TestCase):
    def test_policy_adjusts_with_traits(self) -> None:
        traits = TraitSnapshot(steadiness=0.5, curiosity=0.7, warmth=0.1, tension=-0.2)
        base = SamplingPolicy(
            temperature=0.7,
            top_p=0.9,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            max_tokens=500,
        )
        policy = derive_policy(traits, base=base)
        self.assertGreater(policy.temperature, base.temperature)
        self.assertGreater(policy.top_p, base.top_p)
        self.assertGreater(policy.presence_penalty, base.presence_penalty)
        self.assertLess(policy.frequency_penalty, base.frequency_penalty)
        self.assertGreater(policy.max_tokens, base.max_tokens)

    def test_policy_clamps_ranges(self) -> None:
        traits = TraitSnapshot(steadiness=-1.0, curiosity=-1.0, warmth=-1.0, tension=1.0)
        policy = derive_policy(traits)
        self.assertGreaterEqual(policy.temperature, 0.4)
        self.assertLessEqual(policy.top_p, 0.98)
        self.assertLessEqual(policy.max_tokens, 1024)

    def test_policy_kwargs(self) -> None:
        policy = SamplingPolicy(temperature=0.66, top_p=0.88, presence_penalty=0.05, frequency_penalty=0.1, max_tokens=400)
        kwargs = policy.as_kwargs()
        self.assertEqual(kwargs["temperature"], 0.66)
        self.assertEqual(kwargs["max_tokens"], 400)


if __name__ == "__main__":
    unittest.main()

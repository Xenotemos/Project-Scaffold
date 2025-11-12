import unittest

from utils.redactor import contains_markers, sanitize_blocks, strip_internal_markers, strip_markers_from_messages


class RedactorTests(unittest.TestCase):
    def test_strip_internal_markers(self) -> None:
        text = "Hello <<HORMONE:dopamine=70>>world [[DEBUG:state]]!"
        cleaned = strip_internal_markers(text)
        self.assertEqual(cleaned, "Hello world !")
        self.assertFalse(contains_markers(cleaned))

    def test_strip_messages(self) -> None:
        messages = [
            "Line one <<TRAIT:steady=0.4>>",
            "Line two",
        ]
        sanitized = strip_markers_from_messages(messages)
        self.assertEqual(sanitized[0], "Line one")
        self.assertEqual(sanitized[1], "Line two")

    def test_sanitize_blocks_drops_empty(self) -> None:
        blocks = ("<<AFFECT:curious=0.6>>", "  meaningful text  ")
        sanitized = sanitize_blocks(blocks)
        self.assertEqual(sanitized, ("meaningful text",))


if __name__ == "__main__":
    unittest.main()

"""Malformed indexes must be preserved before a new capture index is written."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from capture import CaptureStore


class CaptureRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.index = self.root / "captures.json"

    def assert_recovered(self, original):
        self.index.write_bytes(original)
        store = CaptureStore(self.root)
        self.assertEqual(store.records, [])
        self.assertEqual(store.pending(), [])
        store._write_index()
        backups = list(self.root.glob('captures.corrupt-*.json'))
        self.assertTrue(any(path.read_bytes() == original for path in backups),
                        "original bytes were lost instead of quarantined")
        self.assertEqual(json.loads(self.index.read_text()), [])

    def test_invalid_json_is_preserved(self):
        self.assert_recovered(b'[{"email": "person@example.test"')

    def test_non_array_json_is_preserved(self):
        self.assert_recovered(b'{"email": "person@example.test"}')

    def test_non_object_records_are_quarantined_before_use(self):
        self.assert_recovered(b'[{"email": "person@example.test"}, null]')

    def test_invalid_utf8_is_preserved(self):
        self.assert_recovered(b'[{"email": "person@example.test", "text": "\xff"}]')

    def test_repeated_recovery_in_one_second_preserves_both_backups(self):
        originals = [b'{"first":', b'{"second":']
        with patch('capture.time.time', return_value=12345):
            for original in originals:
                self.index.write_bytes(original)
                CaptureStore(self.root)
        backups = [path.read_bytes() for path in self.root.glob('captures.corrupt-*.json')]
        self.assertCountEqual(backups, originals)

    def test_valid_legacy_records_still_upgrade(self):
        self.index.write_text(json.dumps([{"email": "person@example.test", "emailed": True}]))
        store = CaptureStore(self.root)
        self.assertEqual(store.sent_count, 1)
        self.assertEqual(store.pending(), [])
        self.assertTrue(store.records[0]["id"])
        self.assertEqual(list(self.root.glob('captures.corrupt-*.json')), [])


if __name__ == '__main__':
    unittest.main()

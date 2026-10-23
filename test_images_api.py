"""Reject invalid model dimensions before GPU dispatch and preserve OpenAI errors."""

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from images_api import ImagesMeta, create_images_app


class ImageSizeValidationTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {"API_KEY": "test-key"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.moss = AsyncMock(return_value=["test-image-base64"])
        self.other = AsyncMock(return_value=["other-image-base64"])
        self.client = TestClient(create_images_app(
            models={"moss": self.moss, "other": self.other},
            metas={
                "moss": ImagesMeta("moss", "test", size_multiple=16),
                "other": ImagesMeta("other", "test"),
            },
        ))
        self.headers = {"Authorization": "Bearer test-key"}

    def test_invalid_dimensions_return_400_without_gpu_call(self):
        for size in ["941x1672", "944x1672", "0x1024", "-16x1024"]:
            with self.subTest(size=size):
                result = self.client.post("/v1/images/generations", headers=self.headers,
                    json={"model": "moss", "prompt": "test", "size": size})
                self.assertEqual(result.status_code, 400)
                error = result.json()["error"]
                self.assertEqual(error["type"], "invalid_request_error")
                self.assertEqual(error["param"], "size")
                self.assertIn("positive multiples of 16", error["message"])
        self.moss.assert_not_awaited()

    def test_aligned_dimensions_are_dispatched(self):
        result = self.client.post("/v1/images/generations", headers=self.headers,
            json={"model": "moss", "prompt": "test", "size": "944x1680"})
        self.assertEqual(result.status_code, 200)
        self.moss.assert_awaited_once_with("test", 944, 1680)

    def test_other_models_keep_their_own_validation(self):
        result = self.client.post("/v1/images/generations", headers=self.headers,
            json={"model": "other", "prompt": "test", "size": "941x1672"})
        self.assertEqual(result.status_code, 200)
        self.other.assert_awaited_once_with("test", 941, 1672)

    def test_authentication_still_returns_standard_error(self):
        result = self.client.post("/v1/images/generations",
            json={"model": "moss", "prompt": "test", "size": "944x1680"})
        self.assertEqual(result.status_code, 401)
        self.assertEqual(result.json()["error"]["type"], "authentication_error")
        self.moss.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()

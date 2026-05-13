"""Tests for the AnswerLikesNullFixHook (issue #47)."""

import json
from unittest.mock import Mock

import httpx
import pytest

from src.glean.api_client._hooks.answer_likes_null_fix_hook import (
    AnswerLikesNullFixHook,
)
from src.glean.api_client._hooks.types import AfterSuccessContext, HookContext


def _make_context() -> AfterSuccessContext:
    config = Mock()
    hook_ctx = Mock(spec=HookContext)
    hook_ctx.config = config
    hook_ctx.base_url = "https://example.com"
    hook_ctx.operation_id = "chat"
    hook_ctx.oauth2_scopes = None
    hook_ctx.security_source = None
    return AfterSuccessContext(hook_ctx)


def _json_response(body: str, content_type: str = "application/json") -> httpx.Response:
    return httpx.Response(
        status_code=200,
        headers={"content-type": content_type},
        content=body.encode("utf-8"),
        request=httpx.Request("POST", "https://example.com/rest/api/v1/chat"),
    )


class TestAnswerLikesNullFixHook:
    """Behavioural tests for AnswerLikesNullFixHook."""

    def setup_method(self):
        self.hook = AnswerLikesNullFixHook()
        self.ctx = _make_context()

    def _run(self, response: httpx.Response) -> httpx.Response:
        """Invoke the hook and narrow the union return to a Response."""
        result = self.hook.after_success(self.ctx, response)
        assert isinstance(result, httpx.Response)
        return result

    def test_replaces_liked_by_null_with_empty_list(self):
        """The canonical scenario from issue #47 is rewritten."""
        body = json.dumps({"likedBy": None, "likedByUser": False, "numLikes": 0})
        result = self._run(_json_response(body))

        decoded = json.loads(result.text)
        assert decoded["likedBy"] == []
        assert decoded["likedByUser"] is False
        assert decoded["numLikes"] == 0

    def test_response_status_and_headers_preserved(self):
        """Rewriting the body preserves the response status and content-type."""
        body = json.dumps({"likedBy": None, "likedByUser": False, "numLikes": 0})
        original = _json_response(body, content_type="application/json; charset=utf-8")
        result = self._run(original)

        assert result.status_code == 200
        assert result.headers.get("content-type") == "application/json; charset=utf-8"

    def test_request_preserved_on_rewritten_response(self):
        """The original request is carried over so downstream logging works."""
        body = json.dumps({"likedBy": None, "likedByUser": False, "numLikes": 0})
        original = _json_response(body)
        result = self._run(original)

        # Should not raise; rebuilt response keeps the request reference.
        assert result.request is original.request

    def test_leaves_populated_liked_by_alone(self):
        """A non-null likedBy is left untouched."""
        body = json.dumps(
            {
                "likedBy": [{"user": {"name": "Alice"}, "createTime": "2026-01-01"}],
                "likedByUser": True,
                "numLikes": 1,
            }
        )
        result = self._run(_json_response(body))

        # When nothing changes, the hook returns the original response object
        # (cheap path) so it must still be readable.
        decoded = json.loads(result.text)
        assert decoded["likedBy"] == [
            {"user": {"name": "Alice"}, "createTime": "2026-01-01"}
        ]

    def test_returns_same_response_when_no_rewrite_needed(self):
        """No rewrite -> same response object (avoid copy cost)."""
        body = json.dumps({"likedBy": [], "likedByUser": False, "numLikes": 0})
        original = _json_response(body)
        result = self.hook.after_success(self.ctx, original)

        assert result is original

    def test_returns_same_response_when_liked_by_absent(self):
        """No `likedBy` key in the body -> response untouched."""
        body = json.dumps({"someOther": "value"})
        original = _json_response(body)
        result = self.hook.after_success(self.ctx, original)

        assert result is original

    def test_handles_whitespace_between_key_and_null(self):
        """Match `"likedBy" : null` with surrounding whitespace."""
        body = '{"likedBy" :  null ,"likedByUser":false,"numLikes":0}'
        result = self._run(_json_response(body))

        decoded = json.loads(result.text)
        assert decoded["likedBy"] == []

    def test_handles_multiple_occurrences(self):
        """Rewrites every occurrence in the body (e.g. multiple answers)."""
        body = json.dumps(
            {
                "messages": [
                    {
                        "likes": {
                            "likedBy": None,
                            "likedByUser": False,
                            "numLikes": 0,
                        }
                    },
                    {
                        "likes": {
                            "likedBy": None,
                            "likedByUser": False,
                            "numLikes": 0,
                        }
                    },
                ]
            }
        )
        result = self._run(_json_response(body))

        decoded = json.loads(result.text)
        assert decoded["messages"][0]["likes"]["likedBy"] == []
        assert decoded["messages"][1]["likes"]["likedBy"] == []

    def test_skips_non_json_content_type(self):
        """Streaming or non-JSON responses must be returned untouched."""
        body = '{"likedBy":null}'  # would match but content-type forbids it
        original = httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=body.encode("utf-8"),
            request=httpx.Request("POST", "https://example.com/chat/stream"),
        )

        result = self.hook.after_success(self.ctx, original)

        assert result is original

    def test_skips_response_without_content_type(self):
        """No content-type header -> assume non-JSON, leave alone."""
        original = httpx.Response(
            status_code=204,
            headers={},
            content=b"",
            request=httpx.Request("DELETE", "https://example.com/resource"),
        )

        result = self.hook.after_success(self.ctx, original)

        assert result is original

    def test_does_not_rewrite_liked_by_inside_user_name(self):
        """Substring 'likedBy' inside other tokens must not be rewritten."""
        body = json.dumps({"notLikedBy": None, "likedByUser": False, "numLikes": 0})
        original = _json_response(body)
        result = self.hook.after_success(self.ctx, original)

        # The key 'notLikedBy' contains 'likedBy' but the regex requires the
        # literal "likedBy" token, so this stays unchanged.
        assert result is original

    def test_handles_empty_body(self):
        """An empty JSON body is returned untouched."""
        original = _json_response("")
        result = self.hook.after_success(self.ctx, original)

        assert result is original

    def test_sdk_can_unmarshal_rewritten_body_into_answer_likes(self):
        """End-to-end: rewritten response unmarshals into AnswerLikes cleanly.

        This is the regression case from issue #47: before the fix, the SDK
        raised a ValidationError when likedBy was null.
        """
        from src.glean.api_client.models.answerlikes import AnswerLikes
        from src.glean.api_client.utils.serializers import unmarshal_json

        body = json.dumps({"likedBy": None, "likedByUser": False, "numLikes": 0})
        result = self._run(_json_response(body))

        obj = unmarshal_json(result.text, AnswerLikes)
        assert obj.liked_by == []
        assert obj.liked_by_user is False
        assert obj.num_likes == 0

    def test_old_response_without_hook_raises(self):
        """Sanity check: without the hook, the SDK fails on `likedBy: null`.

        Demonstrates the bug the hook is fixing. Kept as a guardrail so the
        hook is not silently removed in the future.
        """
        from pydantic import ValidationError

        from src.glean.api_client.models.answerlikes import AnswerLikes
        from src.glean.api_client.utils.serializers import unmarshal_json

        raw = json.dumps({"likedBy": None, "likedByUser": False, "numLikes": 0})
        with pytest.raises(ValidationError):
            unmarshal_json(raw, AnswerLikes)

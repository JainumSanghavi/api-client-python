"""Hook to coerce ``likedBy: null`` to ``likedBy: []`` in JSON responses."""

import re
from typing import Union

import httpx

from .types import AfterSuccessContext, AfterSuccessHook


# Matches ``"likedBy" : null`` (with arbitrary intra-token whitespace) when it
# appears as a JSON key/value. Glean does not embed the literal token
# ``"likedBy":null`` inside string values, so a textual rewrite is safe.
_LIKED_BY_NULL_RE = re.compile(r'"likedBy"\s*:\s*null')


class AnswerLikesNullFixHook(AfterSuccessHook):
    """Replace ``"likedBy": null`` with ``"likedBy": []`` in JSON responses.

    The Glean API returns ``"likedBy": null`` for answers with zero likes, but
    the generated ``AnswerLikes`` model declares ``liked_by`` as a required
    ``List[AnswerLike]``. The mismatch surfaces as a Pydantic
    ``ValidationError`` whenever a chat response contains an answer that has
    not been liked.

    Reported in https://github.com/gleanwork/api-client-python/issues/47.
    Until the upstream OpenAPI specification marks ``likedBy`` as nullable,
    this hook rewrites the response body before the SDK unmarshals it.
    """

    def after_success(
        self, hook_ctx: AfterSuccessContext, response: httpx.Response
    ) -> Union[httpx.Response, Exception]:
        # Only touch JSON responses. Streaming bodies (text/event-stream from
        # chat.create_stream, file downloads, etc.) must be left untouched so
        # the SDK can read them lazily.
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type.lower():
            return response

        try:
            text = response.text
        except (httpx.ResponseNotRead, httpx.RequestNotRead, RuntimeError):
            return response

        if '"likedBy"' not in text:
            return response

        new_text = _LIKED_BY_NULL_RE.sub('"likedBy":[]', text)
        if new_text == text:
            return response

        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=new_text.encode("utf-8"),
            request=response.request,
        )

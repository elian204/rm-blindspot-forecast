"""Extract the final answer from a response and check it against the ground truth.

Rules (kept deliberately simple and deterministic):

* If the response contains ``\\boxed{``, the LAST non-empty box decides ("last box
  wins"). If the last box never closes (typically truncated output) the response
  has no usable answer: status ``unparseable``. Empty boxes are ignored.
* Otherwise the last "The answer is ..." / "Final answer is ..." phrase decides.
* Equivalence is decided by ``math-verify`` (``parse`` + ``verify``).
* A missing, unparseable or uncheckable answer counts as wrong, with its own
  status, so such cases can be counted separately and are never silently dropped.

Note: math-verify enforces timeouts with signals, which only work in the main
thread. Call the checker from the main thread (or from worker processes).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from math_verify import parse, verify
from math_verify.errors import TimeoutException


class AnswerStatus(StrEnum):
    """Outcome of checking one response."""

    CORRECT = "correct"
    WRONG = "wrong"
    NO_ANSWER = "no_answer"  # no box, no "the answer is", or only empty boxes
    UNPARSEABLE = "unparseable"  # an answer was found but could not be parsed
    CHECK_ERROR = "check_error"  # math-verify timed out or failed while comparing


@dataclass(frozen=True)
class CheckResult:
    """Result of checking one response against its ground truth."""

    status: AnswerStatus
    extracted: str | None  # the answer text that was graded, if any
    n_boxed: int  # number of complete, non-empty \boxed{...} in the response
    # Several boxed answers that are not equivalent to the last one. Correctness
    # still follows the last box; this flag only marks the ambiguity.
    multiple_distinct_answers: bool
    truncated: bool  # generation stopped because of the token limit

    @property
    def correct(self) -> bool:
        """Only the 'correct' status counts as correct."""
        return self.status is AnswerStatus.CORRECT


# --------------------------------------------------------------------------
# Answer extraction
# --------------------------------------------------------------------------

_BOXED_START = re.compile(r"\\boxed\s*\{")
# "answer is" as whole words: "answer isn't 5" must not match.
_ANSWER_IS = re.compile(r"(?:final\s+)?\banswer\s+is(?![\w'’])\s*:?\s*", re.IGNORECASE)


def find_boxed(text: str) -> list[str | None]:
    """Contents of every top-level ``\\boxed{...}``, in order.

    Braces are matched with nesting, and escaped braces (``\\{``) are skipped.
    An entry is ``None`` if the box never closes. Boxes nested inside another
    box are part of the outer box's content, not separate entries.
    """
    found: list[str | None] = []
    pos = 0
    while True:
        m = _BOXED_START.search(text, pos)
        if m is None:
            return found
        depth, i = 1, m.end()
        while i < len(text) and depth > 0:
            char = text[i]
            if char == "\\":  # skip the escaped character, e.g. \{ or \}
                i += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            i += 1
        if depth > 0:
            found.append(None)
            return found
        found.append(text[m.end() : i - 1])
        pos = i


def _clean_phrase_answer(raw: str) -> str:
    """Trim a 'The answer is ...' tail down to the answer itself."""
    line = raw.strip().splitlines()[0] if raw.strip() else ""
    # Cut at the end of the sentence; a period inside "18.5" is not followed by a space.
    line = re.split(r"\.(?:\s|$)", line, maxsplit=1)[0]
    line = line.strip().strip("*").strip()
    for left, right in (("$$", "$$"), ("$", "$"), ("\\(", "\\)"), ("\\[", "\\]")):
        if line.startswith(left) and line.endswith(right) and len(line) >= len(left) + len(right):
            line = line[len(left) : len(line) - len(right)]
            break
    return line.strip().rstrip(".").strip()


def extract_answer(response: str) -> tuple[str | None, bool, int, list[str]]:
    """Find the final answer text.

    Returns ``(answer, unclosed_box, n_boxed, boxed_contents)`` where ``answer``
    is None when no usable answer exists, ``unclosed_box`` says the deciding box
    never closed, and ``boxed_contents`` are the complete non-empty boxes.
    """
    boxes = find_boxed(response)
    complete = [b.strip() for b in boxes if b is not None and b.strip()]
    if boxes:
        if boxes[-1] is None:  # the last box never closes
            return None, True, len(complete), complete
        return (complete[-1] if complete else None), False, len(complete), complete
    matches = list(_ANSWER_IS.finditer(response))
    if matches:
        answer = _clean_phrase_answer(response[matches[-1].end() :])
        return (answer or None), False, 0, []
    return None, False, 0, []


# --------------------------------------------------------------------------
# Answer normalisation and comparison
# --------------------------------------------------------------------------

_NUMBER_WORDS = re.compile(r"^(-?[\d,]*\.?\d+)\s+([A-Za-z]{3,}(?:\s+[A-Za-z]+)*)\.?$")
_MATH_WORDS = {"sqrt", "sin", "cos", "tan", "log", "exp", "cdot", "times", "text", "frac"}
_TEXT_WRAPPER = re.compile(r"^\\text\{\s*(.*?)\s*\}$", re.DOTALL)


def strip_trailing_units(answer: str) -> str:
    """Drop plain-word units after a number: ``18 dollars`` -> ``18``.

    math-verify strips only some units, so "18 dollars" would otherwise be read
    as a product of letters. Only a leading plain number followed by letters-only
    words is touched; anything with math in it is left to math-verify.
    """
    text = answer.strip()
    wrapped = _TEXT_WRAPPER.match(text)
    if wrapped:
        text = wrapped.group(1)
    m = _NUMBER_WORDS.match(text)
    if m and not (set(m.group(2).lower().split()) & _MATH_WORDS):
        return m.group(1)
    return answer


def _parse_math(text: str) -> list:
    """Parse into sympy objects; empty list when math-verify finds nothing.

    Raises math-verify's own errors (including ``TimeoutException``, which is a
    ``BaseException``) instead of hiding them; callers decide what they mean.
    """
    return parse(f"${text}$", fallback_mode="no_fallback", raise_on_error=True)


def parse_ground_truth(ground_truth: str) -> list:
    """Parse a ground-truth answer. A bad ground truth is a data error, not a wrong answer."""
    try:
        parsed = _parse_math(ground_truth)
    except (Exception, TimeoutException) as error:
        raise ValueError(f"Cannot parse ground truth {ground_truth!r}: {error!r}") from error
    if not parsed:
        raise ValueError(f"Cannot parse ground truth: {ground_truth!r}")
    return parsed


def _equivalent(a: str, b: str) -> bool:
    """Are two answer strings the same math? Falls back to whitespace-free text equality.

    ``verify`` is not symmetric (it treats its first argument as the reference),
    so both directions are tried. Any math-verify failure counts as "different".
    """
    try:
        pa, pb = _parse_math(strip_trailing_units(a)), _parse_math(strip_trailing_units(b))
        if pa and pb:
            return bool(verify(pa, pb, raise_on_error=True) or verify(pb, pa, raise_on_error=True))
    except (Exception, TimeoutException):
        pass
    return re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b)


def check_response(
    response: str, ground_truth: str, finish_reason: str | None = None
) -> CheckResult:
    """Extract the final answer from ``response`` and grade it against ``ground_truth``.

    ``finish_reason`` comes from the sampler ("stop" or "length"); "length" marks
    the response as truncated.
    """
    truncated = finish_reason == "length"
    answer, unclosed, n_boxed, boxes = extract_answer(response)

    multiple = False
    if answer is not None and len(boxes) > 1:
        multiple = any(not _equivalent(box, answer) for box in boxes[:-1])

    if answer is None:
        # An unclosed box means an answer was started but is unusable.
        status = AnswerStatus.UNPARSEABLE if unclosed else AnswerStatus.NO_ANSWER
        return CheckResult(status, None, n_boxed, multiple, truncated)

    gold = parse_ground_truth(ground_truth)
    try:
        predicted = _parse_math(strip_trailing_units(answer))
        if not predicted:
            return CheckResult(AnswerStatus.UNPARSEABLE, answer, n_boxed, multiple, truncated)
        is_correct = bool(verify(gold, predicted, raise_on_error=True))
    except (Exception, TimeoutException):
        # math-verify failed or timed out on this answer: wrong, but reported as such.
        return CheckResult(AnswerStatus.CHECK_ERROR, answer, n_boxed, multiple, truncated)
    status = AnswerStatus.CORRECT if is_correct else AnswerStatus.WRONG
    return CheckResult(status, answer, n_boxed, multiple, truncated)

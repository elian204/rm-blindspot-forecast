"""Hand-written cases for the answer checker, including the tricky ones."""

from __future__ import annotations

import pytest

from blindspots.checker import (
    AnswerStatus,
    check_response,
    extract_answer,
    find_boxed,
    parse_ground_truth,
    strip_trailing_units,
)

C, W, N, U = (
    AnswerStatus.CORRECT,
    AnswerStatus.WRONG,
    AnswerStatus.NO_ANSWER,
    AnswerStatus.UNPARSEABLE,
)

# (id, response, ground truth, expected status)
STATUS_CASES = [
    # --- basic formats ---
    ("boxed_correct", r"So the total is \boxed{18}.", "18", C),
    ("boxed_wrong", r"So the total is \boxed{17}.", "18", W),
    ("answer_is_plain", "The answer is 18.", "18", C),
    ("answer_is_colon_dollar", "The answer is: $18$", "18", C),
    ("final_answer_is", "Final answer is 18.", "18", C),
    ("answer_is_bold", "The answer is **18**.", "18", C),
    ("answer_is_then_chatter", "The answer is 18. Hope this helps!", "18", C),
    ("answer_is_decimal", "The answer is 18.5.", "18.5", C),
    ("answer_is_thousands", "The answer is 1,000.", "1000", C),
    ("answer_is_last_wins", "The answer is 17. Wait, recompute. The answer is 18.", "18", C),
    ("boxed_beats_phrase", r"The answer is 17 at first, but \boxed{18}.", "18", C),
    ("boxed_with_space", r"\boxed {18}", "18", C),
    ("boxed_nested_in_boxed", r"\boxed{\boxed{18}}", "18", C),
    # --- fractions and numbers ---
    ("frac_dfrac", r"\boxed{\dfrac{1}{2}}", r"\frac{1}{2}", C),
    ("frac_vs_decimal", r"\boxed{\frac12}", "0.5", C),
    ("frac_unreduced", r"\boxed{\frac{6}{8}}", r"\frac{3}{4}", C),
    ("frac_wrong", r"\boxed{\frac{3}{5}}", r"\frac{3}{4}", W),
    ("mixed_number", r"\boxed{2\frac{1}{2}}", r"\frac{5}{2}", C),
    ("decimal_trailing_zeros", r"\boxed{3.140}", "3.14", C),
    ("decimal_vs_int", r"\boxed{18.0}", "18", C),
    ("decimal_wrong", r"\boxed{18.5}", "18", W),
    ("negative_vs_positive", r"\boxed{3}", "-3", W),
    ("thousands_separator", r"\boxed{1,000}", "1000", C),
    ("short_decimal_for_third", r"\boxed{0.333}", r"\frac{1}{3}", W),
    # --- units and symbols ---
    ("unit_text_macro", r"\boxed{18\text{ dollars}}", "18", C),
    ("unit_plain_words", r"\boxed{18 dollars}", "18", C),
    ("unit_inside_text", r"\boxed{\text{18 dollars}}", "18", C),
    ("unit_wrong_number", r"\boxed{17 dollars}", "18", W),
    ("unit_short_cm", r"\boxed{5 cm}", "5", C),
    ("unit_hours_text_macro", r"\boxed{2.5 \text{ hours}}", "2.5", C),
    ("unit_after_fraction", r"\boxed{\frac{1}{2} \text{ cup}}", r"\frac{1}{2}", C),
    ("percent_equals_decimal", r"\boxed{50\%}", "0.5", C),
    ("number_word_is_not_a_number", r"\boxed{eighteen}", "18", U),
    ("time_notation_is_not_the_number", r"\boxed{1:00}", "1", W),
    ("dollar_sign", r"\boxed{\$18}", "18", C),
    ("percent", r"\boxed{50\%}", "50", C),
    ("degrees", r"\boxed{90^\circ}", "90", C),
    ("word_only_answer", r"\boxed{apples}", "18", W),
    # --- LaTeX expressions ---
    ("sqrt_equivalent", r"\boxed{\sqrt{12}}", r"2\sqrt{3}", C),
    ("expression_reordered", r"\boxed{1+x^2}", "x^2+1", C),
    ("tuple_wrong_order", r"\boxed{(2,1)}", "(1,2)", W),
    ("interval", r"\boxed{(-\infty,3]}", r"(-\infty, 3]", C),
    ("set_braces_escaped", r"\boxed{\{2,1\}}", r"\{1,2\}", C),
    ("pi_multiple", r"\boxed{3\pi}", r"3\pi", C),
    ("equation_form", r"\boxed{x = 18}", "18", C),
    ("text_answer", r"\boxed{\text{Evelyn}}", r"\text{Evelyn}", C),
    ("text_answer_wrong", r"\boxed{\text{Dana}}", r"\text{Evelyn}", W),
    ("several_values_not_one", r"\boxed{5, 6}", "5", W),
    # --- missing or unusable answers ---
    ("no_answer_prose", "I think it is about eighteen.", "18", N),
    ("empty_response", "", "18", N),
    ("whitespace_response", "  \n ", "18", N),
    ("empty_box", r"\boxed{}", "18", N),
    ("closed_box_then_empty_box", r"Answer \boxed{18}. Format: \boxed{}", "18", C),
    ("answer_isnt_is_not_a_phrase_match", "The answer isn't clear to me.", "18", N),
    ("answer_is_needs_whole_words", "The unanswer is 5.", "5", N),
    ("unclosed_box_truncated", r"so the result is \boxed{1", "18", U),
    ("closed_then_unclosed_box", r"\boxed{18} and then \boxed{1", "18", U),
    ("closed_box_with_garbage", r"\boxed{???}", "18", U),
    ("closed_box_only_punctuation", r"\boxed{,}", "18", U),
    ("closed_box_unknown_macro", r"\boxed{\unknownmacro{3}}", "18", U),
    ("closed_box_incomplete_fraction", r"\boxed{\frac{1}{}}", "18", U),
]


@pytest.mark.parametrize(
    ("response", "truth", "status"),
    [pytest.param(r, t, s, id=i) for i, r, t, s in STATUS_CASES],
)
def test_status(response: str, truth: str, status: AnswerStatus) -> None:
    result = check_response(response, truth)
    assert result.status is status
    # Only the 'correct' status counts as correct; everything else is wrong.
    assert result.correct is (status is C)


def test_math_verify_failure_is_reported_as_check_error(monkeypatch) -> None:
    """A timeout in math-verify is wrong, but not mistaken for an ordinary wrong answer."""
    import blindspots.checker as checker

    def timing_out(*args, **kwargs):
        raise checker.TimeoutException("Operation timed out!")

    monkeypatch.setattr(checker, "verify", timing_out)
    result = check_response(r"\boxed{17}", "18")
    assert result.status is AnswerStatus.CHECK_ERROR
    assert not result.correct
    assert result.extracted == "17"


def test_wrong_and_missing_answers_are_distinguishable() -> None:
    """Missing answers are wrong but keep their own status for separate logging."""
    results = [check_response(r, "18") for r in ("", r"\boxed{17}", r"\boxed{1")]
    assert [r.status for r in results] == [N, W, U]
    assert not any(r.correct for r in results)


# --- several boxed answers ---


def test_last_box_wins_when_last_is_correct() -> None:
    result = check_response(r"First \boxed{17}, no wait, \boxed{18}.", "18")
    assert result.correct
    assert result.multiple_distinct_answers
    assert result.n_boxed == 2


def test_last_box_wins_when_last_is_wrong() -> None:
    result = check_response(r"First \boxed{18}, no wait, \boxed{17}.", "18")
    assert result.status is W
    assert result.multiple_distinct_answers


def test_repeated_identical_boxes_are_not_distinct() -> None:
    result = check_response(r"\boxed{5} ... therefore \boxed{5}", "5")
    assert result.correct
    assert result.n_boxed == 2
    assert not result.multiple_distinct_answers


def test_equivalent_forms_are_not_distinct() -> None:
    result = check_response(r"\boxed{1/2} that is \boxed{\frac{1}{2}}", r"\frac{1}{2}")
    assert result.correct
    assert not result.multiple_distinct_answers


def test_single_box_is_not_multiple() -> None:
    result = check_response(r"\boxed{18}", "18")
    assert result.n_boxed == 1
    assert not result.multiple_distinct_answers


def test_equivalence_check_is_tried_in_both_directions() -> None:
    # verify() treats its first argument as the reference; x=3 and 3 must not
    # count as different answers just because of the argument order.
    result = check_response(r"\boxed{3} and finally \boxed{x=3}", "3")
    assert result.correct
    assert not result.multiple_distinct_answers


def test_empty_boxes_are_ignored_for_the_distinct_flag() -> None:
    result = check_response(r"Put it in \boxed{}. Answer: \boxed{18}", "18")
    assert result.correct
    assert result.n_boxed == 1
    assert not result.multiple_distinct_answers


# --- truncation ---


def test_truncated_flag_follows_finish_reason() -> None:
    assert check_response(r"\boxed{18}", "18", finish_reason="length").truncated
    assert not check_response(r"\boxed{18}", "18", finish_reason="stop").truncated
    assert not check_response(r"\boxed{18}", "18").truncated


def test_truncated_without_answer_is_wrong_and_flagged() -> None:
    result = check_response("Let us compute step by step. First, 16 - 3", "18", "length")
    assert result.status is N
    assert result.truncated


def test_truncated_mid_box_is_unparseable_and_flagged() -> None:
    result = check_response(r"Therefore the answer is \boxed{1", "18", "length")
    assert result.status is U
    assert result.truncated
    assert result.extracted is None


def test_closed_box_before_truncation_still_counts() -> None:
    result = check_response(r"\boxed{18}. Let me double check the arithmetic by", "18", "length")
    assert result.correct
    assert result.truncated


# --- helpers ---


def test_find_boxed_handles_nested_and_escaped_braces() -> None:
    text = r"\boxed{\frac{1}{2}} then \boxed{\{1,2\}}"
    assert find_boxed(text) == [r"\frac{1}{2}", r"\{1,2\}"]


def test_find_boxed_marks_unclosed_box() -> None:
    assert find_boxed(r"\boxed{3} and \boxed{\frac{1}{") == ["3", None]
    assert find_boxed("no box here") == []


def test_extract_answer_prefers_boxed_over_phrase() -> None:
    answer, unclosed, n_boxed, _ = extract_answer(r"The answer is 1. \boxed{2}")
    assert (answer, unclosed, n_boxed) == ("2", False, 1)


def test_strip_trailing_units_only_touches_plain_number_plus_words() -> None:
    assert strip_trailing_units("18 dollars") == "18"
    assert strip_trailing_units(r"\text{18 dollars}") == "18"
    assert strip_trailing_units("-2.5 meters per second") == "-2.5"
    assert strip_trailing_units(r"\frac{1}{2}") == r"\frac{1}{2}"
    assert strip_trailing_units(r"2\sqrt{3}") == r"2\sqrt{3}"
    assert strip_trailing_units("5 and 6") == "5 and 6"  # not a single unit phrase
    assert strip_trailing_units("3 times") == "3 times"  # math word, left alone


def test_unparseable_ground_truth_is_an_error_not_a_wrong_answer() -> None:
    with pytest.raises(ValueError):
        parse_ground_truth("")
    with pytest.raises(ValueError):
        check_response(r"\boxed{18}", "")


def test_long_noisy_response_still_extracts_last_box() -> None:
    filler = "Let's think. " * 500
    result = check_response(filler + r"\boxed{18}", "18")
    assert result.correct

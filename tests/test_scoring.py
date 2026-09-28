"""Unit tests for the scoring classes. Run: pytest -q tests/test_scoring.py"""
import pytest

from blend.response_language import AzerbaijaniResponseClassifier, PersianResponseClassifier
from blend.scoring import (AnswerExtractor, BlendScorer, EnglishLemmatizer, LanguageNormalizer, NO_MATCH,
                           ResponseCleaner, ScoreResult)


@pytest.fixture(scope="module")
def scorer():
    return BlendScorer.default()


def ann(answers, en_answers, count=3):
    return {"annotations": [{"answers": answers, "en_answers": en_answers, "count": count}]}


# ---------------------------------------------------------------- ResponseCleaner
def test_clean_response_strips_answer_prefix_and_period():
    assert ResponseCleaner.clean("Answer: Plov.", "Read ...\n\nQuestion: q?\nAnswer:") == "plov"


def test_clean_response_handles_none():
    assert ResponseCleaner.clean(None, "Answer:") == ""


# ---------------------------------------------------------------- BlendScorer
def test_english_lemmatised_match(scorer):
    assert scorer.matches("sweet", "sweets", "English")


def test_empty_reference_never_matches(scorer):
    # BLEnD Iran New-gr-08 has "" as an English reference; it must not match everything
    assert not scorer.matches("", "anything at all", "English")
    assert scorer.score(ann([""], [""]), "anything", "English") == NO_MATCH


def test_local_answers_checked_before_english(scorer):
    a = {"annotations": [{"answers": ["plov"], "en_answers": ["rice"], "count": 3},
                         {"answers": ["dolma"], "en_answers": ["dolma"], "count": 1}]}
    assert scorer.score(a, "plov", "Azerbaijani") == ScoreResult(1, 1.0, "plov")


def test_weight_is_vote_share(scorer):
    a = {"annotations": [{"answers": ["x"], "en_answers": ["tea"], "count": 4},
                         {"answers": ["y"], "en_answers": ["coffee"], "count": 1}]}
    assert scorer.score(a, "coffee", "English") == ScoreResult(1, 0.25, "coffee")


def test_persian_match(scorer):
    assert scorer.score(ann(["میوه"], ["fruit"]), "میوه", "Persian").binary == 1


def test_scorer_requires_english():
    with pytest.raises(ValueError):
        BlendScorer({})


def test_unknown_language_raises(scorer):
    with pytest.raises(ValueError):
        scorer.matches("a", "b c", "Klingon")


def test_new_language_plugs_in_without_changing_the_scorer():
    """Open/closed: a new LanguageNormalizer is enough to support a new language."""
    class UpperCaseToy(LanguageNormalizer):
        language = "Toy"

        def tokens(self, text):
            return text.upper().split()

    s = BlendScorer({"English": EnglishLemmatizer(), "Toy": UpperCaseToy()})
    assert s.score(ann(["b a"], ["zzz"]), "a x b", "Toy").binary == 1


# ---------------------------------------------------------------- AnswerExtractor
@pytest.mark.parametrize("text, first", [
    ("cafés, sports bars, teahouses (çayxana)", "cafés"),
    ("فلافل، سمبوسه", "فلافل"),
    ("چای یا قهوه", "چای"),
    ("tea (çay)", "tea"),                       # aside, not a list
    ("bread and cheese", "bread and cheese"),   # one compound answer
    ("10/05", "10/05"),                         # dates are not split
    ("in iran, new year's eve is celebrated", "in iran new year's eve is celebrated"),  # intro phrase merged
])
def test_first_answer(text, first):
    assert AnswerExtractor.first_answer(text) == first


def test_is_multi_answer():
    assert AnswerExtractor.is_multi_answer("a, b and c")
    assert not AnswerExtractor.is_multi_answer("tea (çay)")
    assert not AnswerExtractor.is_multi_answer("in azerbaijan, it is traditional to give sweets")


# ---------------------------------------------------------------- response language
def test_response_language_classifiers():
    az, fa = AzerbaijaniResponseClassifier(), PersianResponseClassifier()
    assert az.classify("Şəki halvası") == "Azerbaijani"
    assert az.classify("Плов") == "Russian/Cyrillic"
    assert az.classify("Plov") == "uncertain"          # one word: not classifiable
    assert az.classify(None) == "empty"
    assert fa.classify("قورمه سبزی") == "Persian script"
    assert fa.classify("Ghormeh sabzi") == "Latin script"
    assert fa.classify("چلوکباب (Chelow kabab)") == "mixed"

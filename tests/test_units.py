"""Unit tests for config, data, stats and inference classes (small synthetic inputs).
Run: pytest -q tests/test_units.py"""

import pandas as pd
import pytest

from blend.config import COUNTRIES, MODELS, GenerationSettings, Paths, StatsSettings, model_name
from blend.inference import Generation, InferenceRunner, ModelClient, ResponseStore, RetryPolicy
from blend.stats import BetweenSettingComparison, PairedAnalysis, PairedScores

FAST = StatsSettings(n_boot=500, n_perm=500)


# ---------------------------------------------------------------- config
def test_country_specs():
    assert COUNTRIES["Iran"].languages == ("English", "Persian")
    assert COUNTRIES["Azerbaijan"].language == "Azerbaijani"


def test_model_names():
    assert model_name("gpt-4.1") == "GPT-4.1" and model_name("unknown") == "unknown"


def test_generation_defaults_reproduce_reported_runs(monkeypatch):
    monkeypatch.delenv("GEMINI_TEMPERATURE", raising=False)
    s = GenerationSettings.from_env()
    assert s.temperature == 0.0 and s.gemini_temperature == "default"


def test_paths_layout(tmp_path):
    p = Paths(blend_dir=tmp_path, results=tmp_path / "r")
    assert p.raw("Iran") == tmp_path / "r" / "Iran" / "raw" and p.comparison == tmp_path / "r" / "comparison"


# ---------------------------------------------------------------- data (needs BLEnD)
def test_valid_question_counts(iran, azerbaijan):
    assert len(iran.valid_questions) == 456 and len(azerbaijan.valid_questions) == 469


def test_build_prompt_uses_language_specific_template(iran):
    row = iran.valid_questions.iloc[0]
    en, fa = iran.build_prompt(row, "English"), iran.build_prompt(row, "Persian")
    assert row["Translation"] in en and row["Question"] in fa and en != fa


def test_empty_reference_exists_in_blend(iran):
    """The data problem that motivated the empty-reference safeguard."""
    assert any(a == "" for g in iran.annotation("New-gr-08")["annotations"] for a in g["en_answers"])


# ---------------------------------------------------------------- stats
def scores(en, loc, topics=None):
    n = len(en)
    return PairedScores(pd.DataFrame({"id": [f"q{i}" for i in range(n)],
                                      "topic": topics or ["Food"] * n, "en": en, "loc": loc}))


def test_paired_scores_validates_columns():
    with pytest.raises(ValueError):
        PairedScores(pd.DataFrame({"id": [1]}))


def test_summary_counts_and_gap():
    s = PairedAnalysis(scores([1, 1, 0, 0, 1], [1, 0, 1, 1, 1]), FAST).summary()
    assert (s["both_correct"], s["en_only"], s["loc_only"], s["both_wrong"]) == (2, 1, 2, 0)
    assert s["gap_pp"] == pytest.approx(-20.0)          # 60% - 80%: local better -> negative
    assert s["gap_lo"] <= s["gap_pp"] <= s["gap_hi"]


def test_mcnemar_exact_p():
    # 10 discordant pairs all in one direction -> exact two-sided p = 2 * 0.5**10
    s = PairedAnalysis(scores([0] * 10 + [1] * 5, [1] * 10 + [1] * 5), FAST).summary()
    assert s["mcnemar_p"] == pytest.approx(2 * 0.5 ** 10)


def test_results_do_not_depend_on_call_order():
    a = PairedAnalysis(scores([1, 0, 1, 0, 1, 1], [0, 0, 1, 1, 1, 0]), FAST)
    first = a.summary()
    a.discordance_homogeneity()
    assert a.summary() == first


def test_domain_table_has_holm_column():
    t = PairedAnalysis(scores([1, 0, 1, 0], [0, 1, 0, 1], ["Food", "Food", "Sport", "Sport"]), FAST).domain_table()
    assert list(t.domain) == ["Food", "Sport"] and "mcnemar_p_holm" in t


def test_between_setting_difference():
    a = scores([1] * 10, [0] * 10)             # gap +100
    b = scores([0] * 10, [0] * 10)             # gap 0
    d = BetweenSettingComparison(a, b, FAST).difference()
    assert d["diff_pp"] == pytest.approx(100.0) and d["p_permutation"] < 0.01


# ---------------------------------------------------------------- inference (fake model, no API)
class FakeClient(ModelClient):
    """Dependency inversion: the runner works with any ModelClient."""
    def __init__(self, fail_first=0):
        super().__init__(MODELS["gpt-4.1"], GenerationSettings())
        self.calls, self.fail_first = 0, fail_first

    def generate(self, prompt):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise TimeoutError("simulated")
        return Generation("Plov", {"finish_reason": "stop", "temperature": 0})


def test_runner_saves_and_resumes(iran, tmp_path):
    store = ResponseStore(tmp_path / "raw.jsonl")
    runner = InferenceRunner(iran, FakeClient(), store, RetryPolicy(sleep=lambda s: None))
    assert runner.run("Persian", limit=2) == 2
    assert runner.run("Persian", limit=2) == 2           # resume: the NEXT two questions
    recs = list(store.records())
    assert len({r["id"] for r in recs}) == 4
    assert all(r["prompt"] and r["temperature"] == 0 and r["language"] == "Persian" for r in recs)


def test_retry_then_success(iran, tmp_path):
    client = FakeClient(fail_first=2)
    store = ResponseStore(tmp_path / "raw.jsonl")
    InferenceRunner(iran, client, store, RetryPolicy(sleep=lambda s: None)).run("English", limit=1)
    assert client.calls == 3 and len(list(store.records())) == 1


def test_failed_question_is_not_saved(iran, tmp_path):
    store = ResponseStore(tmp_path / "raw.jsonl")
    InferenceRunner(iran, FakeClient(fail_first=99), store, RetryPolicy(tries=2, sleep=lambda s: None)).run("English", limit=1)
    assert list(store.records()) == []


def test_store_keeps_last_response_per_id(tmp_path):
    store = ResponseStore(tmp_path / "raw.jsonl")
    store.append({"id": "a", "response": "old"})
    store.append({"id": "a", "response": "new"})
    assert store.latest_by_id()["a"]["response"] == "new"

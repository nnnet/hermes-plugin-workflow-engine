"""Unit tests for core.detectors — pure functions, no mocks needed."""
import pytest
from core import detectors as d


# ─── tokens / jaccard / stem ────────────────────────────────────────────

def test_tokens_filters_short():
    assert d.tokens("я и ты в Москве") == {"москве"}  # >=3 chars

def test_tokens_lowercases():
    assert d.tokens("Москва Москвой") == {"москва", "москвой"}

def test_tokens_empty():
    assert d.tokens(None) == set()
    assert d.tokens("") == set()

def test_jaccard_identical():
    a = {"x", "y"}
    assert d.jaccard(a, a) == 1.0

def test_jaccard_disjoint():
    assert d.jaccard({"a"}, {"b"}) == 0.0

def test_jaccard_both_empty():
    assert d.jaccard(set(), set()) == 1.0

def test_stem_match_russian_declension():
    # «крипторынок» vs «крипторынке» — same root, different form
    assert d.has_stem_match("крипторынок", {"крипторынке"}) is True

def test_stem_match_unrelated():
    assert d.has_stem_match("яблоко", {"машина"}) is False

def test_stem_match_short_token_exact():
    # «бот» is 3 chars (< min_stem=4) — falls back to exact match
    assert d.has_stem_match("бот", {"бот"}) is True
    assert d.has_stem_match("бот", {"бота"}) is False  # exact only


# ─── non-answer detection ───────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "начинай с defaults",
    "уже всё ответил выше",
    "делай как считаешь",
    "продолжай",
    "go ahead",
])
def test_non_answer_explicit_push(text):
    assert d.is_non_answer(text, "open") is True

def test_non_answer_surface_affirm_to_open_q():
    assert d.is_non_answer("да", "open") is True
    assert d.is_non_answer("ок", "open") is True

def test_non_answer_surface_affirm_to_yesno_q():
    # «да» is valid for yes/no questions, NOT a non-answer
    assert d.is_non_answer("да", "yesno") is False

def test_non_answer_substantive_open():
    assert d.is_non_answer(
        "хочу финансовой независимости", "open"
    ) is False

def test_non_answer_empty():
    assert d.is_non_answer(None) is True
    assert d.is_non_answer("") is True


# ─── user-pushed-for-action ────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "начинай",
    "Уже всё ответил",
    "делай с defaults",
    "если можешь начать — начинай",
    "поехали",
])
def test_user_pushed_positive(text):
    assert d.detect_user_pushed_for_action(text) is True

@pytest.mark.parametrize("text", [
    "ещё подумаю",
    "хочу обсудить детали",
    "какие у тебя варианты?",
])
def test_user_pushed_negative(text):
    assert d.detect_user_pushed_for_action(text) is False


# ─── means-as-goal ─────────────────────────────────────────────────────

@pytest.mark.parametrize("goal", [
    "бот",
    "магазин",
    "прибыль",
    "канал на YouTube",
    "торговый бот",
])
def test_means_as_goal_flags(goal):
    assert d.is_means_as_goal(goal) is True, goal

@pytest.mark.parametrize("goal", [
    "финансовая независимость через бот",  # «независимость» qualifier
    "магазин для помощи людям",  # «для» + «помощи»
    "выучить английский",  # no means at all
    "научиться играть на гитаре",
])
def test_means_as_goal_no_flag(goal):
    assert d.is_means_as_goal(goal) is False, goal


# ─── slot extraction ───────────────────────────────────────────────────

def test_extract_slots_basic():
    text = """
— **Истинная цель**: прибыль от крипты
— **Средство**: торговый бот
— **Место/контекст**: крипторынок
"""
    out = d.extract_slots(text, {
        "истинная_цель": ["Истинная цель"],
        "средство": ["Средство"],
        "место": ["Место/контекст"],
    })
    assert out["истинная_цель"] == "прибыль от крипты"
    assert out["средство"] == "торговый бот"
    assert out["место"] == "крипторынок"

def test_extract_slots_placeholder_returns_none():
    text = """
— **Истинная цель**: **не указано**
— **Средство**: торговый бот
"""
    out = d.extract_slots(text, {
        "истинная_цель": ["Истинная цель"],
        "средство": ["Средство"],
    })
    assert out["истинная_цель"] is None
    assert out["средство"] == "торговый бот"

def test_extract_slots_aliases():
    text = "— **True goal**: financial freedom"
    out = d.extract_slots(text, {
        "истинная_цель": ["Истинная цель", "True goal"],
    })
    assert out["истинная_цель"] == "financial freedom"


# ─── lock-signal ───────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Истинная цель определена: переехать",
    "Цель зафиксирована: переезд",
    "Фиксирую цель: получить визу",
])
def test_lock_signal_default(text):
    assert d.has_lock_signal(text) is True

def test_lock_signal_no_phrase():
    assert d.has_lock_signal("Понял, начинаю работать") is False


# ─── grounding ─────────────────────────────────────────────────────────

def test_grounding_all_from_user():
    slots = ["торговый бот", "крипторынок"]
    user = ["хочу торгового бота на крипторынке"]
    assert d.grounding_score(slots, user) == 1.0

def test_grounding_zero_when_training_fill():
    slots = ["EMA crossover Bybit Python ccxt"]
    user = ["хочу прибыль"]
    assert d.grounding_score(slots, user) == 0.0

def test_grounding_empty_slots_zero():
    assert d.grounding_score([None, None], ["text"]) == 0.0

def test_grounding_no_user_neutral():
    assert d.grounding_score(["x"], []) == 0.5

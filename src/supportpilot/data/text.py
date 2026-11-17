"""Text normalisation, script-based language checks and PII masking (DATA_CONTRACTS §6).

Everything here is deterministic and dictionary/regex based. Language and code-switch
results are automatic signals, never a substitute for a fluent human reviewer (EVALUATION §1).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

LangGuess = Literal["en", "ru", "kk", "unknown"]

_WS_RE = re.compile("[ \t  -​　]+")
_NL_RE = re.compile(r"\s*\n\s*")
_CTRL_RE = re.compile("[\u0000-\u0008\u000b-\u001f\u007f-\u009f﻿]")
_WORD_RE = re.compile("[^\\W\\d_]+(?:['’-][^\\W\\d_]+)*", re.UNICODE)
_MOJIBAKE_RE = re.compile("(?:Ð[\u0080-¿]|Ñ[\u0080-¿]|Ã[\u0080-¿]|�)")

KK_LETTERS = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
# Frequent Kazakh function words and question particles (lower case, Cyrillic).
KK_MARKERS = frozenset(
    "ма ме ба бе мен және керек қалай қашан қайда неше маған менің мені сен сіз сізге сіздің "
    "бұл осы ол олар үшін деп емес еді болады болса ғой ғана қандай қанша кім неге неліктен "
    "жатыр тұр беру айт көрсет".split()
)  # no words that are also common Russian words (не, бар, па)
# Russian words that do not exist in Kazakh; enough of them in a Kazakh text = code-switch.
RU_MARKERS = frozenset(
    "и в не на что это как мне я вы у меня по с за из от до когда где почему можно нужно "
    "надо хочу пожалуйста здравствуйте спасибо заказ доставка оплата возврат товар деньги "
    "карта карту ещё еще уже очень просто только".split()
)


def normalize_text(text: str) -> str:
    """NFC, no control characters, single spaces, `\\n` newlines, stripped."""
    text = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return _NL_RE.sub("\n", text).strip()


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def dedup_key(text: str) -> str:
    """Case/punctuation-insensitive key used for exact-duplicate detection."""
    return " ".join(words(unicodedata.normalize("NFKC", text).replace("ё", "е")))


def has_mojibake(text: str) -> bool:
    return _MOJIBAKE_RE.search(text) is not None


@dataclass(frozen=True)
class LangSignal:
    guess: LangGuess
    code_switch: bool
    latin: int
    cyrillic: int
    kk_letters: int
    kk_markers: int
    ru_markers: int


def detect_language(text: str) -> LangSignal:
    """Script + lexicon signal. Kazakh-specific letters or particles decide kk vs ru."""
    latin = sum(1 for c in text if "a" <= c.lower() <= "z")
    cyr = sum(1 for c in text if "Ѐ" <= c <= "ӿ")
    kk_letters = sum(1 for c in text if c in KK_LETTERS)
    toks = words(text)
    kk_m = sum(1 for t in toks if t in KK_MARKERS)
    ru_m = sum(1 for t in toks if t in RU_MARKERS)
    guess: LangGuess
    if latin == 0 and cyr == 0:
        guess = "unknown"
    elif latin > cyr:
        guess = "en"
    elif kk_letters or kk_m:
        guess = "kk"
    else:
        guess = "ru"
    kk_side = kk_letters > 0 or kk_m > 0
    switch = cyr > 0 and kk_side and ru_m >= 2
    return LangSignal(guess, switch, latin, cyr, kk_letters, kk_m, ru_m)


# ---------------------------------------------------------------- PII

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.UNICODE)),
    ("url", re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)),
    # card numbers: 13-19 digits, optionally grouped by spaces or dashes
    ("card", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    # phones: +7 / 8 prefixes (KZ/RU), US-style (xxx) xxx-xxxx, or 10-12 digit runs
    (
        "phone",
        re.compile(
            r"(?:\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]?\d{2}[\s.-]?\d{2}\b"
            r"|\+\d{10,12}\b"
        ),
    ),
    # long identifiers (IIN = 12 digits, order/account numbers): 7+ digit runs
    ("number", re.compile(r"\b\d{7,}\b")),
)


def mask_pii(text: str, public: frozenset[str] = frozenset()) -> tuple[str, dict[str, int]]:
    """Replace e-mails, URLs, card/phone numbers and long digit runs with `<kind>` tags.

    Values in `public` (e.g. an agency phone printed in the grounding document) are kept."""
    counts: dict[str, int] = {}
    for kind, pattern in _PII_PATTERNS:

        def repl(m: re.Match[str], kind: str = kind) -> str:
            if m.group(0) in public:
                return m.group(0)
            counts[kind] = counts.get(kind, 0) + 1
            return f"<{kind}>"

        text = pattern.sub(repl, text)
    return text, counts


def pii_values(text: str) -> frozenset[str]:
    """All substrings the PII patterns match (used as the public allow-list of a document)."""
    return frozenset(m.group(0) for _, p in _PII_PATTERNS for m in p.finditer(text))


def residual_pii(text: str, public: frozenset[str] = frozenset()) -> list[str]:
    """PII kinds still present after masking (must be empty for published rows)."""
    return [
        kind
        for kind, pattern in _PII_PATTERNS
        if any(m.group(0) not in public for m in pattern.finditer(text))
    ]


_INTRO_NAME = re.compile(
    r"(?P<lead>\b(?:my name is|this is|меня зовут|моё имя|мое имя|менің атым|атым)\s+)"
    r"(?P<name>(?-i:[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ][\w'-]+)(?:\s+(?-i:[A-ZА-ЯЁӘҒҚҢӨҰҮҺІ][\w'-]+)){0,2})",
    re.IGNORECASE,
)
_NOT_NAMES = frozenset(
    {"i", "a", "the", "hi", "hello", "trying", "having", "looking", "wondering", "not", "sure"}
)


def mask_intro_name(text: str) -> tuple[str, int]:
    """Mask a capitalised name after a self-introduction ("my name is Joe Doe")."""
    count = 0

    def repl(m: re.Match[str]) -> str:
        nonlocal count
        if m.group("name").split()[0].lower() in _NOT_NAMES:
            return m.group(0)
        count += 1
        return m.group("lead") + "<name>"

    return _INTRO_NAME.sub(repl, text), count


_TAG_RE = re.compile(r"<[a-z_]+>")


def content_words(text: str) -> list[str]:
    """Words without masking tags such as `<name>`."""
    return words(_TAG_RE.sub(" ", text))


def mask_values(text: str, values: dict[str, list[str]]) -> tuple[str, dict[str, int]]:
    """Mask known values (e.g. a role-play profile's name) case-insensitively, longest first."""
    counts: dict[str, int] = {}
    pairs = sorted(
        ((v, k) for k, vs in values.items() for v in vs if len(v.strip()) >= 3),
        key=lambda p: (-len(p[0]), p[0]),
    )
    for value, kind in pairs:
        pattern = re.compile(r"(?<!\w)" + re.escape(value.strip()) + r"(?!\w)", re.IGNORECASE)
        text, n = pattern.subn(f"<{kind}>", text)
        if n:
            counts[kind] = counts.get(kind, 0) + n
    return text, counts

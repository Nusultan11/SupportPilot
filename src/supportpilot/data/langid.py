"""Language identification for en/ru/kk with script, confidence and code-switch (ADR-020).

Detector: lingua (Apache-2.0, offline n-gram models) over our three languages plus a set of
"other" languages, so that a confidently different language is caught. Decision rule:

1. Fewer than `min_letters` letters -> `und` (undetermined; e.g. "?", "👍", "ok").
2. Lingua's top language is not en/ru/kk with confidence >= `other_min_conf` -> `other:<iso>`.
3. Otherwise the best of en/ru/kk that fits the script (Latin -> en; Cyrillic -> ru or kk,
   Kazakh-specific letters break near-ties). `conf` = lingua's confidence for that language
   among all candidates, so short or typo-heavy text gets a low confidence, never a wrong
   language. Calibrated on Belebele (gold language labels), see the EDA report.
4. `code_switch`: >= 2 words each in Latin and Cyrillic script, or Cyrillic segments that are
   confidently ru and kk in the same text.

Latin-script Kazakh is not detected (lingua's Kazakh model is Cyrillic); it shows up as
`en` with low confidence or `other:*`. This is a known limitation (DATA_CONTRACTS §1).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache

from lingua import ConfidenceValue, Language, LanguageDetector, LanguageDetectorBuilder

SUPPORTED: dict[Language, str] = {
    Language.ENGLISH: "en",
    Language.RUSSIAN: "ru",
    Language.KAZAKH: "kk",
}
OTHERS: tuple[Language, ...] = (
    Language.UKRAINIAN,
    Language.BELARUSIAN,
    Language.BULGARIAN,
    Language.MACEDONIAN,
    Language.SERBIAN,
    Language.MONGOLIAN,
    Language.TURKISH,
    Language.AZERBAIJANI,
    Language.GERMAN,
    Language.FRENCH,
    Language.SPANISH,
    Language.ITALIAN,
    Language.PORTUGUESE,
    Language.DUTCH,
    Language.POLISH,
    Language.INDONESIAN,
    Language.TAGALOG,
    Language.CHINESE,
    Language.ARABIC,
    Language.HINDI,
)
KK_LETTERS = frozenset("әғқңөұүһіӘҒҚҢӨҰҮҺІ")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ɏ]")
_CYR_RE = re.compile(r"[Ѐ-ӿ]")
_WORD_RE = re.compile(r"\w+")
_SEGMENT_RE = re.compile(r"[.!?;:\n]+")


@dataclass(frozen=True)
class LanguageResult:
    lang: str  # en | ru | kk | other:<iso639-1> | und
    conf: float
    script: str  # Latn | Cyrl | Mixed | Other | None
    code_switch: bool
    langs: tuple[str, ...]


def _iso(lang: Language) -> str:
    return SUPPORTED.get(lang) or lang.iso_code_639_1.name.lower()


@cache
def _detector() -> LanguageDetector:
    langs = (*SUPPORTED, *OTHERS)
    return LanguageDetectorBuilder.from_languages(*langs).with_preloaded_language_models().build()


@cache
def _cyr_detector() -> LanguageDetector:
    return LanguageDetectorBuilder.from_languages(Language.RUSSIAN, Language.KAZAKH).build()


def _script(text: str) -> tuple[str, int, int, int]:
    latin = len(_LATIN_RE.findall(text))
    cyr = len(_CYR_RE.findall(text))
    letters = sum(ch.isalpha() for ch in text)
    other = letters - latin - cyr
    if letters == 0:
        return "None", 0, 0, 0
    for name, n in (("Latn", latin), ("Cyrl", cyr), ("Other", other)):
        if n / letters >= 0.8:
            return name, latin, cyr, letters
    return "Mixed", latin, cyr, letters


def _script_words(text: str) -> tuple[int, int]:
    """Lower-case-initial words per script. Capitalised words are mostly names/brands
    ("Cinque Terre", "ASUS Eee PC") and do not make a text bilingual."""
    words = [w for w in _WORD_RE.findall(text) if w[:1].isalpha() and w[:1].islower()]
    lat = sum(1 for w in words if _LATIN_RE.search(w) and not _CYR_RE.search(w))
    cyr = sum(1 for w in words if _CYR_RE.search(w) and not _LATIN_RE.search(w))
    return lat, cyr


def _cyr_choice(text: str, conf: dict[str, float]) -> str:
    ru, kk = conf.get("ru", 0.0), conf.get("kk", 0.0)
    if any(ch in KK_LETTERS for ch in text) and kk >= 0.5 * ru:
        return "kk"
    return "kk" if kk > ru else "ru"


def _ru_kk_switch(text: str) -> bool:
    found: set[str] = set()
    det = _cyr_detector()
    for seg in _SEGMENT_RE.split(text):
        if len(_WORD_RE.findall(seg)) < 3 or not _CYR_RE.search(seg):
            continue
        values = det.compute_language_confidence_values(seg)
        top = values[0]
        if top.value >= 0.8:
            found.add(SUPPORTED[top.language])
    return found == {"ru", "kk"}


def _decide(
    text: str, values: Sequence[ConfidenceValue], *, other_min_conf: float, min_letters: int
) -> LanguageResult:
    script, latin, cyr, letters = _script(text)
    if letters < min_letters:
        return LanguageResult("und", 0.0, script, False, ())
    conf = {_iso(v.language): v.value for v in values}
    top = values[0] if values else None
    if top is not None and top.language not in SUPPORTED and top.value >= other_min_conf:
        iso = _iso(top.language)
        return LanguageResult(f"other:{iso}", top.value, script, False, (f"other:{iso}",))
    if script == "Other":
        iso = _iso(top.language) if top is not None else "xx"
        return LanguageResult(f"other:{iso}", conf.get(iso, 0.0), script, False, ())
    lat_words, cyr_words = _script_words(text)
    if script == "Cyrl" or (
        script == "Mixed" and (cyr_words > lat_words or (cyr_words == lat_words and cyr >= latin))
    ):
        lang = _cyr_choice(text, conf)
    else:
        lang = "en"
    langs: tuple[str, ...] = (lang,)
    code_switch = False
    if lat_words >= 2 and cyr_words >= 2:
        second = "en" if lang != "en" else _cyr_choice(text, conf)
        langs, code_switch = tuple(sorted({lang, second})), True
    elif cyr_words >= 6 and _ru_kk_switch(text):
        langs, code_switch = ("kk", "ru"), True
    return LanguageResult(lang, round(conf.get(lang, 0.0), 4), script, code_switch, langs)


class LanguageIdentifier:
    def __init__(self, *, other_min_conf: float = 0.9, min_letters: int = 3) -> None:
        self.other_min_conf = other_min_conf
        self.min_letters = min_letters

    def detect(self, text: str) -> LanguageResult:
        return self.detect_many([text])[0]

    def detect_many(self, texts: Sequence[str]) -> list[LanguageResult]:
        values = _detector().compute_language_confidence_values_in_parallel(list(texts))
        return [
            _decide(t, v, other_min_conf=self.other_min_conf, min_letters=self.min_letters)
            for t, v in zip(texts, values, strict=True)
        ]

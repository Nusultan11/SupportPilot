"""Translation quality estimation, domain post-editing and Kazakh checks (Stage 4, ADR-024).

No reference translations exist for our texts, so quality is estimated from signals that
need none:
- back-translation chrF: the target is translated back to English and compared with the
  source (character n-gram F-score, chrF with beta=2, n=1..6);
- cross-lingual embedding similarity (multilingual-e5) between source and target;
- hard checks: script, language ID, Kazakh-specific letters, Russian leakage in Kazakh,
  numbers and names preserved, untranslated Latin words, length ratio, repetition loops;
- domain glossary: known mistranslations of support terms are repaired (`fixes`) when a
  deterministic repair exists, and flagged otherwise.
Thresholds are calibrated on the reviewed MT sample (docs/reviews/stage4_mt_review.yaml).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from supportpilot.data.langid import KK_LETTERS, LanguageIdentifier

_WORD_RE = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", re.UNICODE)
_NUM_RE = re.compile(r"\d[\d  ,.]*\d|\d")
_CYR_RE = re.compile(r"[Ѐ-ӿ]")
_LAT_RE = re.compile(r"[A-Za-z]")
# Lower-case Latin words left in a RU/KK translation are copied source typos ("uhelp",
# "paymenr", "od"): the reviewed post-QE sample showed they mark broken translations.
_LOWER_LATIN_RE = re.compile(r"(?<![\w@.])[a-z]{2,}(?![\w@])")
LATIN_ALLOWED = frozenset({"sms", "qr", "eta", "id", "ok", "pdf", "e", "mail"})

# Russian function words that do not exist in Kazakh: their presence in a Kazakh MT output
# means a Russian fragment. (Kazakh speakers do use Russian *nouns* in chat; the code-switch
# data is authored separately, so MT output must be Kazakh.)
RU_FUNCTION_WORDS = frozenset(
    [
        "и",
        "в",
        "во",
        "на",
        "не",
        "что",
        "это",
        "для",
        "по",
        "как",
        "с",
        "со",
        "у",
        "я",
        "мне",
        "меня",
        "мой",
        "моя",
        "мои",
        "вы",
        "вас",
        "вам",
        "ваш",
        "ваша",
        "есть",
        "нет",
        "или",
        "но",
        "же",
        "был",
        "была",
        "было",
        "будет",
        "когда",
        "где",
        "почему",
        "можно",
        "нужно",
        "хочу",
        "пожалуйста",
        "уже",
        "еще",
        "ещё",
        "только",
        "если",
        "чтобы",
        "его",
        "её",
        "их",
        "от",
        "до",
        "из",
        "за",
        "при",
        "без",
        "над",
        "под",
    ]
)
KK_FUNCTION_WORDS = frozenset(
    [
        "мен",
        "сен",
        "сіз",
        "біз",
        "олар",
        "бұл",
        "осы",
        "және",
        "немесе",
        "бірақ",
        "үшін",
        "туралы",
        "қалай",
        "қашан",
        "қайда",
        "неге",
        "бар",
        "жоқ",
        "керек",
        "болады",
        "емес",
        "ма",
        "ме",
        "ба",
        "бе",
        "па",
        "пе",
        "ғой",
        "да",
        "де",
        "та",
        "те",
    ]
)


def chrf(hyp: str, ref: str, *, n: int = 6, beta: float = 2.0) -> float:
    """chrF (Popović 2015): character n-gram F-beta, whitespace removed, n = 1..`n`."""
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    precs, recs = [], []
    for k in range(1, n + 1):
        hg = Counter(h[i : i + k] for i in range(len(h) - k + 1))
        rg = Counter(r[i : i + k] for i in range(len(r) - k + 1))
        if not hg or not rg:
            continue
        overlap = sum((hg & rg).values())
        precs.append(overlap / sum(hg.values()))
        recs.append(overlap / sum(rg.values()))
    if not precs:
        return 0.0
    p, rc = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + rc == 0:
        return 0.0
    b2 = beta * beta
    return round((1 + b2) * p * rc / (b2 * p + rc), 4)


def numbers(text: str) -> set[str]:
    """Digit runs with thousands separators removed: "15,000" / "15 000" -> "15000"."""
    out = set()
    for m in _NUM_RE.finditer(text):
        raw = m.group(0)
        if re.fullmatch(r"\d{1,3}(?:[  ,]\d{3})+", raw):
            out.add(re.sub(r"\D", "", raw))
        else:
            out.update(re.findall(r"\d+", raw))
    return out


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def repetition(text: str) -> bool:
    """A token or bigram repeated 4+ times in a row (decoder loop)."""
    w = words(text)
    for size in (1, 2):
        run, prev = 1, None
        for i in range(0, len(w) - size + 1, size):
            gram = tuple(w[i : i + size])
            run = run + 1 if gram == prev else 1
            prev = gram
            if run >= 4:
                return True
    return False


@dataclass
class GlossaryRule:
    lang: str
    pattern: re.Pattern[str]
    replacement: dict[str, str] | None  # suffix -> repaired word; None = flag only
    stem: str
    target: str
    flag: str


def _rule(
    lang: str, stem: str, target: str, suffixes: dict[str, str] | None, flag: str
) -> GlossaryRule:
    pat = re.compile(rf"\b({re.escape(stem)})([а-яәғқңөұүһіё]*)", re.IGNORECASE)
    return GlossaryRule(lang, pat, suffixes, stem, target, flag)


# "order" translated as "task" (kk) / "decree" (ru). Suffix tables map the inflected form of
# the wrong word to the same inflection of the right one (vowel harmony and consonant
# assimilation differ, so it is a table, not a string replace).
_KK_TAPSYRMA = {
    "": "тапсырыс", "м": "тапсырысым", "мды": "тапсырысымды", "мның": "тапсырысымның",
    "ма": "тапсырысыма", "мда": "тапсырысымда", "мнан": "тапсырысымнан", "ң": "тапсырысың",
    "ңыз": "тапсырысыңыз", "ңызды": "тапсырысыңызды", "ңыздың": "тапсырысыңыздың",
    "ңызға": "тапсырысыңызға", "ңызда": "тапсырысыңызда", "сы": "тапсырысы",
    "сын": "тапсырысын", "сының": "тапсырысының", "сына": "тапсырысына",
    "сында": "тапсырысында", "сынан": "тапсырысынан", "ға": "тапсырысқа", "ны": "тапсырысты",
    "ның": "тапсырыстың", "да": "тапсырыста", "дан": "тапсырыстан", "мен": "тапсырыспен",
    "лар": "тапсырыстар", "ларым": "тапсырыстарым", "ларымды": "тапсырыстарымды",
    "ларыңыз": "тапсырыстарыңыз", "ларыңызды": "тапсырыстарыңызды", "лары": "тапсырыстары",
    "ларды": "тапсырыстарды", "ларын": "тапсырыстарын",
}  # fmt: skip
_RU_PRIKAZ = {
    "": "заказ", "а": "заказа", "у": "заказу", "ом": "заказом", "е": "заказе", "ы": "заказы",
    "ов": "заказов", "ам": "заказам", "ами": "заказами", "ах": "заказах",
}  # fmt: skip
_RU_TELEZHKA = {
    "а": "корзина", "и": "корзины", "у": "корзину", "е": "корзине", "ой": "корзиной",
}  # fmt: skip
# Rules from the reviewed MT comparison (docs/reviews/stage4_mt_review.yaml). A rule with a
# suffix table repairs the word; a rule without one marks the translation as unusable.
GLOSSARY: tuple[GlossaryRule, ...] = (
    _rule("kk", "тапсырма", "тапсырыс", _KK_TAPSYRMA, "glossary:kk_order_task"),
    _rule("kk", "қоқыс жәшіг", "себет", None, "glossary:kk_cart_trash_bin"),
    _rule("kk", "қоршау", "себет", None, "glossary:kk_cart_fence"),
    _rule("kk", "адам агент", "оператор", None, "glossary:kk_human_agent_calque"),
    _rule("kk", "мақала", "тауар", None, "glossary:kk_item_article"),
    _rule("kk", "пошта жәшіг", "сәлемдеме", None, "glossary:kk_parcel_mailbox"),
    _rule("kk", "банкнот", "шот", None, "glossary:kk_bill_banknote"),
    _rule("ru", "приказ", "заказ", _RU_PRIKAZ, "glossary:ru_order_decree"),
    _rule("ru", "тележк", "корзина", _RU_TELEZHKA, "glossary:ru_cart_trolley"),
    _rule("ru", "человеческ", "живой оператор", None, "glossary:ru_human_agent_calque"),
    _rule("ru", "стать", "товар", None, "glossary:ru_item_article"),
    _rule("ru", "возмезди", "возврат", None, "glossary:ru_refund_retribution"),
    _rule("ru", "купюр", "счёт", None, "glossary:ru_bill_banknote"),
)


def post_edit(text: str, lang: str) -> tuple[str, list[str], list[str]]:
    """Apply deterministic glossary repairs. Returns (text, fixes, unresolved_flags)."""
    fixes: list[str] = []
    unresolved: list[str] = []
    for rule in GLOSSARY:
        if rule.lang != lang or not rule.pattern.search(text):
            continue
        if rule.replacement is None:
            unresolved.append(rule.flag)
            continue
        table = rule.replacement

        def repl(m: re.Match[str], table: dict[str, str] = table) -> str:
            suffix = m.group(2).lower()
            if suffix not in table:
                return m.group(0)
            word = table[suffix]
            return word.capitalize() if m.group(1)[0].isupper() else word

        new = rule.pattern.sub(repl, text)
        if rule.pattern.search(new):
            unresolved.append(rule.flag)
        if new != text:
            fixes.append(rule.flag)
        text = new
    return text, fixes, unresolved


@dataclass
class KkReport:
    kk_letter_words: float  # share of words with a Kazakh-specific letter
    ru_function_words: int
    kk_function_words: int
    has_kk_letters: bool


def kk_report(text: str) -> KkReport:
    w = words(text)
    with_kk = sum(1 for x in w if any(ch in KK_LETTERS for ch in x))
    return KkReport(
        kk_letter_words=round(with_kk / len(w), 4) if w else 0.0,
        ru_function_words=sum(1 for x in w if x in RU_FUNCTION_WORDS),
        kk_function_words=sum(1 for x in w if x in KK_FUNCTION_WORDS),
        has_kk_letters=with_kk > 0,
    )


@dataclass
class QEResult:
    text: str  # post-edited target
    lang_detected: str
    lid_conf: float
    bt_chrf: float | None
    emb_sim: float | None
    fixes: list[str] = field(default_factory=list)
    hard: list[str] = field(default_factory=list)
    soft: list[str] = field(default_factory=list)
    accept: bool = False


@dataclass(frozen=True)
class QEThresholds:
    min_bt_chrf: float
    min_emb_sim: float
    length_ratio: tuple[float, float] = (0.45, 2.6)
    max_latin_words: float = 0.34


def assess(
    src: str,
    tgt_raw: str,
    lang: str,
    *,
    lid: LanguageIdentifier,
    back: str | None,
    emb_sim: float | None,
    keep: Iterable[str] = (),
    th: QEThresholds,
) -> QEResult:
    """Quality estimate for one translation `src` (en) -> `tgt_raw` (`lang`).

    `keep` holds strings that must survive translation (names and cities in target form,
    product codes); numbers are checked separately.
    """
    text, fixes, unresolved = post_edit(tgt_raw.strip(), lang)
    res = lid.detect(text) if text else None
    hard: list[str] = [*unresolved]
    soft: list[str] = []
    if not text:
        return QEResult(text, "und", 0.0, None, emb_sim, fixes, ["empty"], [], False)
    assert res is not None
    if res.lang != lang:
        hard.append(f"lid:{res.lang}")
    cyr, lat = len(_CYR_RE.findall(text)), len(_LAT_RE.findall(text))
    if cyr < 0.6 * (cyr + lat):
        hard.append("script_not_cyrillic")
    w = words(text)
    latin_words = sum(1 for x in w if _LAT_RE.search(x))
    if w and latin_words / len(w) > th.max_latin_words:
        hard.append("untranslated_latin")
    if any(m.group(0) not in LATIN_ALLOWED for m in _LOWER_LATIN_RE.finditer(text)):
        hard.append("latin_word")
    missing = numbers(src) - numbers(text)
    if missing:
        hard.append("numbers_changed")
    for s in keep:
        stem = s[: max(4, len(s) - 3)]  # inflected names keep the stem ("Алматыға")
        if stem.lower() not in text.lower():
            hard.append("name_lost")
            break
    ratio = len(text) / max(1, len(src))
    if not th.length_ratio[0] <= ratio <= th.length_ratio[1]:
        hard.append("length_ratio")
    if repetition(text):
        hard.append("repetition")
    if lang == "kk":
        rep = kk_report(text)
        if rep.ru_function_words >= 1 and rep.ru_function_words >= rep.kk_function_words:
            hard.append("kk:ru_leak")
        if len(w) >= 4 and not rep.has_kk_letters:
            soft.append("kk:no_kk_letters")
    bt = chrf(back, src) if back else None
    if bt is not None and bt < th.min_bt_chrf:
        hard.append("bt_chrf_low")
    if emb_sim is not None and emb_sim < th.min_emb_sim:
        hard.append("emb_sim_low")
    if res.conf < 0.5:
        soft.append("lid_low_conf")
    return QEResult(
        text=text,
        lang_detected=res.lang,
        lid_conf=res.conf,
        bt_chrf=bt,
        emb_sim=emb_sim,
        fixes=fixes,
        hard=sorted(set(hard)),
        soft=sorted(set(soft)),
        accept=not hard,
    )

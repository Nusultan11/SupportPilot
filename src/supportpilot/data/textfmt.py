"""Language-aware value formatting for authored templates (Stage 4, ADR-025).

Templates write `{name}` or `{name:fmt}` (optionally `{name:fmt@case}` for a Kazakh case
form: `abl` -дан/-ден, `dat` -ға/-ге, `loc` -да/-де). Russian plural agreement and Kazakh
case suffixes are table-driven, so a template never produces "1 дней" (a Stage 3 finding in
support-json-ru) or a suffix that breaks vowel harmony.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any

from supportpilot.common.schemas import LanguageCode

PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_.]*)(?::([a-z_]+)(?:@([a-z]+))?)?\}")

MONTHS: dict[str, tuple[str, ...]] = {
    "en": ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"),
    "ru": ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
           "сентября", "октября", "ноября", "декабря"),
    "kk": ("қаңтар", "ақпан", "наурыз", "сәуір", "мамыр", "маусым", "шілде", "тамыз",
           "қыркүйек", "қазан", "қараша", "желтоқсан"),
}  # fmt: skip
# Kazakh case forms of the month names (ablative, dative, locative).
_KK_MONTH_CASES: dict[str, dict[str, str]] = {
    "abl": dict(zip(MONTHS["kk"], ("қаңтардан", "ақпаннан", "наурыздан", "сәуірден", "мамырдан",
        "маусымнан", "шілдеден", "тамыздан", "қыркүйектен", "қазаннан", "қарашадан",
        "желтоқсаннан"), strict=True)),
    "dat": dict(zip(MONTHS["kk"], ("қаңтарға", "ақпанға", "наурызға", "сәуірге", "мамырға",
        "маусымға", "шілдеге", "тамызға", "қыркүйекке", "қазанға", "қарашаға",
        "желтоқсанға"), strict=True)),
    "loc": dict(zip(MONTHS["kk"], ("қаңтарда", "ақпанда", "наурызда", "сәуірде", "мамырда",
        "маусымда", "шілдеде", "тамызда", "қыркүйекте", "қазанда", "қарашада",
        "желтоқсанда"), strict=True)),
}  # fmt: skip
# Kazakh nouns used after numerals (no plural after a number), with case forms.
_KK_NOUN: dict[str, dict[str, str]] = {
    "теңге": {"nom": "теңге", "abl": "теңгеден", "dat": "теңгеге", "loc": "теңгеде"},
    "күн": {"nom": "күн", "abl": "күннен", "dat": "күнге", "loc": "күнде"},
    "жұмыс күні": {"nom": "жұмыс күні", "abl": "жұмыс күнінен", "dat": "жұмыс күніне",
                   "loc": "жұмыс күнінде"},
    "ай": {"nom": "ай", "abl": "айдан", "dat": "айға", "loc": "айда"},
    "сағат": {"nom": "сағат", "abl": "сағаттан", "dat": "сағатқа", "loc": "сағатта"},
}  # fmt: skip

# (one, few, many) for nominative/accusative counting, plus the genitive pair (sg, pl).
_RU_UNITS: dict[str, tuple[tuple[str, str, str], tuple[str, str]]] = {
    "day": (("день", "дня", "дней"), ("дня", "дней")),
    "wday": (("рабочий день", "рабочих дня", "рабочих дней"), ("рабочего дня", "рабочих дней")),
    "month": (("месяц", "месяца", "месяцев"), ("месяца", "месяцев")),
    "hour": (("час", "часа", "часов"), ("часа", "часов")),
}
_EN_UNITS = {"day": "day", "wday": "working day", "month": "month", "hour": "hour"}
_KK_UNITS = {"day": "күн", "wday": "жұмыс күні", "month": "ай", "hour": "сағат"}

LABELS: dict[str, dict[str, dict[str, str]]] = {
    "non_returnable": {
        "underwear": {"en": "underwear", "ru": "нижнее бельё", "kk": "іш киім"},
        "opened_cosmetics": {"en": "opened cosmetics", "ru": "вскрытая косметика",
                             "kk": "ашылған косметика"},
        "perishables": {"en": "perishable food", "ru": "скоропортящиеся продукты",
                        "kk": "тез бұзылатын азық-түлік"},
        "gift_cards": {"en": "gift cards", "ru": "подарочные карты", "kk": "сыйлық карталары"},
        "custom_items": {"en": "made-to-order items", "ru": "товары, изготовленные на заказ",
                         "kk": "тапсырыспен жасалған тауарлар"},
    },
}  # fmt: skip
FREE = {"en": "free", "ru": "бесплатно", "kk": "тегін"}
AND = {"en": "and", "ru": "и", "kk": "және"}


class FormatError(ValueError):
    """A template asks for a missing value or an unknown format."""


def group_digits(n: int, lang: str) -> str:
    """12345 -> "12,345" (en) or "12 345" (ru/kk, plain space)."""
    s = f"{n:,}"
    return s if lang == "en" else s.replace(",", " ")


def ru_plural(n: int, forms: tuple[str, str, str]) -> str:
    n100, n10 = abs(n) % 100, abs(n) % 10
    if n10 == 1 and n100 != 11:
        return forms[0]
    if 2 <= n10 <= 4 and not 12 <= n100 <= 14:
        return forms[1]
    return forms[2]


def ru_genitive(n: int, forms: tuple[str, str]) -> str:
    return forms[0] if abs(n) % 10 == 1 and abs(n) % 100 != 11 else forms[1]


def kk_noun(noun: str, case: str | None) -> str:
    return _KK_NOUN[noun][case or "nom"]


def count(n: int, unit: str, lang: str, *, genitive: bool = False, case: str | None = None) -> str:
    """ "3 days" / "3 дня" (or genitive "3 дней") / "3 күн" (+ Kazakh case)."""
    if lang == "en":
        word = _EN_UNITS[unit]
        return f"{n} {word}" + ("" if n == 1 else "s")
    if lang == "ru":
        forms, gen = _RU_UNITS[unit]
        return f"{n} {ru_genitive(n, gen) if genitive else ru_plural(n, forms)}"
    return f"{n} {kk_noun(_KK_UNITS[unit], case)}"


def kzt(n: int, lang: str, case: str | None = None) -> str:
    if lang == "en":
        return f"KZT {group_digits(n, lang)}"
    if lang == "ru":
        return f"{group_digits(n, lang)} тенге"
    return f"{group_digits(n, lang)} {kk_noun('теңге', case)}"


def fmt_date(value: str | date, lang: str, case: str | None = None) -> str:
    d = value if isinstance(value, date) else date.fromisoformat(value)
    month = MONTHS[lang][d.month - 1]
    if lang == "en":
        return f"{d.day} {month}"
    if lang == "kk" and case:
        month = _KK_MONTH_CASES[case][month]
    return f"{d.day} {month}"


def range_days(value: Sequence[int], lang: str, case: str | None = None) -> str:
    lo, hi = int(value[0]), int(value[1])
    if lo == hi:
        return count(lo, "day", lang, case=case)
    dash = "–"
    if lang == "en":
        return f"{lo}{dash}{hi} days"
    if lang == "ru":
        return f"{lo}{dash}{hi} {ru_plural(hi, _RU_UNITS['day'][0])}"
    return f"{lo}{dash}{hi} {kk_noun('күн', case)}"


def join_list(items: Sequence[str], lang: str) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + f" {AND[lang]} " + items[-1]


def localized(value: Any, lang: str) -> str:
    """A `{en:..., ru:..., kk:...}` mapping -> the language's string; strings pass through."""
    if isinstance(value, Mapping):
        if lang not in value:
            raise FormatError(f"no {lang} form in {value!r}")
        return str(value[lang])
    return str(value)


Lookup = Callable[[str, str], Mapping[str, Any]]


def render(
    template: str,
    values: Mapping[str, Any],
    lang: LanguageCode,
    *,
    catalog: Lookup | None = None,
) -> str:
    """Fill `template`. `catalog(kind, id)` resolves ids of cities, carriers, payment methods."""

    def lookup(kind: str, key: str) -> Mapping[str, Any]:
        if catalog is None:
            raise FormatError(f"no catalogue for {kind}")
        return catalog(kind, key)

    def one(m: re.Match[str]) -> str:
        name, fmt, case = m.group(1), m.group(2), m.group(3)
        if case and lang != "kk":
            case = None
        root = name.split(".", 1)[0]
        if root not in values:
            raise FormatError(f"missing value {name!r}")
        v: Any = values[root]
        for part in name.split(".")[1:]:
            v = v[part]
        return _format(v, fmt, lang, case, lookup)

    return PLACEHOLDER_RE.sub(one, template)


def _format(v: Any, fmt: str | None, lang: str, case: str | None, lookup: Lookup) -> str:
    match fmt:
        case None | "store":
            return localized(v, lang)
        case "kzt":
            return kzt(int(v), lang, case)
        case "kzt_free":
            return FREE[lang] if int(v) == 0 else kzt(int(v), lang, case)
        case "days" | "wdays" | "months" | "hours":
            return count(int(v), fmt[:-1], lang, case=case)
        case "days_gen" | "wdays_gen" | "months_gen" | "hours_gen":
            return count(int(v), fmt[: -len("s_gen")], lang, genitive=True, case=case)
        case "range_days":
            return range_days(v, lang, case)
        case "clock_range":
            return f"{v[0]}–{v[1]}"
        case "percent":
            return f"{v}%"
        case "date":
            return fmt_date(v, lang, case)
        case "city":
            city = lookup("city", v["city"] if isinstance(v, Mapping) else str(v))
            return localized(city[lang]["nom"] if lang != "en" else city["en"], lang)
        case "address":
            return localized(v["address"], lang)
        case "sale_name":
            return localized(v, lang)
        case "sale_from":
            return fmt_date(v["from"], lang, case)
        case "sale_to":
            return fmt_date(v["to"], lang, case)
        case "carriers" | "payment_methods":
            kind = fmt[:-1]
            return join_list([localized(lookup(kind, k), lang) for k in v], lang)
        case "non_returnable":
            return join_list([LABELS["non_returnable"][k][lang] for k in v], lang)
        case _:
            raise FormatError(f"unknown format {fmt!r}")

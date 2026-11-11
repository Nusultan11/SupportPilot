"""Text cleaning, artefact detection, PII masking and dedup keys (Stage 3, ADR-020).

Cleaning is conservative: it repairs encoding/whitespace artefacts only. Typos, profanity,
emoji and casing are part of how customers write and are *kept* (Bitext adds typos and
offensive variants on purpose); they are flagged so later stages can sample by register.

Two derived keys drive deduplication:
- `exact_key`: NFKC, casefolded, punctuation-free form. Equal keys = exact duplicates.
- `sim_key`: `exact_key` without profanity tokens. Its character shingles feed MinHash.
"""

from __future__ import annotations

import html
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from supportpilot.common.hashing import sha256_text
from supportpilot.common.ids import normalize_text

ZERO_WIDTH_RE = re.compile("[\u200b-\u200d\u2060\ufeff]")
CONTROL_RE = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
UNICODE_SPACE_RE = re.compile("[\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]")
HTML_ENTITY_RE = re.compile(r"&(?:[a-zA-Z]{2,8}|#\d{2,6}|#x[0-9a-fA-F]{2,5});")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]{0,15}(?:\s[^<>]{0,200})?/?>")
# UTF-8 text decoded as cp1252/latin-1 (e.g. U+00C3 U+00A9 instead of U+00E9)
MOJIBAKE_RE = re.compile(
    "(?:\u00c3[\u0080-\u00bf]|\u00e2\u20ac|\u00d0[\u0080-\u00bf]"
    "|\u00d1[\u0080-\u00bf]|\u00c2[\u00a0-\u00bf])"
)
URL_RE = re.compile(r"\bhttps?://\S+|\bwww\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}(?![\w-])")
PHONE_RE = re.compile(
    r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{3}[\s.-]\d{2,4}[\s.-]\d{2,4}"
    r"(?:[\s.-]\d{2,4})?(?!\w)"
)
CARD_RE = re.compile(r"(?<![\w#])(?:\d[ -]?){12,18}\d(?!\w)")
REPEAT_RE = re.compile(r"(.)\1{4,}")
EMOJI_RE = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff]")
WORD_RE = re.compile(r"\w+", re.UNICODE)

# Offensive words seen in the sources (Bitext W-tag variants) plus common RU forms.
# Used only for flags and to make the similarity key ignore inserted profanity.
OFFENSIVE_RE = re.compile(
    r"\b(?:f+u+c*k+\w*|fuk\w*|fck\w*|motherf\w*|shit\w*|bullshit|damn\w*|goddamn\w*|bloody|crap\w*|"
    r"bitch\w*|asshole\w*|wtf|бля\w*|сука|суки|хрен\w*|хуй\w*|нахуй|пизд\w*|еба\w*|ёба\w*)\b",  # noqa: RUF001
    re.IGNORECASE,
)

_QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u00ab": '"',
        "\u00bb": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u2026": "...",
    }
)
_NON_WORD_RE = re.compile(r"[^\w\s<>]", re.UNICODE)
_SPACES_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class Cleaned:
    text: str
    fixes: tuple[str, ...] = ()


def clean_text(raw: str) -> Cleaned:
    """Repair artefacts; keep content. Returns the text and the list of repairs applied."""
    fixes: list[str] = []
    t = raw
    if "\r" in t:
        fixes.append("fix:crlf")
    if ZERO_WIDTH_RE.search(t):
        t = ZERO_WIDTH_RE.sub("", t)
        fixes.append("fix:zero_width")
    if CONTROL_RE.search(t):
        t = CONTROL_RE.sub(" ", t)
        fixes.append("fix:control_chars")
    if UNICODE_SPACE_RE.search(t):
        t = UNICODE_SPACE_RE.sub(" ", t)
        fixes.append("fix:unicode_spaces")
    if HTML_ENTITY_RE.search(t):
        t = html.unescape(t)
        fixes.append("fix:html_entities")
    if unicodedata.normalize("NFC", t) != t:
        fixes.append("fix:nfc")
    out = normalize_text(t)
    if out != t.replace("\r\n", "\n").replace("\r", "\n") and "fix:whitespace" not in fixes:
        fixes.append("fix:whitespace")
    return Cleaned(out, tuple(fixes))


def levenshtein(a: str, b: str, limit: int = 3) -> int:
    """Edit distance, stopping early once it exceeds `limit`."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if min(cur) > limit:
            return limit + 1
        prev = cur
    return prev[-1]


def _fuzzy_offensive(word: str) -> bool:
    """Typo variants of the most frequent inserted word: fuckijg, fvucking, fukcing, ufcking."""
    w = word.casefold()
    if not (3 <= len(w) <= 10) or not (w.startswith("uf") or (w[0] == "f" and "u" in w[1:3])):
        return False
    return levenshtein(w, "fucking", 2) <= 2 if len(w) >= 6 else levenshtein(w, "fuck", 1) <= 1


def is_offensive(text: str) -> bool:
    return bool(OFFENSIVE_RE.search(text)) or any(
        _fuzzy_offensive(w) for w in WORD_RE.findall(text)
    )


def scripts(text: str) -> Counter[str]:
    """Letters per Unicode script (first word of the character name: LATIN, CYRILLIC, CJK...)."""
    out: Counter[str] = Counter()
    for ch in text:
        if ch.isalpha():
            out[unicodedata.name(ch, "UNKNOWN").split()[0]] += 1
    return out


def garbled(text: str, *, min_letters: int = 20, min_share: float = 0.03) -> bool:
    """Token soup from a failed generation: >= 3 scripts, each >= 3% of >= 20 letters."""
    counts = scripts(text)
    total = sum(counts.values())
    if total < min_letters:
        return False
    return sum(1 for v in counts.values() if v / total >= min_share) >= 3


def has_letters(text: str) -> bool:
    return any(ch.isalpha() for ch in text)


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def artefact_flags(text: str) -> list[str]:
    """Flags describing the cleaned text (nothing is removed here)."""
    flags: list[str] = []
    if HTML_TAG_RE.search(text):
        flags.append("html_tag")
    if MOJIBAKE_RE.search(text):
        flags.append("mojibake")
    if "\ufffd" in text:
        flags.append("replacement_char")
    if URL_RE.search(text):
        flags.append("url")
    if REPEAT_RE.search(text):
        flags.append("repeated_chars")
    if EMOJI_RE.search(text):
        flags.append("emoji")
    if is_offensive(text):
        flags.append("offensive")
    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 10 and sum(c.isupper() for c in letters) / len(letters) >= 0.7:
        flags.append("all_caps")
    n_words = word_count(text)
    if n_words <= 2:
        flags.append("short_text")
    if len(text) > 2000:
        flags.append("long_text")
    if "{" in text.replace("{{", "").replace("}}", "") or "}" in text.replace("}}", ""):
        flags.append("brace_artifact")
    return flags


@dataclass
class PiiMasker:
    """Replaces e-mails, phone numbers and card-like numbers with synthetic values.

    Values are derived from a hash of the original, so the same input always gets the same
    replacement, and they use reserved/test ranges: example.com, +1 202 555 01xx, and
    Luhn-valid test card numbers starting with 4000.
    """

    found: list[str] = field(default_factory=list)

    def _email(self, m: re.Match[str]) -> str:
        self.found.append("pii:email")
        return f"customer{int(sha256_text(m.group(0))[:6], 16) % 10000:04d}@example.com"

    def _phone(self, m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) < 10 or len(digits) > 13:
            return m.group(0)
        self.found.append("pii:phone")
        return f"+1 202 555 01{int(sha256_text(m.group(0))[:4], 16) % 100:02d}"

    def _card(self, m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if not (13 <= len(digits) <= 19 and luhn_ok(digits)):
            return m.group(0)
        self.found.append("pii:card")
        return "4000 0000 0000 0002"

    def mask(self, text: str) -> str:
        text = EMAIL_RE.sub(self._email, text)
        text = CARD_RE.sub(self._card, text)
        return PHONE_RE.sub(self._phone, text)


def fold(text: str) -> str:
    """NFKC + casefold + unified quotes/dashes (comparison form, never stored as text)."""
    return unicodedata.normalize("NFKC", text).translate(_QUOTES).casefold()


def exact_key(text: str) -> str:
    """Punctuation-free, case-free form. `<slot>` markers survive (see placeholders.key_form)."""
    return _SPACES_RE.sub(" ", _NON_WORD_RE.sub(" ", fold(text))).strip()


def sim_key(text: str) -> str:
    """`exact_key` without profanity: inserted swear words do not make a new utterance."""
    k = exact_key(text)
    words = [w for w in OFFENSIVE_RE.sub(" ", k).split() if not _fuzzy_offensive(w)]
    return " ".join(words)

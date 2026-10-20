"""SA-V prompts: an SVG2 scene graph -> one fluent sentence naming every object.

Each tracked object gets a clause ending in its ``<traj_i>`` token, and the
returned metadata records, per object, the word index of its head noun (where
its appearance token goes) and of its ``<traj_i>`` token.

Conventions:
  - One clause per selected object.
  - <traj_i> placed at the end of the clause as a motion descriptor
    (replaces a directional adverbial / "across", "around", "left", etc.).
  - The head noun of the object's clause (found backwards from <traj_i>,
    else forwards) is where its appearance token goes.
  - Verbs come from the SVG2 relation's predicate (already verb-shaped).
  - Animate subjects don't take colors as prenominal adjectives. The
    color (or wearing attribute) becomes a postnominal "in a <color>
    <clothing>" phrase.

Sentence shape for one clip: the action clauses (objects with a relation),
joined by ", while", then the context clauses (the others), after ", with"
(one) or ", amid" (several), the last after ", and"::

  "<action_0>, while <action_1>, amid <context_0>, <context_1>, and <context_2>."

Clause shapes:
  Action subject:  "a <attr-phrase> <noun> <verb_phrase_with_dst> <traj_i>"
                  e.g. "a person in red walks alongside another man <traj_0>"
  Plain object:    "a <attr-phrase> <noun> <traj_i>"
                  e.g. "a metallic pole <traj_5>"
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, NamedTuple

from trajloc.preprocess.stage1.prompt_words import locate_object_words, starts_with_vowel
from trajloc.preprocess.stage1.records import PromptSynthesis
from trajloc.preprocess.stage1.vocabulary import ANIMATE_LABELS

# --------------------------------------------------------------------- Animacy


def _is_animate(label: str) -> bool:
    return label.strip().lower() in ANIMATE_LABELS


# Words that appear inside SVG2 attributes and indicate clothing — when
# present we phrase animates as 'a person in a <color> jacket' instead
# of 'a vibrant red person'.
CLOTHING_NOUNS = frozenset(
    {
        "shirt",
        "t-shirt",
        "tshirt",
        "blouse",
        "polo shirt",
        "tank top",
        "jacket",
        "puffer jacket",
        "suit jacket",
        "coat",
        "trench coat",
        "sweater",
        "sweatshirt",
        "hoodie",
        "cardigan",
        "trousers",
        "pants",
        "jeans",
        "shorts",
        "leggings",
        "skirt",
        "dress",
        "gown",
        "robe",
        "saree",
        "sari",
        "kimono",
        "uniform",
        "vest",
        "scarf",
        "tie",
        "belt",
        "jersey",
        "cycling jersey",
        "swimsuit",
        "hat",
        "cap",
        "helmet",
        "headscarf",
        "headband",
        "headwear",
        "gloves",
        "shoes",
        "boots",
        "sneakers",
        "sandals",
        "slippers",
    }
)

# Plural / mass-noun clothing items that don't take an indefinite article.
NO_ARTICLE_CLOTHING = frozenset(
    {
        "trousers",
        "pants",
        "jeans",
        "shorts",
        "leggings",
        "gloves",
        "shoes",
        "boots",
        "sneakers",
        "sandals",
        "slippers",
    }
)


# --------------------------------------------------------- Attribute filtering

ATTR_BLOCKLIST = frozenset(
    {
        "modern",
        "classic",
        "vintage",
        "natural",
        "casual",
        "professional",
        "subtle",
        "subtle sheen",
        "subtle-textured",
        "smooth",
        "uniform",
        "consistent",
        "consistent height",
        "consistent-width",
        "consistent-pattern",
        "utilitarian design",
        "unadorned",
        "varied-textured",
        "varying-shaded",
        "richly-textured",
        "rich-textured",
        "differently sized and shaped leaves",
        "muted gray",
        "darker patches",
        "intricate pattern",
        "well-trimmed",
        "robust",
        "durable",
        "weather-resistant",
        "expansive",
        "distinct",
        "healthy",
        "thick-canopied",
        "thick",
        "dense",
        "rich green",
        "vibrant",
        "lush",
        "verdant",
        "bold",
        "clear view",
        "enclosed cargo area",
        "compact shape",
        "boxy cab",
        "round headlights",
        "warm glow",
        "slightly rounded front",
        "small windshield",
        "long-sleeved",
        "high-collared",
        "padded",
        "voluminous",
        "puffed",
        "slightly puffed",
        "vertical-sided",
        "flat-topped",
        "long",
        "narrow",
        "tall",
        "short",
        "wide",
        "small",
        "large",
        "medium",
        "elongated",
        "rectangular",
    }
)

EXTRA_BLOCKLIST_TOKENS = frozenset(
    {
        "numerous",
        "small",
        "oval",
        "rectangular",
        "elongated",
        "compact",
        "contrasting",
        "sleek",
        "varying",
        "varied",
        "uniform",
        "consistent",
        "subtle",
        "smooth",
        "rough",
        "soft",
        "hard",
    }
)

COLOR_RE = re.compile(
    r"\b(black|white|gray|grey|yellow|red|blue|green|brown|orange|purple|pink|"
    r"silver|gold|chrome|tan|beige|navy|maroon|crimson|scarlet|amber|teal|"
    r"cyan|violet|magenta|copper|bronze|cream|ivory|charcoal|olive|"
    r"yellow-and-green|yellow and green|red-and-white|black-and-white)\b",
    re.IGNORECASE,
)
MATERIAL_RE = re.compile(
    r"\b(metallic|wooden|concrete|stone|brick|plastic|glass|fabric|leather|"
    r"metal|steel|aluminum|marble|asphalt|paper|cardboard|rubber|denim|"
    r"velvet|canvas|silk)\b",
    re.IGNORECASE,
)


def _attr_compatible_with_label(attr: str, label: str) -> bool:
    """Whether ``attr`` adds to ``label``: non-empty, and sharing no word with it."""
    attr_lc = attr.lower().strip()
    label_lc = label.lower().strip()
    if not attr_lc:
        return False
    if attr_lc == label_lc or attr_lc in label_lc or label_lc in attr_lc:
        return False
    label_words = set(label_lc.split())
    return not any(word in label_words for word in attr_lc.split())


def _strip_color(attr: str) -> str | None:
    """The bare color word of a (possibly compound) attribute, or ``None``."""
    match = COLOR_RE.search(attr)
    return match.group(0).lower() if match else None


# Iterate CLOTHING_NOUNS in deterministic, longest-first order so multi-
# word items like "puffer jacket" match before "jacket". Without this, set
# iteration order is hash-randomized across processes (PYTHONHASHSEED) and
# the prompt text drifts run-to-run.
_CLOTHING_NOUNS_ORDERED = sorted(CLOTHING_NOUNS, key=lambda s: (-len(s), s))


def _attr_contains_clothing(attr: str) -> str | None:
    """The (longest) clothing noun inside ``attr``, or ``None``."""
    attr_lc = attr.lower()
    return next((clothing for clothing in _CLOTHING_NOUNS_ORDERED if clothing in attr_lc), None)


def _usable_attribute(attr_lc: str, label: str) -> bool:
    """Whether a (lowercased) attribute may describe an inanimate ``label``:
    non-empty, not blocklisted, no blocklisted word, compatible with the label."""
    if not attr_lc or attr_lc in ATTR_BLOCKLIST:
        return False
    if any(word in EXTRA_BLOCKLIST_TOKENS for word in attr_lc.split()):
        return False
    return _attr_compatible_with_label(attr_lc, label)


MAX_FALLBACK_ATTRIBUTE_WORDS = 3  # an attribute naming no color or material must be this short


def _pick_attribute_for_inanimate(attrs: list[str], label: str) -> str:
    """The first usable attribute that names a color, else a material, else
    the first short one ('' if none)."""
    usable = [a.lower().strip() for a in attrs]
    usable = [attr_lc for attr_lc in usable if _usable_attribute(attr_lc, label)]
    for hint_re in (COLOR_RE, MATERIAL_RE):
        for attr_lc in usable:
            if hint_re.search(attr_lc):
                return attr_lc
    return next(
        (a for a in usable if len(a.split()) <= MAX_FALLBACK_ATTRIBUTE_WORDS),
        "",
    )


def _wearing_phrase(clothing: str, color: str | None) -> str:
    """'in a <color> <clothing>' (no article for plural or mass clothing)."""
    article = "" if clothing in NO_ARTICLE_CLOTHING else "a "
    return f"in {article}{color} {clothing}" if color else f"in {article}{clothing}"


def _pick_animate_phrasing(attrs: list[str]) -> str:
    """An animate subject's postnominal phrase ('' if none): the first
    attribute naming a clothing item, as 'in a <color> <clothing>' (its color
    if it names one: 'red shirt' -> 'in a red shirt'), else the first color,
    as 'in <color>'."""
    attrs_lc = [attr_lc for attr_lc in (a.lower().strip() for a in attrs) if attr_lc]
    # 1. An attribute naming a clothing item (and, preferably, its color).
    for attr_lc in attrs_lc:
        clothing = _attr_contains_clothing(attr_lc)
        if clothing:
            return _wearing_phrase(clothing, _strip_color(attr_lc))
    # 2. A color.
    color = next((c for c in map(_strip_color, attrs_lc) if c), None)
    return f"in {color}" if color else ""


# ------------------------------------------------------- Relationship cleaning


def _is_camera_predicate(pred: str) -> bool:
    pred_lc = pred.lower()
    # "doll": dolly, dollies, dollying.
    return ("camera" in pred_lc) or pred_lc.startswith(("pan", "zoom", "tilt", "orbit", "doll"))


# Image-space self motion, relative to the camera: the trajectory encodes it.
CAMERA_MOTION_PREDICATES = frozenset(
    {
        "approaches camera",
        "moves toward camera",
        "moves away from camera",
        "moves closer to camera",
        "passes camera",
        "passes by camera",
        "drives past camera",
        "walks past camera",
        "rides past camera",
        "approaches and passes camera",
        "walks toward camera",
        "drives toward camera",
        "moves past camera",
        "moves out of view",
        "moves into view",
        "leaves view",
        "leaves view of",
        "enters view",
        "moves left in view",
        "moves right in view",
        "moves left",
        "moves right",
        "moves leftward",
        "moves rightward",
        "moves up",
        "moves down",
        "moves upward",
        "moves downward",
        "moves left relative to",
        "moves right relative to",
        "moves leftward relative to",
        "moves rightward relative to",
    }
)


def _is_camera_relatum_motion(pred: str) -> bool:
    """Whether ``pred`` is image-space self motion relative to the camera
    (dropped: the trajectory token encodes it)."""
    return pred.lower().strip() in CAMERA_MOTION_PREDICATES


# Stripped from predicates to compare them (``moves left of`` == ``moves right of``).
DIRECTION_TOKENS = (
    "right",
    "left",
    "rightward",
    "leftward",
    "upward",
    "downward",
    "forward",
    "backward",
)


class _Relation(NamedTuple):
    """An SVG2 relation between selected objects (``dst`` ``None`` or -1: none)."""

    src: int
    pred: str
    dst: int | None
    kind: str


DROPPED_RELATION_KINDS = ("spatial", "attentional")


def _clean_relationships(rels: list[Any], selected: set[int]) -> list[_Relation]:
    """The relations usable in the prompt, as ``(src, pred, dst, kind)``.

    Dropped: relations of unselected objects, camera predicates and
    image-space motion relative to the camera (the trajectory encodes it),
    spatial and attentional relations, and duplicates up to direction words
    (``moves left relative to`` and ``moves right relative to`` are one).
    The kept predicate is the original one.
    """
    seen: set[tuple[int, int | None, str]] = set()
    out: list[_Relation] = []
    for r in rels:
        if not isinstance(r, list) or len(r) < 5:
            continue
        src, pred, dst, _, kind = r[:5]
        if not isinstance(src, int) or src not in selected:
            continue
        if not isinstance(pred, str):
            continue
        if _is_camera_predicate(pred):
            continue
        if _is_camera_relatum_motion(pred):
            continue
        if dst != -1 and dst is not None:
            if not isinstance(dst, int) or dst not in selected:
                continue
        if kind in DROPPED_RELATION_KINDS:
            continue
        # Direction-strip into a canonical form for dedup
        canon_pred = pred.lower()
        for tok in DIRECTION_TOKENS:
            canon_pred = canon_pred.replace(" " + tok + " ", " ")
            if canon_pred.endswith(" " + tok):
                canon_pred = canon_pred[: -(len(tok) + 1)]
        canon_pred = re.sub(r"\s+", " ", canon_pred).strip()
        key = (src, dst, canon_pred)
        if key in seen:
            continue
        seen.add(key)
        # Keep the original predicate (not the direction-stripped one).
        out.append(_Relation(src, pred, dst, kind))
    return out


# A subject's relation is its highest-priority one (lowest number).
KIND_PRIORITY = {"functional": 0, "social": 1, "event_level": 2, "motion": 3, "stateful": 4}
UNKNOWN_KIND_PRIORITY = 9


# ----------------------------------------------------------- Sentence assembly


def _article(noun_phrase: str, repeat: bool = False) -> str:
    if repeat:
        return "another"
    return "an" if starts_with_vowel(noun_phrase) else "a"


class _SubjectPhrase(NamedTuple):
    prenominal: str  # '<article> [<attribute>] <noun>'
    postnominal: str  # 'in a red shirt' (animates), else ''


def _build_subject_phrase(
    label: str, attrs_for_obj: list[str], *, is_repeat: bool, is_animate: bool
) -> _SubjectPhrase:
    """The object's subject phrase. Attributes are chosen per object: two
    objects with the same label may get the same one."""
    if is_animate:
        return _SubjectPhrase(
            f"{_article(label, repeat=is_repeat)} {label}", _pick_animate_phrasing(attrs_for_obj)
        )
    attr = _pick_attribute_for_inanimate(attrs_for_obj, label)
    if attr:
        return _SubjectPhrase(f"{_article(attr, repeat=is_repeat)} {attr} {label}", "")
    return _SubjectPhrase(f"{_article(label, repeat=is_repeat)} {label}", "")


# Bare prepositions occasionally appear as stateful preds in SVG2 (e.g.
# 'on', 'in', 'against'). When we see them we promote to a verb form so the
# clause has a finite verb.
BARE_PREPOSITION_PREDS = frozenset(
    {
        "on",
        "in",
        "at",
        "by",
        "near",
        "against",
        "behind",
        "above",
        "below",
        "beside",
        "next to",
        "under",
        "over",
        "inside",
        "within",
        "alongside",
    }
)

# Common gerund -> 3rd person singular present tense substitutions. We
# normalize the verb form so each clause reads as a finite-tense sentence.
GERUND_TO_PRESENT = {
    "holding": "holds",
    "carrying": "carries",
    "walking": "walks",
    "running": "runs",
    "moving": "moves",
    "approaching": "approaches",
    "pushing": "pushes",
    "pulling": "pulls",
    "riding": "rides",
    "driving": "drives",
    "watching": "watches",
    "looking": "looks",
    "talking": "talks",
    "dancing": "dances",
    "interacting": "interacts",
    "passing": "passes",
    "playing": "plays",
    "sitting": "sits",
    "standing": "stands",
    "lying": "lies",
    "leaning": "leans",
    "browsing": "browses",
    "exploring": "explores",
    "investigating": "investigates",
    "wearing": "wears",
    "operating": "operates",
    "manipulating": "manipulates",
    "handling": "handles",
    "touching": "touches",
    "petting": "pets",
    "feeding": "feeds",
    "stirring": "stirs",
    "pouring": "pours",
    "opening": "opens",
    "closing": "closes",
    "lifting": "lifts",
    "throwing": "throws",
    "kicking": "kicks",
    "hitting": "hits",
    "grabbing": "grabs",
    "dropping": "drops",
    "stepping": "steps",
    "covering": "covers",
    "uncovering": "uncovers",
    "wrapping": "wraps",
    "performing": "performs",
    "facing": "faces",
    "fitting": "fits",
    "resting": "rests",
    "supporting": "supports",
    "hanging": "hangs",
    "swimming": "swims",
    "climbing": "climbs",
    "jumping": "jumps",
    "flying": "flies",
    "shopping": "shops",
}


# Predicate fragments found in SVG2's relations and the verb phrase used
# instead. Identity entries matter too: they normalize case and whitespace.
FRAGMENT_REWRITES = {
    "moves relative to": "moves alongside",
    "moves toward": "moves toward",
    "moves away from": "moves away from",
    "moves closer to": "moves closer to",
    "moves past": "moves past",
    "moves into view of": "comes into view of",
    "moves out of view of": "leaves view of",
}


def _normalize_verb(pred: str) -> str:
    """Replace gerund (-ing) forms with present-tense 3SG, and rewrite
    predicate fragments into clean verb phrases."""
    pred_lc = pred.lower().strip()
    if pred_lc in FRAGMENT_REWRITES:
        pred = FRAGMENT_REWRITES[pred_lc]

    words = pred.split()
    if not words:
        return pred
    head = words[0].lower()
    if head in GERUND_TO_PRESENT:
        words[0] = GERUND_TO_PRESENT[head]
        return " ".join(words)
    return pred


def _dst_in_pred(dst_label: str, pred: str) -> bool:
    """Whether the destination's label (possibly several words, like 'stone
    wall') appears in the predicate as whole words."""
    if not dst_label:
        return False
    return f" {dst_label.lower()} " in f" {pred.lower()} "


def _modifier_for_relation(
    relation: _Relation, label_of: dict[int, str], *, dst_already_mentioned: bool
) -> str:
    """The suffix that follows the noun: the verb and a reference to dst. It
    comes before <traj_i>.

    Each clause is a full finite-tense simple sentence; the caller decides
    how clauses are joined ('while' or ','), so none is prepended here.

    dst_already_mentioned: the dst object's clause comes earlier. When dst
    has src's label it is then 'the other one' (else 'another one');
    otherwise it is always 'the <label>'.
    """
    src, pred, dst, kind = relation
    pred = _normalize_verb(pred)
    # A bare-preposition stateful predicate gets a verb: 'on' -> 'is on'.
    if kind == "stateful" and pred.lower().strip() in BARE_PREPOSITION_PREDS:
        pred = "is " + pred

    if dst == -1 or dst is None:
        return f" {pred}"
    dst_label = (label_of.get(dst) or "object").lower()
    # Avoid duplicating dst noun if predicate already mentions it
    if _dst_in_pred(dst_label, pred):
        return f" {pred}"
    same_label = dst_label == label_of.get(src, "")
    # When src and dst share their noun, naming dst with it would put dst's
    # word closer to <traj_src> than src's, and the backward search for
    # <traj_src>'s appearance anchor would find the wrong one: a pronoun-like
    # phrase that doesn't repeat the noun is used instead.
    if same_label:
        if dst_already_mentioned:
            return f" {pred} the other one"
        return f" {pred} another one"
    return f" {pred} the {dst_label}"


def build_scene_graph_prompt(
    labels: list[str | None],
    rels: list[Any],
    obj_attrs: list[list[str]],
    selected_indices: list[int],
) -> PromptSynthesis:
    """The prompt naming ``selected_indices`` (scene-graph object indices;
    object ``i`` of the list is ``<traj_i>``), with each object's word indices."""
    selected = list(selected_indices)
    object_index_of = {source_id: i for i, source_id in enumerate(selected)}
    label_of = {i: (labels[i] or "object").lower().strip() for i in selected}

    relation_of = _best_relations(rels, set(selected))
    # Objects with a relation (action clauses) are named first, so the first
    # mention of a label gets 'a'/'an' and repeats get 'another'; the rest
    # (context clauses) follow.
    emission_order = [o for o in selected if o in relation_of] + [
        o for o in selected if o not in relation_of
    ]
    modifiers = _relation_modifiers(relation_of, emission_order, label_of)
    clauses = _clauses(emission_order, object_index_of, label_of, obj_attrs, modifiers)

    prompt = _assemble_sentence(clauses)
    obj_meta = _locate_objects(prompt, clauses, selected)

    errors = []
    for m in obj_meta:
        if m["trajectory_token_word_index"] is None:
            errors.append(f"<traj_{m['object_index']}> not found in prompt")
        if m["noun_word_index"] is None:
            errors.append(f"noun {m['noun']!r} not recoverable for object {m['object_index']}")

    # Sorted by object_index, so objects[i].object_index == i (clauses are emitted
    # in sentence order, which differs when action and context clauses mix).
    obj_meta.sort(key=lambda m: m["object_index"])

    return PromptSynthesis(prompt=prompt, objects=obj_meta, ok=len(errors) == 0, errors=errors)


def _best_relations(rels: list[Any], selected: set[int]) -> dict[int, _Relation]:
    """Per subject, its highest-priority usable relation (``KIND_PRIORITY``;
    the first on ties)."""
    relations = sorted(
        _clean_relationships(rels, selected),
        key=lambda r: KIND_PRIORITY.get(r.kind, UNKNOWN_KIND_PRIORITY),
    )
    relation_of: dict[int, _Relation] = {}
    for relation in relations:
        relation_of.setdefault(relation.src, relation)
    return relation_of


def _relation_modifiers(
    relation_of: dict[int, _Relation], emission_order: list[int], label_of: dict[int, str]
) -> dict[int, str]:
    """Per subject, its relation's text; the destination is 'the ...' when its
    own clause comes first."""
    position = {source_id: k for k, source_id in enumerate(emission_order)}
    modifiers = {}
    for src, relation in relation_of.items():
        dst = relation.dst
        dst_already = (
            dst is not None and dst != -1 and dst in position and position[dst] < position[src]
        )
        modifiers[src] = _modifier_for_relation(
            relation, label_of, dst_already_mentioned=dst_already
        )
    return modifiers


@dataclass(frozen=True)
class _Clause:
    """One object's clause: ``<article + attributes + noun [+ modifier]> <traj_i>``."""

    text: str
    noun: str
    object_index: int
    modifier: str  # its relation's text ('' for a context clause)


def _clauses(
    emission_order: list[int],
    object_index_of: dict[int, int],
    label_of: dict[int, str],
    obj_attrs: list[list[str]],
    modifiers: dict[int, str],
) -> list[_Clause]:
    """The objects' clauses, in emission order (the second object of a label
    is 'another <label>')."""
    seen_label_count: defaultdict[str, int] = defaultdict(int)
    clauses = []
    for source_id in emission_order:
        object_index = object_index_of[source_id]
        label = label_of[source_id]
        seen_label_count[label] += 1
        subject = _build_subject_phrase(
            label=label,
            attrs_for_obj=obj_attrs[source_id],
            is_repeat=seen_label_count[label] > 1,
            is_animate=_is_animate(label),
        )
        modifier = modifiers.get(source_id, "")
        # <traj_i> always ends the clause, as its motion descriptor. The
        # modifier is inlined without its leading space; the recorded
        # ``modifier`` keeps it.
        inline_modifier = modifier.lstrip(", ")
        inline_modifier = (" " + inline_modifier) if inline_modifier else ""
        postnominal = f" {subject.postnominal}" if subject.postnominal else ""
        text = f"{subject.prenominal}{postnominal}{inline_modifier} <traj_{object_index}>"
        clauses.append(_Clause(text, label, object_index, modifier))
    return clauses


def _assemble_sentence(clauses: list[_Clause]) -> str:
    """The sentence: the action clauses (with a modifier) joined by ', while',
    then the context clauses after ', with' (one) or ', amid' (several), the
    last after ', and'; punctuation tidied, first letter capitalized."""
    action_clauses = [c.text for c in clauses if c.modifier]
    context_clauses = [c.text for c in clauses if not c.modifier]

    pieces = []
    if action_clauses:
        pieces.append(action_clauses[0])
        pieces += [", while " + text for text in action_clauses[1:]]
    if context_clauses:
        connector = ""
        if action_clauses:
            connector = ", amid " if len(context_clauses) > 1 else ", with "
        pieces.append(connector + context_clauses[0])
        pieces += [", " + text for text in context_clauses[1:-1]]
        if len(context_clauses) > 1:
            pieces.append(", and " + context_clauses[-1])

    prompt = " ".join(pieces).rstrip(",") + "."
    prompt = re.sub(r"\s+,", ",", prompt)
    prompt = re.sub(r"\s+\.", ".", prompt)
    prompt = re.sub(r",\s*\.", ".", prompt)
    prompt = re.sub(r"\s+", " ", prompt).strip()
    if prompt:  # it starts with 'a' / 'an' / 'another'
        prompt = prompt[0].upper() + prompt[1:]
    return prompt


def _locate_objects(
    prompt: str, clauses: list[_Clause], selected: list[int]
) -> list[dict[str, Any]]:
    """Per clause, its object's prompt record, with the word indices of its
    ``<traj_i>`` and head noun (:func:`~..prompt_words.locate_object_words`,
    searching forwards too)."""
    words = prompt.split()
    obj_meta = []
    for c in clauses:
        trajectory_token_word_index, noun_word_index = locate_object_words(
            words, c.object_index, c.noun, search_forward=True
        )
        obj_meta.append(
            {
                "object_index": c.object_index,
                "source_object_id": selected[c.object_index],
                "noun": c.noun,
                "modifier": c.modifier,
                "noun_word_index": noun_word_index,
                "trajectory_token_word_index": trajectory_token_word_index,
            }
        )
    return obj_meta

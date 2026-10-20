"""Label vocabularies shared by the Stage-1 prompt builders."""

from __future__ import annotations

# Labels (lower case) of living things: an SA-V object with one of these is
# described as animate ("a man in a red shirt", not "a red man").
ANIMATE_LABELS = frozenset(
    {
        "person",
        "man",
        "woman",
        "child",
        "boy",
        "girl",
        "kid",
        "baby",
        "individual",
        "people",
        "human",
        "dog",
        "cat",
        "horse",
        "cow",
        "sheep",
        "lamb",
        "bird",
        "fish",
        "duck",
        "chicken",
        "monkey",
        "rat",
        "hamster",
        "eagle",
        "parakeet",
        "judoka",
        "dancer",
        "cyclist",
        "biker",
        "athlete",
        "runner",
        "shopper",
        "pedestrian",
        "soldier",
        "monk",
        "tourist",
        "schoolgirl",
        "schoolboy",
        "rider",
    }
)

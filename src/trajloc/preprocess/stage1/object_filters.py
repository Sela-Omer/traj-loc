"""Object filters shared by every Stage-1 builder.

An object is kept only if its largest mask (over the clip's frames, at clip
size) is at least ``MIN_PEAK_AREA_PX`` and at most ``MAX_PEAK_AREA_FRACTION`` of
the frame, and it is visible on at least ``MIN_VISIBLE_FRAMES`` frames.
"""

from __future__ import annotations

from enum import Enum

MAX_PEAK_AREA_FRACTION = 0.70  # peak area above 70% of the frame: a full-frame artefact
MIN_PEAK_AREA_PX = 800  # peak area below 800 px, at any resolution
MIN_VISIBLE_FRAMES = 8


class FilterFailure(Enum):
    """The shared filter an object fails (checked in this order)."""

    TINY = "tiny"
    HUGE = "huge"
    LOW_VISIBILITY = "low_visibility"


def failed_filter(peak: int, n_visible: int, H: int | None, W: int | None) -> FilterFailure | None:
    """The filter an object with peak mask area ``peak`` (pixels), visible on
    ``n_visible`` frames, fails (``None`` if it passes all). Without a frame
    size there is no upper area bound."""
    if peak < MIN_PEAK_AREA_PX:
        return FilterFailure.TINY
    if H is not None and W is not None and peak > MAX_PEAK_AREA_FRACTION * H * W:
        return FilterFailure.HUGE
    if n_visible < MIN_VISIBLE_FRAMES:
        return FilterFailure.LOW_VISIBILITY
    return None

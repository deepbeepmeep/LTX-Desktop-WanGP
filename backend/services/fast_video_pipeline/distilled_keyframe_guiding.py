"""Per-call Distilled helper swap for multi-keyframe generate.

DistilledPipeline hardcodes ``combined_image_conditionings`` (frame 0 replaces
the latent). MKF interpolation needs ``image_conditionings_by_adding_guiding_latent``
instead. This is scoped to one Fast generate so first/last i2v keeps replace-at-0.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_SWAP_LOCK = threading.Lock()


@contextmanager
def distilled_keyframe_guiding() -> Iterator[None]:
    import ltx_pipelines.distilled as distilled
    from ltx_pipelines.utils.helpers import image_conditionings_by_adding_guiding_latent

    with _SWAP_LOCK:
        original = distilled.combined_image_conditionings
        distilled.combined_image_conditionings = image_conditionings_by_adding_guiding_latent
        try:
            yield
        finally:
            distilled.combined_image_conditionings = original

from __future__ import annotations

import ctypes
import gc
import os
import sys
from collections.abc import Mapping
from typing import Any

from app.services.buffalo_face_analysis import (
    get_buffalo_face_analyzer,
    release_buffalo_face_analyzer,
)
from app.services.face_analysis import (
    get_face_analyzer,
    release_face_analyzer,
)
from app.services.face_swap import (
    get_face_swapper,
    release_face_swapper,
)


def _current_rss_mb() -> float | None:
    try:
        with open(
            "/proc/self/status",
            "r",
            encoding="utf-8",
        ) as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None

    return None


def _clear_object_session(value: Any) -> None:
    for attribute_name in (
        "session",
        "_session",
        "sess",
    ):
        if not hasattr(value, attribute_name):
            continue

        try:
            setattr(
                value,
                attribute_name,
                None,
            )
        except Exception:
            pass


def _dispose_face_analysis(value: Any) -> None:
    models = getattr(
        value,
        "models",
        None,
    )

    if isinstance(models, Mapping):
        for model in list(models.values()):
            _clear_object_session(model)

        try:
            models.clear()
        except Exception:
            pass

    _clear_object_session(value)


def _dispose_cached_face_analyzers() -> None:
    if get_face_analyzer.cache_info().currsize:
        try:
            _dispose_face_analysis(
                get_face_analyzer()
            )
        except Exception:
            pass

    release_face_analyzer()

    if get_buffalo_face_analyzer.cache_info().currsize:
        try:
            _dispose_face_analysis(
                get_buffalo_face_analyzer()
            )
        except Exception:
            pass

    release_buffalo_face_analyzer()


def _dispose_cached_face_swapper() -> None:
    if get_face_swapper.cache_info().currsize:
        try:
            swapper = get_face_swapper()
            _clear_object_session(swapper)

            if hasattr(swapper, "emap"):
                try:
                    swapper.emap = None
                except Exception:
                    pass
        except Exception:
            pass

    release_face_swapper()


def _release_loaded_torch_cuda_cache() -> None:
    torch_module = sys.modules.get("torch")

    if torch_module is None:
        return

    try:
        cuda = torch_module.cuda

        if cuda.is_available():
            cuda.empty_cache()
            cuda.ipc_collect()
    except Exception:
        pass


def _malloc_trim() -> None:
    try:
        ctypes.CDLL(
            "libc.so.6"
        ).malloc_trim(0)
    except Exception:
        pass


def release_worker_runtime_memory(
    *,
    stage: str | None = None,
) -> None:
    """Release every heavy model cached in the Celery task process.

    ConsistentID and BiRefNet run in separate process groups. This function
    is responsible for the CPU-side InsightFace and InSwapper objects that
    live in the task process between those external stages.
    """

    rss_before = _current_rss_mb()

    _dispose_cached_face_swapper()
    _dispose_cached_face_analyzers()

    gc.collect(2)
    _release_loaded_torch_cuda_cache()
    _malloc_trim()
    gc.collect(2)
    _malloc_trim()

    if os.getenv(
        "MEMORY_CLEANUP_LOG",
        "true",
    ).strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    rss_after = _current_rss_mb()

    before_text = (
        f"{rss_before:.1f} MiB"
        if rss_before is not None
        else "unknown"
    )
    after_text = (
        f"{rss_after:.1f} MiB"
        if rss_after is not None
        else "unknown"
    )

    print(
        "[worker-memory] cleanup"
        + (
            f" stage={stage}"
            if stage
            else ""
        )
        + f"; rss_before={before_text}; rss_after={after_text}",
        flush=True,
    )
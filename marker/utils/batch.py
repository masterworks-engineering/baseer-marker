import math

import psutil

from marker.settings import settings


def get_worker_count(oversubscribe: float = 1.5, no_server: bool = False) -> int:
    """Worker processes for batch conversion.

    With a server (OCR enabled), workers are bounded by CPU work (pdftext,
    rendering, postprocessing) and by how many concurrent submitters keep the
    server saturated. With no server (disable_ocr), the work is pure CPU and
    the pool is sized by cores alone."""
    physical_cores = psutil.cpu_count(logical=False) or 4
    if no_server:
        return max(1, physical_cores - 2)
    if settings.MODEL_BACKEND == "chandra":
        return max(1, min(physical_cores - 2, settings.CHANDRA_MAX_WORKERS))
    from surya.settings import settings as surya_settings

    server_parallel = surya_settings.SURYA_INFERENCE_PARALLEL
    if server_parallel is None:
        # Default: parallelism auto-scales to server capacity, which is only
        # known once the backend spawns. Size the pool by CPU alone and let
        # the server govern its own concurrency.
        return max(1, physical_cores - 2)

    workers = min(
        max(1, physical_cores - 2),
        math.ceil(server_parallel * oversubscribe),
    )
    return workers

import resource
import sys
import time
from collections.abc import Callable
from functools import wraps

from credit_scoring.pipeline_log import bind_run_id, get_pipeline_logger, new_run_id, unbind_run_id

STEP_STARTED = "step_started"
STEP_SUCCEEDED = "step_succeeded"
STEP_FAILED = "step_failed"
_MAXRSS_BYTES_PER_UNIT = 1 if sys.platform == "darwin" else 1024


def peak_memory_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * _MAXRSS_BYTES_PER_UNIT


class _ResourceMeter:
    def __init__(self) -> None:
        self.wall_started = time.perf_counter()
        self.cpu_started = time.process_time()

    def usage(self) -> dict[str, float]:
        return {
            "duration_seconds": round(time.perf_counter() - self.wall_started, 3),
            "cpu_seconds": round(time.process_time() - self.cpu_started, 3),
            "peak_memory_bytes": peak_memory_bytes(),
        }


def tracked_step(stage: str) -> Callable:
    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def run(*args, **kwargs):
            token = bind_run_id(new_run_id())
            log = get_pipeline_logger(stage)
            log.info(STEP_STARTED)
            meter = _ResourceMeter()
            try:
                result = function(*args, **kwargs)
            except BaseException as error:
                log.error(
                    STEP_FAILED,
                    extra={
                        **meter.usage(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                raise
            else:
                log.info(STEP_SUCCEEDED, extra=meter.usage())
                return result
            finally:
                unbind_run_id(token)

        return run

    return decorator

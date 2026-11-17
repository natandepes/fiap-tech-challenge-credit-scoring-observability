import argparse

from prometheus_client import (
    CollectorRegistry,
    disable_created_metrics,
    push_to_gateway,
    write_to_textfile,
)

from credit_scoring.config import METRICS_SNAPSHOT_PATH, PUSHGATEWAY_JOB, PUSHGATEWAY_URL
from credit_scoring.observability.metrics import build_registry, load_inputs
from credit_scoring.pipeline_log import get_pipeline_logger

PUSH_TIMEOUT_SECONDS = 5


def series_count(registry: CollectorRegistry) -> int:
    return sum(len(family.samples) for family in registry.collect())


def write_snapshot(registry: CollectorRegistry) -> None:
    METRICS_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_to_textfile(str(METRICS_SNAPSHOT_PATH), registry)


def push(registry: CollectorRegistry, log) -> bool:
    try:
        push_to_gateway(
            PUSHGATEWAY_URL, job=PUSHGATEWAY_JOB, registry=registry, timeout=PUSH_TIMEOUT_SECONDS
        )
    except OSError as error:
        log.warning("metrics_push_failed", extra={"gateway": PUSHGATEWAY_URL, "error": str(error)})
        return False
    log.info("metrics_pushed", extra={"gateway": PUSHGATEWAY_URL, "job": PUSHGATEWAY_JOB})
    return True


def export_metrics(push_to_pushgateway: bool) -> None:
    log = get_pipeline_logger("monitor")
    disable_created_metrics()
    registry = build_registry(load_inputs())
    write_snapshot(registry)
    log.info(
        "metrics_snapshot_written",
        extra={"path": str(METRICS_SNAPSHOT_PATH), "series": series_count(registry)},
    )
    if push_to_pushgateway:
        push(registry, log)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn pipeline logs and drift reports into Prometheus metrics."
    )
    parser.add_argument(
        "--no-push", action="store_true", help="only write the metrics snapshot file"
    )
    export_metrics(push_to_pushgateway=not parser.parse_args().no_push)


if __name__ == "__main__":
    main()

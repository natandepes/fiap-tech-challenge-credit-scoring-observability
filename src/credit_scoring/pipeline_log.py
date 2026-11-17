import json
import logging
import sys
import uuid
from contextvars import ContextVar, Token
from datetime import UTC, datetime

from credit_scoring.config import PIPELINE_LOG_PATH

_STANDARD_RECORD_FIELDS = set(vars(logging.makeLogRecord({}))) | {"message"}
_bound_run_id: ContextVar[str | None] = ContextVar("run_id", default=None)


class JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        payload |= {
            key: value for key, value in vars(record).items() if key not in _STANDARD_RECORD_FIELDS
        }
        return json.dumps(payload, default=str)


class RunContextAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        kwargs["extra"] = {**self.extra, **kwargs.get("extra", {})}
        return msg, kwargs


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def bind_run_id(run_id: str) -> Token:
    return _bound_run_id.set(run_id)


def unbind_run_id(token: Token) -> None:
    _bound_run_id.reset(token)


def get_pipeline_logger(stage: str) -> RunContextAdapter:
    logger = logging.getLogger(f"credit_scoring.{stage}")
    if not logger.handlers:
        PIPELINE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        formatter = JsonLinesFormatter()
        for handler in (
            logging.FileHandler(PIPELINE_LOG_PATH),
            logging.StreamHandler(sys.stderr),
        ):
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return RunContextAdapter(
        logger, {"stage": stage, "run_id": _bound_run_id.get() or new_run_id()}
    )

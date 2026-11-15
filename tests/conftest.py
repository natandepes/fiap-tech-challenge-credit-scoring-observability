import pytest

from credit_scoring import pipeline_log


@pytest.fixture(autouse=True, scope="session")
def isolated_pipeline_log(tmp_path_factory):
    patcher = pytest.MonkeyPatch()
    patcher.setattr(
        pipeline_log, "PIPELINE_LOG_PATH", tmp_path_factory.mktemp("logs") / "pipeline.jsonl"
    )
    yield
    patcher.undo()

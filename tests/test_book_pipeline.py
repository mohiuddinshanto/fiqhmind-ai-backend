"""Phase 15 M4 per-book task-graph tests (requirements D, E).

Proves the orchestration primitives in `app/tasks/book.py`:

- the per-book chain/group composition
  `extraction → metadata → chunking → group(embed, index)`,
- the automatic pipeline dispatcher (`start_ingestion_pipeline_task`) that
  creates the stage jobs and kicks the graph off after an upload, and
- child-failure propagation: a raising leaf fires the `fail_book_jobs`
  errback, which fails every active job of the book (parent included) and
  marks the upload failed.

Overlap/ordering of page-level extraction and batched embedding live in
`test_extraction_concurrency.py` and `test_indexing_service.py`.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.tasks.book as book_module
from app.db.base import Base
from app.db.models import IngestionJob, Upload
from app.db.repositories import IngestionJobRepository, UploadRepository
from app.worker.celery_app import celery_app


@pytest.fixture()
def session() -> Session:
    engine = create_engine(
        "sqlite://",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(bind=engine, expire_on_commit=False)
    yield testing_session()
    engine.dispose()


def _make_upload(session: Session, *, status: str = "queued") -> Upload:
    return UploadRepository(session).create(
        Upload(
            original_filename="kitab.pdf",
            filename="kitab.pdf",
            storage_path="kitab.pdf",
            mime="application/pdf",
            status=status,
        )
    )


def _make_job(
    session: Session, upload_id: str, kind: str, *, status: str = "queued"
) -> IngestionJob:
    return IngestionJobRepository(session).create(
        IngestionJob(upload_id=upload_id, kind=kind, status=status)
    )


def _eager_session_factory(session: Session):
    """Zero-arg `get_session_factory` returning a fresh sessionmaker (as the app does)."""
    factory = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    return lambda: factory


def _errbacks(signature) -> list:
    wired = signature.options.get("link_error", [])
    return wired if isinstance(wired, list) else [wired]


def _canvas_kind(signature) -> str:
    """`_chain` / `_group` — Celery 5 `chain()`/`group()` return canvas types."""
    return type(signature).__name__


def test_book_graph_chain_structure() -> None:
    """The full book graph is extract → metadata → chunk → group(embed, index)."""
    graph = book_module.build_book_graph(
        extraction_job_id="extraction-id",
        upload_id="upload-id",
        chunking_job_id="chunking-id",
        indexing_job_id="indexing-id",
        metadata_job_id="metadata-id",
    )

    assert _canvas_kind(graph) == "_chain"
    assert [task.task for task in graph.tasks[:3]] == [
        "app.tasks.ingestion.extract_pdf_task",
        "app.tasks.ingestion.run_metadata_task",
        "app.tasks.ingestion.run_chunking_task",
    ]
    assert tuple(graph.tasks[0].args) == ("extraction-id", "upload-id")
    assert tuple(graph.tasks[1].args) == ("metadata-id", "upload-id")
    assert tuple(graph.tasks[2].args) == ("chunking-id", "upload-id", "metadata-id")

    indexing_stage = graph.tasks[3]
    assert indexing_stage.task == "celery.group"
    assert {task.task for task in indexing_stage.tasks} == {
        "app.tasks.ingestion.embed_book_chunks",
        "app.tasks.ingestion.index_book_chunks",
    }
    by_name = {task.task: task for task in indexing_stage.tasks}
    assert tuple(by_name["app.tasks.ingestion.index_book_chunks"].args) == (
        "indexing-id",
        "upload-id",
        "chunking-id",
    )
    assert tuple(by_name["app.tasks.ingestion.embed_book_chunks"].args) == (
        "upload-id",
        "chunking-id",
    )


def test_book_graph_executes_without_chain_result_threading(monkeypatch) -> None:
    """Regression: Celery must not inject the previous result into chain links.

    `run_metadata_task` and `run_chunking_task` are called with `.si()` (immutable
    signatures) precisely so Celery does not prepend the previous stage's return
    value to their arguments. A mutable `.s()` link makes the metadata stage
    receive `(<extract result>, metadata_job_id, upload_id)` — the real-world
    `run_metadata_task(job_id, upload_id)` signature failure seen in production.
    This test executes the full graph eagerly (not just inspects its shape) so a
    future mutable-link regression is caught by Celery itself.
    """
    calls: list[tuple] = []

    def make_bound(name: str, returns: object):
        def stub(self, *args, **kwargs) -> object:
            calls.append((name, args, kwargs))
            return returns

        return celery_app.task(name=name, bind=True)(stub)

    def make_plain(name: str, returns: object):
        def stub(*args, **kwargs) -> object:
            calls.append((name, args, kwargs))
            return returns

        return celery_app.task(name=name, bind=False)(stub)

    extract_stub = make_bound("app.tasks.book.regr_extract", {"stage": "extraction"})
    metadata_stub = make_bound("app.tasks.book.regr_metadata", {"stage": "metadata"})
    chunking_stub = make_bound("app.tasks.book.regr_chunking", {"stage": "chunking"})
    embed_stub = make_plain("app.tasks.book.regr_embed", {"stage": "embed"})
    index_stub = make_bound("app.tasks.book.regr_index", {"stage": "index"})

    monkeypatch.setattr(book_module, "extract_pdf_task", extract_stub)
    monkeypatch.setattr(book_module, "run_metadata_task", metadata_stub)
    monkeypatch.setattr(book_module, "run_chunking_task", chunking_stub)
    monkeypatch.setattr(book_module, "embed_chunks_task", embed_stub)
    monkeypatch.setattr(book_module, "run_indexing_task", index_stub)

    celery_app.conf.task_always_eager = True
    try:
        graph = book_module.build_book_graph(
            extraction_job_id="extraction-id",
            upload_id="upload-id",
            chunking_job_id="chunking-id",
            indexing_job_id="indexing-id",
            metadata_job_id="metadata-id",
        )
        result = graph.apply_async()
    finally:
        celery_app.conf.task_always_eager = False

    assert result.get() == [{"stage": "embed"}, {"stage": "index"}]
    by_name = {name: args for name, args, _ in calls}
    assert tuple(by_name["app.tasks.book.regr_extract"]) == ("extraction-id", "upload-id")
    assert tuple(by_name["app.tasks.book.regr_metadata"]) == ("metadata-id", "upload-id")
    assert tuple(by_name["app.tasks.book.regr_chunking"]) == (
        "chunking-id",
        "upload-id",
        "metadata-id",
    )
    assert tuple(by_name["app.tasks.book.regr_embed"]) == ("upload-id", "chunking-id")
    assert tuple(by_name["app.tasks.book.regr_index"]) == (
        "indexing-id",
        "upload-id",
        "chunking-id",
    )

    ordered = [name for name, _, _ in calls]
    assert ordered[:3] == [
        "app.tasks.book.regr_extract",
        "app.tasks.book.regr_metadata",
        "app.tasks.book.regr_chunking",
    ]
    assert set(ordered[3:]) == {"app.tasks.book.regr_embed", "app.tasks.book.regr_index"}


def test_stage_graphs_compose_expected_primitives() -> None:
    """Each endpoint dispatches the right primitive with the right arguments."""
    extraction = book_module.build_extraction_stage("extraction-id", "upload-id")
    assert _canvas_kind(extraction) == "chain"
    assert extraction.tasks[0].task == "app.tasks.ingestion.extract_pdf_task"
    assert tuple(extraction.tasks[0].args) == ("extraction-id", "upload-id")

    chunking = book_module.build_chunking_stage("chunking-id", "upload-id", "metadata-id")
    assert _canvas_kind(chunking) == "chain"
    assert chunking.tasks[0].task == "app.tasks.ingestion.run_chunking_task"
    assert tuple(chunking.tasks[0].args) == ("chunking-id", "upload-id", "metadata-id")

    indexing = book_module.build_indexing_stage("indexing-id", "upload-id", "chunking-id")
    assert _canvas_kind(indexing) == "group"
    by_name = {task.task: task for task in indexing.tasks}
    assert set(by_name) == {
        "app.tasks.ingestion.embed_book_chunks",
        "app.tasks.ingestion.index_book_chunks",
    }
    assert tuple(by_name["app.tasks.ingestion.index_book_chunks"].args) == (
        "indexing-id",
        "upload-id",
        "chunking-id",
    )
    assert tuple(by_name["app.tasks.ingestion.embed_book_chunks"].args) == (
        "upload-id",
        "chunking-id",
    )


def test_stage_graphs_wire_child_failure_errback() -> None:
    """Every leaf links its failure to fail_book_jobs with the stage's job id."""
    cases = [
        (book_module.build_extraction_stage("e", "u"), "e"),
        (book_module.build_chunking_stage("c", "u", "m"), "c"),
    ]
    for graph, job_id in cases:
        (errback,) = _errbacks(graph.tasks[0])
        assert errback.task == "app.tasks.book.fail_book_jobs"
        assert errback.kwargs == {"upload_id": "u", "failed_job_id": job_id}

    indexing = book_module.build_indexing_stage("i", "u", "c")
    for task in indexing.tasks:
        (errback,) = _errbacks(task)
        assert errback.task == "app.tasks.book.fail_book_jobs"
        assert errback.kwargs == {"upload_id": "u", "failed_job_id": "i"}


def test_build_stage_graph_unknown_stage_raises() -> None:
    with pytest.raises(ValueError):
        book_module.build_stage_graph("nope", job_id="j", upload_id="u")


def test_fail_book_jobs_fails_active_siblings_and_upload(session: Session, monkeypatch) -> None:
    """A failed child fails the parent and every other in-flight job + the upload."""
    upload = _make_upload(session)
    extraction = _make_job(session, upload.id, "extraction", status="extracting")
    chunking = _make_job(session, upload.id, "chunking", status="queued")
    indexing = _make_job(session, upload.id, "indexing", status="queued")
    metadata = _make_job(session, upload.id, "metadata", status="completed")
    session.commit()

    monkeypatch.setattr(book_module, "get_session_factory", _eager_session_factory(session))

    book_module.fail_book_jobs.run(
        ("request", RuntimeError("boom"), "traceback"),
        upload_id=upload.id,
        failed_job_id=extraction.id,
    )
    session.expire_all()

    repo = IngestionJobRepository(session)
    assert repo.get(extraction.id).status == "failed"
    assert repo.get(chunking.id).status == "failed"
    assert repo.get(indexing.id).status == "failed"
    assert repo.get(metadata.id).status == "completed"  # terminal jobs are untouched
    assert repo.get(extraction.id).error_message == (
        f"ingestion pipeline failed upstream (job {extraction.id})"
    )

    failed_upload = UploadRepository(session).get(upload.id)
    assert failed_upload.status == "failed"
    assert "failed upstream" in failed_upload.error_message


def test_raising_leaf_fires_errback_and_fails_book(session: Session, monkeypatch) -> None:
    """A raising leaf in the orchestrated graph fails the parent + siblings + upload."""
    upload = _make_upload(session)
    extraction = _make_job(session, upload.id, "extraction", status="queued")
    chunking = _make_job(session, upload.id, "chunking", status="queued")
    session.commit()

    monkeypatch.setattr(book_module, "get_session_factory", _eager_session_factory(session))

    @celery_app.task(name="app.tasks.book.test_raising_leaf")
    def raising_leaf(*args, **kwargs) -> None:
        raise RuntimeError("child exploded")

    monkeypatch.setattr(book_module, "extract_pdf_task", raising_leaf)

    celery_app.conf.task_always_eager = True
    try:
        book_module.build_extraction_stage(extraction.id, upload.id).apply_async()
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    repo = IngestionJobRepository(session)
    assert repo.get(extraction.id).status == "failed"
    assert repo.get(chunking.id).status == "failed"
    assert UploadRepository(session).get(upload.id).status == "failed"


def test_process_book_task_dispatches_stage_graph(session: Session, monkeypatch) -> None:
    """The endpoint entry point applies the stage graph it was asked for."""
    upload = _make_upload(session)
    job = _make_job(session, upload.id, "extraction", status="queued")
    session.commit()

    applied: list[tuple] = []

    class FakeGraph:
        def apply_async(self, *args, **kwargs) -> None:
            applied.append((args, kwargs))

    def build_stage_graph(stage, **kwargs):
        assert stage == "extraction"
        assert kwargs["job_id"] == job.id
        assert kwargs["upload_id"] == upload.id
        return FakeGraph()

    monkeypatch.setattr(book_module, "build_stage_graph", build_stage_graph)
    monkeypatch.setattr(book_module, "get_session_factory", _eager_session_factory(session))

    celery_app.conf.task_always_eager = True
    try:
        book_module.process_book_task.delay(upload.id, stage="extraction", job_id=job.id)
    finally:
        celery_app.conf.task_always_eager = False

    assert len(applied) == 1


def test_process_book_task_skips_unknown_upload(session: Session, monkeypatch) -> None:
    """A missing upload must not dispatch anything (and must not duplicate jobs)."""
    applied: list[tuple] = []

    def boom_graph(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("graph built for a missing upload")

    monkeypatch.setattr(book_module, "build_stage_graph", boom_graph)
    monkeypatch.setattr(book_module, "get_session_factory", _eager_session_factory(session))

    celery_app.conf.task_always_eager = True
    try:
        book_module.process_book_task.delay(
            "00000000000000000000000000000000", stage="extraction", job_id="j"
        )
    finally:
        celery_app.conf.task_always_eager = False

    assert applied == []
    assert len(IngestionJobRepository(session).get_multi()) == 0


def test_start_ingestion_pipeline_creates_stage_jobs_and_dispatches_graph(
    session: Session, monkeypatch
) -> None:
    """The auto-ingestion task creates missing stage jobs and applies the graph."""
    upload = _make_upload(session)
    initial = _make_job(session, upload.id, "initial", status="queued")
    session.commit()

    applied: list[tuple] = []
    captured: dict[str, str] = {}

    class FakeGraph:
        def apply_async(self, *args, **kwargs) -> None:
            applied.append((args, kwargs))

    def build_book_graph(**kwargs):
        captured.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr(book_module, "build_book_graph", build_book_graph)
    monkeypatch.setattr(book_module, "get_session_factory", _eager_session_factory(session))

    celery_app.conf.task_always_eager = True
    try:
        book_module.start_ingestion_pipeline_task.delay(upload.id)
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    repo = IngestionJobRepository(session)
    metadata = repo.find_for_upload(upload.id, "metadata")
    chunking = repo.find_for_upload(upload.id, "chunking")
    indexing = repo.find_for_upload(upload.id, "indexing")
    assert metadata is not None
    assert chunking is not None
    assert indexing is not None
    assert captured == {
        "extraction_job_id": initial.id,
        "upload_id": upload.id,
        "chunking_job_id": chunking.id,
        "indexing_job_id": indexing.id,
        "metadata_job_id": metadata.id,
    }
    assert len(repo.get_multi()) == 4  # initial + metadata + chunking + indexing
    assert len(applied) == 1


def test_start_ingestion_pipeline_reuses_existing_stage_jobs(
    session: Session, monkeypatch
) -> None:
    """A re-run reuses existing stage jobs instead of duplicating rows."""
    upload = _make_upload(session)
    metadata = _make_job(session, upload.id, "metadata", status="completed")
    _make_job(session, upload.id, "initial", status="queued")
    session.commit()

    applied: list[tuple] = []
    captured: dict[str, str] = {}

    class FakeGraph:
        def apply_async(self, *args, **kwargs) -> None:
            applied.append((args, kwargs))

    def build_book_graph(**kwargs):
        captured.update(kwargs)
        return FakeGraph()

    monkeypatch.setattr(book_module, "build_book_graph", build_book_graph)
    monkeypatch.setattr(book_module, "get_session_factory", _eager_session_factory(session))

    celery_app.conf.task_always_eager = True
    try:
        book_module.start_ingestion_pipeline_task.delay(upload.id)
    finally:
        celery_app.conf.task_always_eager = False

    session.expire_all()
    repo = IngestionJobRepository(session)
    assert repo.find_for_upload(upload.id, "metadata").id == metadata.id
    assert repo.find_for_upload(upload.id, "chunking") is not None
    assert repo.find_for_upload(upload.id, "indexing") is not None
    assert len(repo.get_multi()) == 4  # existing initial + metadata, new chunking + indexing
    assert captured["metadata_job_id"] == metadata.id
    assert len(applied) == 1

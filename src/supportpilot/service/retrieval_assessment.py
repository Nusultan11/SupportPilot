"""Fresh locator-reference study; support groups remain draft mechanical proxies."""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

import yaml

from supportpilot.common.config import load_settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import write_json
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.retrieval import open_retriever
from supportpilot.retrieval.experiments import measure
from supportpilot.retrieval.scale_metrics import measure_query, summarise_scale
from supportpilot.retrieval.search import assemble_context
from supportpilot.service.evaluation import SCOPES, root


def main() -> None:
    import mlflow

    settings = load_settings()
    source = settings.root / "authored/system/retrieval.yaml"
    identity = sha256_file(source)
    frozen = settings.resolve(settings.paths.data_dir) / "benchmarks/system_retrieval" / identity
    frozen.mkdir(parents=True, exist_ok=True)
    # One terminal assessment of the frozen supplement; no automatic rerun or tuning.
    if (frozen / "attempt.json").exists():
        raise ValueError("supplement already attempted; retain evidence before technical retry")
    out = root(settings) / "retrieval" / uuid4().hex
    out.mkdir(parents=True)
    write_json(frozen / "attempt.json", {"identity": identity, "path": str(out)})
    raw = yaml.safe_load(source.read_text("utf-8"))
    rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "status": "RUNNING",
        "source_sha256": identity,
        "rows": rows,
        "source": str(source),
        "scope": "draft section-reference coverage, not semantic proof",
        "semantic_quality": "NOT_EVALUATED",
        "selection": "none; assessment only",
    }
    spec = RunSpec(
        experiment="system",
        run_name="retrieval_locator_assessment",
        hypothesis="Measure full/packed reference loss without inferring semantic support",
        dataset_version=identity,
        seed=20261115,
        params={"selection": "none", "quality_scope": "draft_locator_proxy"},
    )
    try:
        with (
            start_run(spec, settings) as run,
            measure() as resources,
            open_retriever(settings) as retriever,
        ):
            report["run_id"] = run.info.run_id
            report["kb_version"] = retriever.artifact.kb_version
            report["index_version"] = retriever.artifact.manifest.dataset_version
            for need_id, family, sections, *texts in raw["needs"]:
                for index, scope in enumerate(SCOPES):
                    lang = "kk" if scope == "mixed" else scope
                    groups = [
                        [
                            c.chunk_id
                            for c in retriever.artifact.chunks
                            if f"{c.doc_id}.{c.section_id}" == section and c.lang == lang
                        ]
                        for section in sections
                    ]
                    if any(not group for group in groups):
                        raise ValueError("unresolvable draft proof section")
                    ranking = retriever.retrieve(texts[index], lang, k=10)
                    for budget in (512, 1024, 2048):
                        context = assemble_context(
                            ranking,
                            retriever.tokenizer,
                            budget,
                            kb_version=retriever.artifact.kb_version,
                            index_version=retriever.artifact.manifest.dataset_version,
                        )
                        ids, packed = (
                            [r.chunk_id for r in ranking],
                            [r.chunk_id for r in context.evidence],
                        )
                        row: dict[str, Any] = {
                            "scenario_id": need_id,
                            "family_id": family,
                            "query_id": f"{need_id}:{scope}:{budget}",
                            "phase": "supplemental_assessment",
                            "answerable": bool(groups),
                            "slices": [f"{scope}_budget{budget}"],
                            "text": texts[index],
                            "ranking": ids,
                            "packed": packed,
                            "required_section_groups": groups,
                            "reference_label": "draft_section_proxy",
                            "timings": dict(retriever.last_timings),
                        }
                        row.update(
                            measure_query(
                                ids,
                                packed,
                                {cid: 1 for g in groups for cid in g},
                                groups,
                                answerable=bool(groups),
                            )
                        )
                        rows.append(row)
                    write_json(out / "report.json", report)
                print("RETRIEVAL " + need_id, flush=True)
            report["summary"] = summarise_scale(rows, seed=20261115)
            report["status"] = "FINISHED"
            mlflow.set_tags(
                {
                    "semantic_quality": "NOT_EVALUATED",
                    "decision": "retain_1024_no_semantic_selection",
                }
            )
    except BaseException as exc:
        report.update(status="FAILED", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        if "resources" in locals():
            report["resources"] = vars(resources)
        write_json(out / "report.json", report)
        if report.get("run_id"):
            mlflow.MlflowClient().log_artifact(report["run_id"], str(out / "report.json"))
    print(json.dumps({"path": str(out / "report.json")}))


if __name__ == "__main__":
    main()

"""Pinned embedding overlap diagnostics; flags never certify semantic independence."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl

from supportpilot.common.config import load_settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import read_json, write_json
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.data.taxonomy import load_taxonomy
from supportpilot.retrieval.experiments import measure
from supportpilot.routing.router import EncoderRouter
from supportpilot.service.evaluation import SCOPES, root, verify_freeze
from supportpilot.service.evaluation_audit import normalise, strings


def main() -> None:
    import mlflow

    settings = load_settings()
    directory = root(settings)
    frozen = verify_freeze(Path(read_json(directory / "freeze_pointer.json")["path"]))
    lexical = read_json(directory / "overlap_audit.json")
    references: dict[str, list[dict[str, object]]] = {}
    readbacks: dict[str, str] = {}
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required for frozen baseline readback")
    for name, expected in lexical["sources"].items():
        path = Path(name)
        content = path.read_bytes()
        readbacks[name] = "working_file"
        if hashlib.sha256(content).hexdigest() != expected:
            commit = frozen["code_commit"].removesuffix("-dirty")
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError("invalid baseline commit")
            relative = path.relative_to(settings.root).as_posix()
            saved = subprocess.run(  # noqa: S603 - read-only Git blob, no shell
                [git, "show", f"{commit}:{relative}"],
                capture_output=True,
                check=True,
                cwd=settings.root,
            ).stdout
            if hashlib.sha256(saved).hexdigest() != expected:
                raise ValueError("reference data changed since lexical audit")
            content = saved
            readbacks[name] = "git@" + commit
        if path.suffix == ".parquet":
            texts = pl.read_parquet(path, columns=["user_text"])["user_text"].to_list()
        elif path.suffix == ".json":
            texts = strings(read_json(path))
        else:
            texts = re.findall(r'"([^"\n]{10,})"', content.decode("utf-8"))
        for index, text in enumerate(texts):
            if isinstance(text, str) and (key := normalise(text)):
                references.setdefault(key, []).append({"source": name, "row": index})
    out = directory / "semantic_overlap" / uuid4().hex
    out.mkdir(parents=True)
    taxonomy = load_taxonomy(settings.resolve(settings.stage4.taxonomy))
    router = EncoderRouter.load(settings.resolve(settings.serving.router), taxonomy, device="cpu")
    router.encoder.spec = router.encoder.spec.model_copy(update={"batch_size": 8})
    spec = RunSpec(
        experiment="system",
        run_name="semantic_overlap_diagnostic",
        hypothesis="Identify near-copy candidates without excluding inspected confirmation rows",
        dataset_version=frozen["identity"],
        seed=20261112,
        base_model=router.encoder.spec.repo_id + "@" + router.encoder.spec.revision,
        params={"threshold": settings.stage4.leak_cosine, "selection": "none"},
    )
    report: dict[str, Any] = {
        "status": "RUNNING",
        "freeze_identity": frozen["identity"],
        "reference_sources": lexical["sources"],
        "reference_readbacks": readbacks,
        "encoder": router.encoder.spec.model_dump(mode="json"),
        "threshold": settings.stage4.leak_cosine,
        "semantic_independence": "NOT_CERTIFIED",
        "interpretation": "embedding flags only; no exclusions, retuning or semantic quality claim",
    }
    try:
        with start_run(spec, settings) as active, measure() as resources:
            report["run_id"] = active.info.run_id
            reference_texts = list(references)
            chunks = []
            for start in range(0, len(reference_texts), 256):
                chunks.append(
                    router.encoder.encode(reference_texts[start : start + 256], query=True)
                )
                print(
                    f"OVERLAP {min(start + 256, len(reference_texts))}/{len(reference_texts)}",
                    flush=True,
                )
            reference_vectors = np.concatenate(chunks)
            queries = [
                (n, scope, text)
                for n in frozen["needs"]
                for scope, text in zip(SCOPES, n["texts"], strict=True)
            ]
            query_vectors = router.encoder.encode(
                [normalise(text) for _, _, text in queries], query=True
            )
            np.save(out / "reference_vectors.npy", reference_vectors, allow_pickle=False)
            np.save(out / "query_vectors.npy", query_vectors, allow_pickle=False)
            rows = []
            for (need, scope, _), vector in zip(queries, query_vectors, strict=True):
                scores = reference_vectors @ vector
                index = int(scores.argmax())
                hits = np.flatnonzero(scores >= settings.stage4.leak_cosine).tolist()
                rows.append(
                    {
                        "need_id": need["need_id"],
                        "phase": need["phase"],
                        "scope": scope,
                        "maximum_cosine": float(scores[index]),
                        "closest": references[reference_texts[index]],
                        "flagged": bool(hits),
                        "matches": [
                            {"cosine": float(scores[i]), "sources": references[reference_texts[i]]}
                            for i in hits
                        ],
                    }
                )
            report.update(
                status="FINISHED",
                rows=rows,
                unique_reference_texts=len(reference_texts),
                flagged_rows=sum(r["flagged"] for r in rows),
                vector_sha256={p.name: sha256_file(p) for p in out.glob("*.npy")},
            )
            mlflow.set_tags(
                {"decision": "diagnostic_no_retuning", "semantic_independence": "NOT_CERTIFIED"}
            )
    except BaseException as exc:
        report.update(status="FAILED", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        router.encoder.close()
        if "resources" in locals():
            report["resources"] = vars(resources)
        write_json(out / "report.json", report)
        if report.get("run_id"):
            mlflow.MlflowClient().log_artifacts(report["run_id"], str(out))
    print(json.dumps({"path": str(out / "report.json")}))


if __name__ == "__main__":
    main()

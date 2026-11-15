"""Reuse verified frozen reference embeddings; never certify semantic independence."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import polars as pl
import yaml

from supportpilot.common.config import load_settings
from supportpilot.common.hashing import sha256_file
from supportpilot.common.io import read_json, write_json
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.data.taxonomy import load_taxonomy
from supportpilot.retrieval.experiments import measure
from supportpilot.routing.router import EncoderRouter
from supportpilot.service.evaluation import SCOPES
from supportpilot.service.evaluation_audit import normalise, strings
from supportpilot.service.quality_evaluation import location, verify


def main() -> None:
    import mlflow

    settings = load_settings()
    base = location(settings)
    frozen = verify(settings, Path(read_json(base / "freeze_pointer.json")["path"]))
    prior = (
        settings.root / "artifacts/reports/system/semantic_overlap/8279c216c374447cb0b24540ec921bd0"
    )
    old = read_json(prior / "report.json")
    references: dict[str, list[dict[str, Any]]] = {}
    for name, expected in old["reference_sources"].items():
        path = Path(name)
        content = path.read_bytes()
        if sha256_file(path) != expected:
            readback = old["reference_readbacks"][name]
            if not readback.startswith("git@"):
                raise ValueError("changed prior reference without frozen Git readback")
            git = shutil.which("git")
            if git is None:
                raise ValueError("Git unavailable")
            content = subprocess.run(  # noqa: S603 - trusted frozen Git object, no shell
                [git, "show", f"{readback[4:]}:{path.relative_to(settings.root).as_posix()}"],
                check=True,
                capture_output=True,
                cwd=settings.root,
            ).stdout
            import hashlib

            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("changed frozen reference")
        texts = (
            pl.read_parquet(path, columns=["user_text"])["user_text"].to_list()
            if path.suffix == ".parquet"
            else strings(json.loads(content))
            if path.suffix == ".json"
            else re.findall(r'"([^"\n]{10,})"', content.decode("utf-8"))
        )
        for i, text in enumerate(texts):
            if isinstance(text, str) and (key := normalise(text)):
                references.setdefault(key, []).append({"source": name, "row": i})
    vector_path = prior / "reference_vectors.npy"
    if sha256_file(vector_path) != old["vector_sha256"][vector_path.name]:
        raise ValueError("changed reference embeddings")
    vectors = np.load(vector_path, allow_pickle=False)
    if len(vectors) != len(references) or len(references) != old["unique_reference_texts"]:
        raise ValueError("reference inventory mismatch")
    extra_path = settings.root / "authored/system/scenarios.yaml"
    extra = yaml.safe_load(extra_path.read_text("utf-8"))
    additional = [normalise(t) for r in extra["needs"] for t in r[6:]]
    reference_texts = list(references)
    queries = [
        (r, scope, text)
        for r in frozen["inputs"]["needs"]
        for scope, text in zip(SCOPES, r[8], strict=True)
    ]
    out = base / "overlap" / uuid4().hex
    out.mkdir(parents=True)
    report: dict[str, Any] = {
        "status": "RUNNING",
        "identity": frozen["identity"],
        "reference_report_sha256": sha256_file(prior / "report.json"),
        "reference_vectors_sha256": sha256_file(vector_path),
        "additional_source_sha256": sha256_file(extra_path),
        "semantic_independence": "NOT_CERTIFIED",
        "exclusions": [],
    }
    write_json(out / "report.json", report)
    taxonomy = load_taxonomy(settings.resolve(settings.stage4.taxonomy))
    router = EncoderRouter.load(settings.resolve(settings.serving.router), taxonomy, device="cpu")
    router.encoder.spec = router.encoder.spec.model_copy(update={"batch_size": 8})
    if router.encoder.spec.repo_id != old["encoder"]["repo_id"] or (
        router.encoder.spec.revision != old["encoder"]["revision"]
    ):
        raise ValueError("embedding model differs")
    spec = RunSpec(
        experiment="system_quality",
        run_name="overlap_diagnostic",
        hypothesis="Screen fresh family splits without exclusions or certification",
        dataset_version=frozen["identity"],
        seed=20261115,
        params={"threshold": 0.95, "selection": "none"},
    )
    try:
        with start_run(spec, settings) as active, measure() as resources:
            report["run_id"] = active.info.run_id
            additional_vectors = router.encoder.encode(additional, query=True)
            vectors = np.concatenate([vectors, additional_vectors])
            reference_texts.extend(additional)
            qtexts = [normalise(text) for _, _, text in queries]
            qvectors = router.encoder.encode(qtexts, query=True)
            np.save(out / "query_vectors.npy", qvectors, allow_pickle=False)
            grams = [{t[i : i + 5] for i in range(max(1, len(t) - 4))} for t in reference_texts]
            rows = []
            for i, (need, scope, _) in enumerate(queries):
                scores = vectors @ qvectors[i]
                query_grams = {qtexts[i][j : j + 5] for j in range(max(1, len(qtexts[i]) - 4))}
                lexical = [
                    j
                    for j, g in enumerate(grams)
                    if len(g & query_grams) / max(1, len(g | query_grams)) >= 0.8
                ]
                cross = [j for j, (r, _, _) in enumerate(queries) if r[1] != need[1]]
                cross_scores = qvectors[cross] @ qvectors[i]
                closest_cross = cross[int(cross_scores.argmax())]
                index = int(scores.argmax())
                rows.append(
                    {
                        "id": need[0] + ":" + scope,
                        "phase": need[1],
                        "lexical_matches": lexical,
                        "max_cosine": float(scores[index]),
                        "closest_reference": reference_texts[index],
                        "embedding_flag": bool(scores[index] >= 0.95),
                        "cross_split_cosine": float(cross_scores.max()),
                        "cross_split_need": queries[closest_cross][0][0],
                        "cross_split_embedding_flag": bool(cross_scores.max() >= 0.95),
                    }
                )
            report.update(
                status="FINISHED",
                rows=rows,
                reference_texts=len(reference_texts),
                lexical_flags=sum(bool(r["lexical_matches"]) for r in rows),
                embedding_flags=dict(Counter(r["phase"] for r in rows if r["embedding_flag"])),
                cross_split_flags=sum(r["cross_split_embedding_flag"] for r in rows),
                encoder=router.encoder.spec.model_dump(mode="json"),
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

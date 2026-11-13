"""Post-selection wording diagnostics on existing development needs only."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict

from supportpilot.common.config import Settings
from supportpilot.common.hashing import sha256_file, sha256_text
from supportpilot.common.io import write_json
from supportpilot.common.manifest import read_model_json, scan_files, verify_files
from supportpilot.common.provenance import code_commit
from supportpilot.common.schemas import LANGUAGES, DatasetManifest
from supportpilot.common.tracking import RunSpec, start_run
from supportpilot.data.eda import load_tokenizer
from supportpilot.data.retrievalset import RetrievalQuery, leakage_checks
from supportpilot.data.sftbuild import current_sft
from supportpilot.data.split import exact_leaks, lexical_leaks, ngram_leaks
from supportpilot.data.steps import partial_dir
from supportpilot.kb.profiles import load_profiles, profile_items
from supportpilot.kb.spec import Tri
from supportpilot.ops.health import make_qdrant_client
from supportpilot.retrieval.abstain import evidence_gate
from supportpilot.retrieval.evaluate import slices
from supportpilot.retrieval.experiments import score_pairs
from supportpilot.retrieval.models import SearchOptions, load_models
from supportpilot.retrieval.scale_data import verify_freeze
from supportpilot.retrieval.scale_metrics import measure_query, summarise_scale
from supportpilot.retrieval.scale_run import prepare_backend
from supportpilot.retrieval.search import assemble_context, scored_chunks


class DiagnosticSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["retrieval_diagnostic_source/1"]
    origin: Literal["llm_generated"]
    review_status: Literal["draft"]
    author: str
    allowed_use: Literal["eval_only"]
    licence_note: str
    scope: str
    templates: dict[str, Tri]


def entity_typo(name: str) -> str:
    """Delete one internal letter of the longest name word; preserve the other words."""
    words = list(re.finditer(r"[^\W\d_]+", name))
    if not words or max(len(m.group()) for m in words) < 5:
        raise ValueError("entity name lacks a word long enough for a controlled typo")
    word = max(words, key=lambda m: len(m.group()))
    position = word.start() + len(word.group()) // 2
    return name[:position] + name[position + 1 :]


def diagnostic_queries(
    settings: Settings,
    parent: Path,
    source: DiagnosticSource,
    seed: int,
) -> tuple[list[RetrievalQuery], dict[str, Any]]:
    meta = json.loads((parent / "metadata.json").read_text())
    native: dict[tuple[str, str], RetrievalQuery] = {}
    for line in (parent / "queries.jsonl").read_text("utf-8").splitlines():
        q = RetrievalQuery.model_validate_json(line)
        if (
            q.answerable
            and not q.cross_lingual
            and not q.code_switch
            and meta[q.query_id]["phase"] == "development"
        ):
            native[q.scenario_id, q.lang] = q
    profiles = profile_items(
        load_profiles(settings.resolve(settings.kb.source_root) / "technical_profiles.yaml")
    )
    items = {item["id"]: item for item in profiles}
    grouped: dict[str, dict[str, tuple[str, str]]] = {}
    for (scenario, lang), q in native.items():
        key = meta[q.query_id]["need_key"]
        if lang != "en" or not key.startswith("technical.") or not key.endswith("|current"):
            continue
        _, slug, field = key.split("|")[0].split(".")
        if field in source.templates and len(meta[q.query_id]["required_groups"]) == 1:
            grouped.setdefault(items[slug]["group"], {})[scenario] = (slug, field)
    selected: list[tuple[str, str, str, str, bool]] = []
    for group, needs in sorted(grouped.items()):
        choices = sorted(needs, key=lambda sid: sha256_text(f"{seed}|{sid}"))[:4]
        if len(choices) != 4:
            raise ValueError(f"insufficient existing development needs in {group}")
        selected.extend(
            (group, scenario, *needs[scenario], index == 0)
            for index, scenario in enumerate(choices)
        )
    queries, metadata = [], {}

    def add(scenario: str, slug: str, field: str, lang: str, corpus: str, kind: str) -> None:
        base = native[scenario, corpus]
        name = getattr(items[slug]["name"], lang)
        text = getattr(source.templates[field], lang).format(
            product=entity_typo(name) if kind == "entity_typo" else name
        )
        if kind == "mixed":
            text += " Уточните по карточке, пожалуйста."
        qid = "rqd_" + sha256_text(f"{base.query_id}|{lang}|{kind}|{text}")[:20]
        payload = {
            **base.model_dump(mode="json"),
            "query_id": qid,
            "query": text,
            "lang": lang,
            "corpus_lang": corpus,
            "code_switch": kind == "mixed",
            "cross_lingual": lang != corpus,
            "category": "hard",
            "author": source.author,
            "review_status": "draft",
            "rationale": f"Existing dev need {scenario}; {field}; {kind}. "
            "No new need or fact; post-selection diagnostic only.",
        }
        queries.append(RetrievalQuery.model_validate(payload))
        metadata[qid] = {
            **meta[base.query_id],
            "reference_query_id": base.query_id,
            "phase": "diagnostic",
            "parent_phase": "development",
            "kind": kind,
            "tags": [*meta[base.query_id]["tags"], "diagnostic_" + kind],
        }

    for _, scenario, slug, field, extra in selected:
        for lang in LANGUAGES:
            for kind in ("paraphrase", "entity_typo"):
                add(scenario, slug, field, lang, lang, kind)
            if extra:
                for corpus in LANGUAGES:
                    if corpus != lang:
                        add(scenario, slug, field, lang, corpus, "cross_paraphrase")
        if extra:
            add(scenario, slug, field, "kk", "kk", "mixed")
    if len({q.query_id for q in queries}) != len(queries):
        raise ValueError("duplicate diagnostic variants")
    meanings: dict[tuple[str, str, str], str] = {}
    for query in queries:
        key = (query.query, query.lang, query.corpus_lang)
        if key in meanings and meanings[key] != query.scenario_id:
            raise ValueError("diagnostic question ambiguously names different needs")
        meanings[key] = query.scenario_id
    return queries, metadata


def run_diagnostics(settings: Settings, parent: Path) -> dict[str, Any]:
    import mlflow

    verify_freeze(parent)
    protocol = json.loads((parent / "protocol.json").read_text())
    audit = json.loads((parent / "audit.json").read_text())
    root = settings.resolve(Path(protocol["output"])) / parent.name
    selection_path = root / "selection.json"
    selection = json.loads(selection_path.read_text())
    if selection["development_sha256"] != sha256_file(root / "development/results.json"):
        raise ValueError("sealed development changed")
    source_path = settings.root / "authored/retrieval_diagnostics/phrasing.yaml"
    source = DiagnosticSource.model_validate(yaml.safe_load(source_path.read_text("utf-8")))
    identity = sha256_text(
        json.dumps(
            {
                "parent": sha256_file(parent / "freeze.json"),
                "selection": sha256_file(selection_path),
                "source": sha256_file(source_path),
                "code": sha256_file(Path(__file__)),
                "seed": protocol["seed"],
            },
            sort_keys=True,
        )
    )
    queries, metadata = diagnostic_queries(settings, parent, source, protocol["seed"])
    planned = {
        "source_sha256": sha256_file(source_path),
        "queries_sha256": sha256_text("".join(q.model_dump_json() + "\n" for q in queries)),
        "metadata_sha256": sha256_text(json.dumps(metadata, sort_keys=True)),
        "selection_sha256": sha256_file(selection_path),
    }
    plan_path = root / "diagnostic_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if any(plan[key] != value for key, value in planned.items()):
            raise ValueError("diagnostic wording or target plan changed")
    else:
        write_json(
            plan_path,
            {**planned, "scope": source.scope, "created_at": datetime.now(UTC).isoformat()},
        )
    out = root / "diagnostics" / identity[:20]
    if (out / "results.json").exists():
        raise ValueError("completed diagnostic is immutable")
    out.mkdir(parents=True, exist_ok=True)
    chosen = selection["configurations"][selection["selected"]]
    if chosen["options"]["rerank_k"] != 20:
        raise ValueError("this diagnostic implements the locked top-20 reranking protocol")
    model = load_models(settings.resolve(settings.retrieval.models))[chosen["encoder"]]
    spec = RunSpec(
        experiment="retrieval_scale",
        run_name=f"{parent.name}_wording_diagnostic",
        hypothesis="Label-echo retrieval may not transfer to paraphrases and misspelled entities",
        dataset_version=identity,
        seed=protocol["seed"],
        base_model=model.repo_id + "@" + model.revision,
        params={
            "scope": "post_selection_existing_dev_only",
            "selection_sha256": sha256_file(selection_path),
            "new_independent_needs": 0,
            "queries": len(queries),
            "existing_needs": len({q.scenario_id for q in queries}),
            "code_sha256": sha256_file(Path(__file__)),
            "source_sha256": sha256_file(source_path),
        },
    )
    client = make_qdrant_client(settings.qdrant, settings)
    with start_run(spec, settings) as run:
        attempt = out / f"attempt-{run.info.run_id}.json"
        try:
            sft = current_sft(settings.resolve(settings.paths.data_dir))
            if sft is None:
                raise ValueError("SFT decontamination input missing")
            manifest = read_model_json(sft / "manifest.json", DatasetManifest)
            if verify_files(sft, manifest.files, exclude=frozenset({"manifest.json"})):
                raise ValueError("SFT input changed")
            leaks = leakage_checks(queries, sft, settings, semantic=True)
            near = {row["query"] for row in leaks.get("semantic_near_sft", [])}
            queries = [
                q.model_copy(update={"quality_flags": [*q.quality_flags, "semantic_near_sft"]})
                if q.query in near
                else q
                for q in queries
            ]
            gold_texts: set[str] = set()
            for relative, checksum in audit["source_hashes"].items():
                path = settings.resolve(Path(relative))
                if sha256_file(path) != checksum:
                    raise ValueError(f"parent source changed: {relative}")
                if path.name == "gold.jsonl":
                    for line in path.read_text("utf-8").splitlines():
                        row = json.loads(line)
                        gold_texts.update(
                            m["content"]
                            for m in row.get("turns", row.get("messages", []))
                            if m["role"] == "user"
                        )
            texts = sorted({q.query for q in queries})
            gold = {
                "exact": len(exact_leaks(sorted(gold_texts), texts)),
                "lexical": len(
                    lexical_leaks(
                        sorted(gold_texts),
                        texts,
                        threshold=settings.stage4.leak_jaccard,
                        seed=protocol["seed"],
                    )
                ),
                "ngram": len(ngram_leaks(sorted(gold_texts), texts, settings.stage4.leak_ngram)),
            }
            if not leaks["passed"] or not gold_texts or any(gold.values()):
                raise ValueError("diagnostic SFT/gold leakage gate failed")
            frozen = (
                settings.resolve(settings.paths.data_dir)
                / "benchmarks/retrieval_diagnostics"
                / identity[:20]
            )
            if not frozen.exists():
                with partial_dir(frozen) as staging:
                    (staging / "queries.jsonl").write_bytes(
                        "".join(q.model_dump_json() + "\n" for q in queries).encode()
                    )
                    write_json(staging / "metadata.json", metadata)
                    write_json(
                        staging / "audit.json",
                        {
                            "passed": True,
                            "sft": leaks,
                            "gold": gold,
                            "parent": parent.name,
                            "source_sha256": sha256_file(source_path),
                            "code_sha256": sha256_file(Path(__file__)),
                            "code_commit": code_commit(settings.root),
                            "scope": source.scope,
                            "independent_human_review": False,
                            "row_counts": {
                                "queries": len(queries),
                                "existing_development_needs": len({q.scenario_id for q in queries}),
                                "new_independent_needs": 0,
                            },
                            "sft_version": manifest.dataset_version,
                        },
                    )
                    write_json(
                        staging / "freeze.json",
                        {
                            "schema_version": "retrieval_diagnostic_freeze/1",
                            "input_identity": identity,
                            "created_at": datetime.now(UTC).isoformat(),
                            "files": [f.model_dump(mode="json") for f in scan_files(staging)],
                        },
                    )
                    staging.rename(frozen)
            verify_freeze(frozen)
            tokenizer = load_tokenizer(settings.eda.tokenizers[0])
            scoped = settings.model_copy(
                update={
                    "retrieval": settings.retrieval.model_copy(
                        update={"namespace": "scale_" + parent.name}
                    )
                }
            )
            backend = prepare_backend(
                scoped,
                chosen["encoder"],
                queries,
                metadata,
                protocol,
                audit["kb_version"],
                client,
                tokenizer,
                out,
            )
            artifact = backend["artifact"]
            options = SearchOptions.model_validate(chosen["options"])
            candidates = {
                q.query_id: scored_chunks(
                    artifact,
                    backend["candidates"][q.query_id][0],
                    backend["candidates"][q.query_id][1],
                    options,
                )[:20]
                for q in queries
            }
            pairs = [
                (q.query, f"{r.chunk.title}\n{r.chunk.text}")
                for q in queries
                for r in candidates[q.query_id]
            ]
            pair_settings = settings.model_copy(
                update={
                    "paths": settings.paths.model_copy(
                        update={
                            "data_dir": settings.resolve(settings.paths.data_dir) / "scale_cache"
                        }
                    )
                }
            )
            values, resources = score_pairs(pair_settings, "bge_reranker", pairs, out / "rerank")
            rows, deltas, offset = [], [], 0
            reference = {
                r["query_id"]: r
                for r in json.loads((root / "development/observations.json").read_text())[
                    selection["selected"]
                ]
            }
            for query in queries:
                selected = candidates[query.query_id]
                ranked = [
                    r.model_copy(update={"score": v, "rerank_score": v})
                    for r, v in zip(selected, values[offset : offset + len(selected)], strict=True)
                ]
                offset += len(selected)
                ranked.sort(key=lambda r: (-r.score, r.chunk_id))
                ranked = evidence_gate(ranked, options.evidence_threshold)[:10]
                packed = assemble_context(
                    ranked,
                    tokenizer,
                    chosen["budget"],
                    kb_version=artifact.kb_version,
                    index_version=artifact.manifest.dataset_version,
                )
                meta = metadata[query.query_id]
                row = {
                    "query_id": query.query_id,
                    "scenario_id": query.scenario_id,
                    "family_id": meta["family_id"],
                    "phase": "diagnostic",
                    "answerable": True,
                    "kb_version": artifact.kb_version,
                    "benchmark_version": identity,
                    "ranking": [r.chunk_id for r in ranked],
                    "packed": [r.chunk_id for r in packed.evidence],
                    "slices": sorted(set(slices(query) + meta["tags"])),
                }
                row.update(
                    measure_query(
                        row["ranking"],
                        row["packed"],
                        {r.chunk_id: r.grade for r in query.relevant},
                        meta["required_groups"],
                        answerable=True,
                    )
                )
                rows.append(row)
                if meta["kind"] in ("paraphrase", "entity_typo"):
                    control = reference[meta["reference_query_id"]]
                    if (
                        control["scenario_id"] != query.scenario_id
                        or control["family_id"] != meta["family_id"]
                    ):
                        raise ValueError("diagnostic/reference pairing changed")
                    deltas.append(
                        {
                            **row,
                            **{
                                key: value - control[key]
                                for key, value in row.items()
                                if isinstance(value, float) and key in control
                            },
                        }
                    )
            result = {
                "run_id": run.info.run_id,
                "frozen": str(frozen),
                "selected": selection["selected"],
                "queries": len(rows),
                "existing_development_needs": len({q.scenario_id for q in queries}),
                "new_independent_needs": 0,
                "summary": summarise_scale(rows, seed=protocol["seed"]),
                "paired_wording_delta": summarise_scale(deltas, seed=protocol["seed"]),
                "pairing": (
                    "Same primary native query/need, language, corpus and proof; "
                    "changed wording only. Cross/mixed variants excluded from paired deltas."
                ),
                "scope": source.scope,
                "backend": backend["receipt"],
                "rerank_resources": resources,
            }
            write_json(out / "observations.json", rows)
            write_json(out / "results.json", result)
            mlflow.log_artifact(str(out / "results.json"))
            mlflow.log_artifact(str(out / "observations.json"))
            mlflow.log_artifact(str(frozen / "audit.json"))
            mlflow.set_tags({"decision": "diagnostic_only_no_retuning", "cost_usd": "0"})
            write_json(attempt, {"status": "FINISHED", "run_id": run.info.run_id})
            return {"path": str(out), "run_id": run.info.run_id}
        except BaseException as exc:
            write_json(
                attempt,
                {
                    "status": "FAILED",
                    "run_id": run.info.run_id,
                    "error": type(exc).__name__,
                    "detail": str(exc),
                },
            )
            mlflow.log_artifact(str(attempt))
            raise
        finally:
            client.close()

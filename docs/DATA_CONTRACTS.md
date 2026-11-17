# DATA_CONTRACTS — schemas, identifiers, invariants

Code: `common/schemas.py`, `common/ids.py`, `common/manifest.py`, `ops/backup.py`.
A breaking change needs a new `schema_version` and an ADR.

## 1. Conventions

- Languages: ISO 639-1 `en`, `ru`, `kk`. Kazakh output in Cyrillic.
  Mixed text: dominant `lang` + `code_switch: true`.
- Slugs (`source_id`, dataset names): `^[a-z0-9]+([_-][a-z0-9]+)*$`.
- Dataset versions: `{name}-{yyyymmdd}-{sha256[:8]}`.
- Timestamps: UTC ISO-8601. Money: integer minor units + ISO 4217 currency (`KZT`).
- Text: UTF-8, NFC, `\n` newlines. Originals stay untouched in raw storage.
- Manifest paths: relative POSIX, no `..` (enforced by `FileEntry`).

## 2. Source registry (`configs/sources.yaml`, `sources/1`)

Every investigated source is listed, including rejected ones with the reason.

| Field | Notes |
|---|---|
| `source_id` | unique slug |
| `url`, `revision` | canonical page; full 40-char commit SHA |
| `licence`, `licence_evidence` | licence id from the policy; where/when it was checked |
| `allowed_use` | `train` > `eval_only` > `reference_only` > `blocked`, capped by licence |
| `status` | `selected` (acquired) · `candidate` · `rejected` · `unavailable` |
| `access` | `public` or `gated` (needs `HF_TOKEN`) |
| `languages`, `size`, `reviewed_at` | |
| `purpose`, `decision`, `notes` | why it is used or rejected |
| `fetch` | `{type: hf_dataset \| github_repo, repo_id, include}`; required when `selected` |

Minimum `allowed_use` per step (`data/policy.py`): acquire, convert, analysis → `reference_only`
(acquire also needs status ≠ `rejected`); evaluation → `eval_only`; training → `train`.

## 3. Raw storage (`data/raw/{source_id}/{revision}/`)

- Original files exactly as downloaded + `manifest.json` (`raw_manifest/1`:
  `source_id, revision, url, retrieved_at, licence, repo_id, files[{path, bytes, sha256}]`).
- Each file is checked against the upstream LFS SHA256 or git blob SHA1 before acceptance.
- Never modified. A failed verification stops the pipeline; restore from backup.
- In-progress downloads: `data/raw/{source_id}/.partial-{revision}/`, resumable.

## 4. Intent taxonomy (`configs/taxonomy.yaml`, `taxonomy/1`)

Authoritative list of intents, actions, tools, escalation reasons and conditions.
`common.schemas.INTENTS` must equal it (unit test). Do not copy the table elsewhere.

## 5. Snapshot manifest (`snapshot/1`, `supportpilot backup`)

`snapshot_id` (`{yyyymmddTHHMMSSZ}-{content_sha256[:8]}`), `name`, `kind` (`data`|`model`),
`created_at`, `source_path`, `code_commit`, `note`, `pinned`, `content_sha256`,
`archive_sha256`, `archive_bytes`, `total_bytes`, `files`.

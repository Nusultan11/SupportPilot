workspace "SupportPilot" "Guarded multilingual local prototype and its offline engineering pipeline; planned capabilities are explicitly tagged." {
    !identifiers hierarchical
    configuration {
        scope softwaresystem
    }
    model {
        customer = person "Customer" "Synthetic customer asking RU, KK, EN or mixed RU-KK store questions."
        operator = person "Operator" "Authorized local operator reviewing recorded assistance requests."
        engineer = person "Engineer / authorized agent" "Develops one owner-authorized scope and verifies evidence."
        sources = softwareSystem "Licensed sources and model Hub" "Pinned datasets and model revisions, subject to source/model registries."
        kaggle = softwareSystem "Authorized free Kaggle compute" "Private versioned training/assessment jobs; terminal outputs require checksum verification."
        sp = softwareSystem "SupportPilot" "Fictional-store grounded support research system; autonomous factual support NOT_READY." {
            properties {
                "documentation" "docs/ARCHITECTURE.md"
                "decisions" "docs/DECISIONS.md"
                // Owning Markdown documents stay external; do not duplicate their contents here.
                "structurizr.inspection.model.softwaresystem.documentation" "info"
                "structurizr.inspection.model.softwaresystem.decisions" "info"
            }
            ui = container "Local UI" "Chat, labeled source excerpts, diagnostics and authorized ticket listing; no status editor. apps/ui.py." "Streamlit / HTTP" "Implemented"
            online = container "Guarded local service" "One Python worker/process; bounded requests and process-local sessions. src/supportpilot/service/runtime.py." "Python 3.12 / FastAPI / PyTorch" "Implemented" {
                api = component "HTTP boundary" "Auth principals, chat/health/ready/escalations; api.py." "FastAPI / Pydantic" "Implemented"
                orchestration = component "Orchestrator and sessions" "Independent risk guards, interpreted intent/action, explicit confirmed tools and bounded sessions; service/orchestrator.py. Pending service/decisions.py continuation is unconfirmed." "Python / single worker" "Implemented"
                lid = component "Language identification" "Dominant language and mixed detection; data/langid.py." "Lingua" "Implemented"
                router = component "Intent/action router" "Frozen CPU encoder head and typed decisions; routing/. Uncalibrated." "E5-small / classifier" "Implemented"
                tools = component "Authorized business tools" "Owned orders/payments, shared stock, cancellation requests; service/store.py." "SQLite transactions" "Implemented"
                retrieval = component "Hybrid retrieval" "Active KB/index readiness, dense+sparse search and filters; retrieval/search.py, index.py." "BGE-M3 / Qdrant" "Implemented"
                packing = component "Reranking and context packing" "Canonical chunks with locators, rerank20, budgeted context; retrieval/search.py." "BGE reranker / token budget" "Implemented"
                generator = component "Research generation" "Optional frozen adapter; off by default and all service KB candidates withheld; llm/backend.py, generation.py." "Transformers / PEFT" "Implemented,Experimental"
                verification = component "Response boundary" "Library schema/citation/number/language checks then explicit support verdict; service still withholds every KB candidate. common/grounding.py, llm/generation.py." "Pydantic / ClaimVerifier protocol" "Implemented"
                queue = component "Durable assistance queue" "Transactional queue and idempotent delivery receipts; service/store.py. No external handoff promise." "SQLite" "Implemented"
                operations = component "Operational controls" "A5/A6: dependency diagnostics, bounded admission, security and retention-growth controls in existing service; Stage 10P2/10P3, not implemented." "Planned Python boundary" "Planned"
                calibrated = component "Calibrated semantic verifier" "A2: independently calibrated ClaimVerifier plus explicit fail-closed service acceptance path, unavailable today. Stage 9Q2/9Q3." "UNDECIDED" "Planned"
            }
            offline = container "Offline pipeline / CLI" "Existing package reused by local commands and authorized cloud payloads; cli.py. A process role, not a new service." "Python 3.12 / uv" "Implemented" {
                acquisition = component "Source acquisition" "Registry rights, pinned resumable downloads and immutable conversion; data/acquire.py, convert.py." "Hub / manifests" "Implemented"
                preparation = component "Preparation and EDA" "Normalization, PII masking, dedup, gates and source/language diagnostics; data/pipeline.py, eda.py." "Polars / Parquet" "Implemented"
                datasets = component "Multilingual datasets" "Checkpointed MT/QE, altered training facts and scenario/family isolation; data/sft.py, gold.py." "Pinned MT / quality gates" "Implemented"
                rendering = component "Fictional document rendering" "Canonical facts to six formats; kb/render.py." "PDF / DOCX / XLSX / CSV / JSON / TXT" "Implemented"
                extraction = component "Document extraction" "Format parsers with explicit scan/structure errors; kb/parse_pdf.py, parse_office.py, parse_text.py." "pdfplumber / python-docx / openpyxl" "Implemented"
                structure = component "Structure and chunking" "Section trees, tables, source locators, gates, immutable publication; kb/structure.py, chunk.py, build.py." "Typed KB contracts" "Implemented"
                indexing = component "Index and synchronization" "Checksummed embeddings, portable indexes and fail-closed activation; retrieval/index.py." "Dense / sparse / writer locks" "Implemented"
                training = component "Behaviour fine-tuning" "Completion-only LoRA/QLoRA, checkpoints/resume, selection seals; llm/train.py." "TRL / PEFT / PyTorch" "Implemented"
                evaluation = component "Evaluation and provenance" "Per-language ranking/routing/generation and draft dialogue/critical outcomes; independent-review calibration contract, overlap/resource/load studies. retrieval/, llm/, service/." "Draft labels / immutable reports" "Implemented"
                release = component "Release qualification" "A4/A7/A8: bind code/config/model/router/prompt/KB/index to independent quality, capacity and recovery evidence; Stage 10P1/10P4, not implemented." "Existing manifests / planned release contract" "Planned"
                backup = component "Snapshot and recovery" "Pinned SHA256 archives, deep verification and actual isolated restores; ops/backup.py." "tar / manifests / Git bundles" "Implemented"
            }
            files = container "Versioned local files" "data/: raw/canonical/SFT/gold/KB/indexes; artifacts/: models/checkpoints/payloads/reports. Private ignored paths." "Filesystem / Parquet / JSON / safetensors" "Implemented"
            vector = container "Vector index" "Dense+sparse materialized collections, isolated load collections and named persistent volume." "Qdrant 1.20 / Docker" "Implemented"
            business = container "Business state and delivery" "Synthetic orders/payments/stock, queue and serialized response replay receipts; no expiry policy. data/service/. Sessions not persisted." "SQLite" "Implemented"
            tracking = container "Experiment tracking" "Local mlruns/mlflow.db plus artifact files; not a required online service." "MLflow / SQLite / filesystem" "Implemented"
            archives = container "Recovery archives" "Ignored pinned backups and code bundles; same disk, no disk-loss protection." "tar / SHA256 / Git bundle" "Implemented"
        }
        engineering = softwareSystem "Local engineering MCP tools" "Optional Shrimp stdio with private persistent tasks and local DSL-only Structurizr HTTP. tools/mcp/; never a runtime dependency."
        customer -> sp.ui "Uses multilingual chat" "HTTP loopback"
        operator -> sp.ui "Lists authorized recorded tickets" "Authenticated HTTP"
        operator -> sp.online.api "Reviews queue and performs allowed ticket transitions" "Operator bearer HTTP"
        engineer -> sp.offline "Runs authorized pipelines and checks" "CLI"
        engineer -> engineering "Plans/verifies internal tasks and validates the canonical DSL" "MCP stdio / HTTP loopback"
        sp.ui -> sp.online.api "Chat and ticket requests" "Bearer HTTP loopback"
        sp.online.api -> sp.online.orchestration "Validates and dispatches authorized requests" "Python calls"
        sp.online.api -> sp.online.operations "Future admission and operational diagnostics; current ready checks local worker/SQLite only" "Planned Python boundary" {
            tags "Planned"
        }
        sp.online.orchestration -> sp.online.lid "Detects reply language" "Bounded component work"
        sp.online.orchestration -> sp.online.router "Routes freeform requests" "Bounded CPU work"
        sp.online.orchestration -> sp.online.tools "Executes explicit authorized operations" "Python / transaction"
        sp.online.orchestration -> sp.online.retrieval "Retrieves current evidence" "Bounded component work"
        sp.online.retrieval -> sp.online.packing "Ranks and packs canonical evidence" "Python calls"
        sp.online.orchestration -> sp.online.generator "Optionally requests an untrusted candidate" "Research mode only"
        sp.online.generator -> sp.online.verification "Checks candidate against packed evidence" "Python calls"
        sp.online.verification -> sp.online.orchestration "Returns diagnostics and untrusted result; current service queues every KB candidate" "GenerationResult"
        sp.online.verification -> sp.online.calibrated "Future calibrated support verdict; unavailable today" "Planned Python boundary" {
            tags "Planned"
        }
        sp.online.orchestration -> sp.online.queue "Commits assistance and receipt before reply" "Transaction"
        sp.online.tools -> sp.business "Reads state / records confirmed requests" "SQLite"
        sp.online.queue -> sp.business "Persists queue and replay receipts" "SQLite"
        sp.online.retrieval -> sp.vector "Searches ready matching-version collections" "REST loopback"
        sp.online.retrieval -> sp.files "Checks active KB/index and canonical chunks" "Local files"
        sp.online.router -> sp.files "Loads sealed classifier and pinned weights" "Local files"
        sp.online.generator -> sp.files "Loads verified inference export" "Local files"
        sp.offline.acquisition -> sources "Downloads approved pinned source bytes" "HTTPS"
        sp.offline.acquisition -> sp.files "Retains immutable raw revisions and conversion manifests" "Local files"
        sp.offline.acquisition -> sp.offline.preparation "Provides immutable converted records" "Parquet / manifest"
        sp.offline.preparation -> sp.files "Publishes canonical records, gates and EDA evidence" "Atomic manifests"
        sp.offline.preparation -> sp.offline.datasets "Provides clean family-grouped inputs" "Parquet"
        sp.offline.datasets -> sp.files "Publishes gated SFT/gold/retrieval inputs" "Atomic manifests"
        sp.offline.rendering -> sp.offline.extraction "Provides synthetic canonical documents" "Six file formats"
        sp.offline.extraction -> sp.offline.structure "Provides parsed blocks and locators" "Typed blocks"
        sp.offline.structure -> sp.files "Publishes immutable KB and active pointer" "Atomic manifests"
        sp.offline.indexing -> sp.files "Reads KB; publishes checksummed portable index/cache" "Local files"
        sp.offline.indexing -> sp.vector "Synchronizes and verifies collections" "REST loopback"
        sp.offline.training -> sp.files "Reads frozen SFT; writes checkpoints and inference export" "Local files"
        sp.offline.training -> kaggle "Submits only authorized private free jobs" "Kaggle CLI / HTTPS"
        sp.offline.evaluation -> sp.files "Reads frozen inputs; retains raw outcomes and failed attempts" "Local files"
        sp.offline.evaluation -> sp.online.api "Exercises actual guarded runtime in isolated assessments" "FastAPI test/HTTP boundary"
        sp.offline.evaluation -> sp.tracking "Logs provenance and scoped metrics" "MLflow Python API"
        sp.offline.release -> sp.files "Future verifies qualified release binding and immutable component identities" "Planned manifest checks" {
            tags "Planned"
        }
        sp.offline.release -> sp.online "Future starts and qualifies existing service on declared workload" "Planned release / HTTP verification" {
            tags "Planned"
        }
        sp.offline.backup -> sp.files "Snapshots and verifies selected private results" "SHA256 inventory"
        sp.offline.backup -> sp.business "Snapshots closed database state" "SQLite files"
        sp.offline.backup -> sp.tracking "Preserves closed tracking store and artifacts" "SHA256 inventory"
        sp.offline.backup -> sp.archives "Creates and restore-tests pinned archives" "Local files"
        sp.offline.backup -> engineering "Snapshots closed private task state" "SHA256 inventory"
    }
    views {
        systemContext sp "SystemContext" "Current guarded system and external actors/dependencies." {
            include *
            autoLayout lr
        }
        container sp "Containers" "Process roles and persistent stores; planned production remains separate." {
            include *
            autoLayout lr
        }
        component sp.online "OnlineComponents" "Implemented online boundaries and the missing calibrated semantic verifier." {
            include *
            autoLayout lr
        }
        component sp.offline "OfflineComponents" "Acquisition through recovery; evaluation and controlled cloud execution." {
            include *
            autoLayout lr
        }
        styles {
            element "Person" {
                shape Person
                background #24476b
                color #ffffff
            }
            element "Implemented" {
                background #d5e8d4
                color #182b18
            }
            element "Experimental" {
                background #eadcf4
                color #4b2863
                border Dashed
            }
            element "Planned" {
                background #fff2cc
                color #664d00
                border Dashed
            }
        }
    }
}

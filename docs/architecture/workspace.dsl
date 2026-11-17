workspace "SupportPilot" "Engineering foundation after the 2026-11-17 reset (ADR-046); rebuilt capabilities are tagged Planned." {
    !identifiers hierarchical
    configuration {
        scope softwaresystem
    }
    model {
        customer = person "Customer" "Synthetic customer asking RU, KK, EN or mixed RU-KK store questions."
        engineer = person "Engineer / authorized agent" "Develops one owner-authorized scope and verifies evidence."
        sources = softwareSystem "Licensed sources and model Hub" "Pinned dataset revisions on Hugging Face Hub and GitHub; eligibility in configs/sources.yaml."
        cloud = softwareSystem "Authorized free cloud compute" "Larger jobs beyond the 4 GB local GPU; owner-authorized only."
        sp = softwareSystem "SupportPilot" "Fictional-store multilingual support assistant; data and ML pipelines restart from scratch." {
            properties {
                "documentation" "docs/ARCHITECTURE.md"
                "decisions" "docs/DECISIONS.md"
                // Owning Markdown documents stay external; do not duplicate their contents here.
                "structurizr.inspection.model.softwaresystem.documentation" "info"
                "structurizr.inspection.model.softwaresystem.decisions" "info"
            }
            cli = container "supportpilot CLI" "Command-line orchestration; src/supportpilot/cli.py." "Python 3.12" "Implemented" {
                common = component "Common foundation" "Settings, redacted logging, hashing, slugs, schemas, manifests, MLflow tracking; common/." "Python / Pydantic" "Implemented"
                acquire = component "Source acquisition" "Pinned download, upstream hash verification, atomic publication; data/acquire.py." "huggingface_hub" "Implemented"
                policy = component "Licence policy" "Allowed use per pipeline step; data/policy.py." "Python" "Implemented"
                taxonomy = component "Taxonomy loader" "Validates intent taxonomy v1; data/taxonomy.py." "Pydantic" "Implemented"
                ops = component "Operations" "Snapshots, environment report, MLflow health check; ops/." "Python / tar / SHA256" "Implemented"
            }
            pipeline = container "Data and ML pipeline" "Dataset build, training, evaluation; rebuilt stage by stage (ROADMAP)." "Python" "Planned" {
                dataset = component "Dataset builder" "Versioned multilingual dataset with family-disjoint splits." "Polars / Parquet" "Planned"
                models = component "Model training and evaluation" "Routing and response models with per-language metrics." "PyTorch / MLflow" "Planned"
                service = component "Support service" "Routing, evidence, tools, clarification and escalation." "Python" "Planned"
            }
            configs = container "Configuration" "base.yaml, sources.yaml, taxonomy.yaml; in Git." "YAML" "Implemented"
            raw = container "Raw data store" "data/raw/<source>/<revision>/ with manifests; immutable." "Local files" "Implemented"
            tracking = container "Experiment tracking" "mlruns/ SQLite and artifacts." "MLflow" "Implemented"
            archives = container "Backups" "backups/ code bundles and data/model snapshots; never deleted without owner permission." "tar / SHA256" "Implemented"
        }

        engineer -> sp.cli "Runs commands" "Terminal"
        customer -> sp.pipeline.service "Asks questions" "Chat (planned)"
        sp.cli.acquire -> sources "Downloads pinned revisions" "HTTPS"
        sp.cli.acquire -> sp.cli.policy "Checks allowed use" "Python call"
        sp.cli.acquire -> sp.raw "Publishes verified files" "Atomic rename"
        sp.cli.acquire -> sp.cli.common "Writes manifests" "Python call"
        sp.cli.taxonomy -> sp.configs "Reads taxonomy.yaml" "YAML"
        sp.cli.common -> sp.configs "Reads settings and registry" "YAML"
        sp.cli.common -> sp.tracking "Logs runs" "MLflow / SQLite"
        sp.cli.ops -> sp.archives "Creates, verifies, restores snapshots" "tar / SHA256"
        sp.cli.ops -> sp.cli.common "Uses manifests and settings" "Python call"
        sp.pipeline.dataset -> sp.raw "Reads raw sources" "Local files"
        sp.pipeline.models -> sp.pipeline.dataset "Trains and evaluates on versioned splits" "Parquet"
        sp.pipeline.models -> sp.tracking "Logs experiments" "MLflow / SQLite"
        sp.pipeline.models -> cloud "Runs larger jobs" "Authorized only"
        sp.pipeline.service -> sp.pipeline.models "Uses selected models" "Python call"
    }
    views {
        systemContext sp "SystemContext" "SupportPilot after the reset and its external dependencies." {
            include *
            autoLayout lr
        }
        container sp "Containers" "Implemented foundation and the planned pipeline." {
            include *
            autoLayout lr
        }
        component sp.cli "CliComponents" "Implemented CLI components." {
            include *
            autoLayout lr
        }
        component sp.pipeline "PipelineComponents" "Planned data, ML and service components." {
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
            element "Planned" {
                background #fff2cc
                color #664d00
                border Dashed
            }
        }
    }
}

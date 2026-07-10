# LLM Instructions for Yomitori OCR Engine

You are an expert Python/ML developer operating within the "Yomitori" OCR engine project. 
Adhere to the following absolute mandates at all times.

## 🏛️ Architectural Mandates
1. **Plugin Architecture (プラグイン型アーキテクチャ):**
   - Adding support for new identity document types MUST only be done by adding a YAML configuration to `configs/document_types/`, a DocumentType Python class to `src/document_types/`, and post-processing validation/normalization rules to `src/postprocessing/`.
   - Core pipeline code (`src/pipeline/`, `src/detection/`, `src/recognition/`, `src/preprocessing/`) MUST NOT be modified when adding new document types.
2. **License Guardrails (ライセンス制限):**
   - DO NOT introduce any AGPL/GPL dependencies (such as PaddleOCR). All libraries must use permissive licenses (Apache 2.0, MIT, BSD, etc.).
   - Always run the license checker `scripts/check_licenses.sh` when adding or upgrading python packages.
3. **Python Package Layout:**
   - Keep application logic inside the `src/` package.
   - Keep CLI entry points in `src/cli.py`.
   - Keep SageMaker integration scripts in `sagemaker/` and training routines in `training/`.
4. **YAGNI (You Aren't Gonna Need It):**
   - Avoid excessive abstractions or generic wrappers. Follow existing patterns.

## 🤖 Agentic Workflow
1. **Self-Correction:** 
   - When generating or editing code, always attempt to test it locally.
   - Run tests using `pytest tests/ -v`.
   - Run the local inference tests: `bash scripts/run_local_test.sh data/samples/sample_license.jpg`.
   - If it fails, analyze the output and fix it (max 3 attempts).
2. **Test Coverage:** 
   - All new logic and custom document fields/rules MUST be accompanied by unit tests in the `tests/` directory.
3. **License Verification:**
   - Run `bash scripts/check_licenses.sh` to ensure compliance.

## 🛠️ Tool Usage
- Use `pytest` for testing.
- Use `ruff` for linting and formatting (if configured).
- Use `scripts/check_licenses.sh` to verify package licensing.

## 📁 Subagent Preferences
- **Primary Workspace:** Always utilize the `.shared-agents` directory as the primary location for subagent operations, logs, rules, and intermediate artifacts. Avoid creating or using `.agents` or other new directories for this purpose.
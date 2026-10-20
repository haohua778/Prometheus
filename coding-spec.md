# Computational verification subagent

**Author:** Pierre project / Codex
**Date:** 2026-10-07
**Status:** Approved implementation plan; cross-team interface remains a draft.
**Reviewers:** Project owner; Review Agent teammate for the interface draft.

## Context

Review Agent delegates numerical checks against submitted attachments. The child returns evidence for a specific claim and does not create flags or speak directly to users. Existing model configuration, LangGraph and structured runtime logs are reused.

The owner approved a Python internal entry point and CSV/JSON inputs. Docker-dependent acceptance passed on 2026-10-15 on Linux Mint with Docker Engine, including real-model checks; on Windows, Docker Desktop must be restored and the image rebuilt before the same tests can run.

## Functional Requirements

- FR-1: The module MUST expose async run_check(CheckRequest, authorized_files=...) returning CheckResult.
- FR-2: Requests MUST identify the claim, question and authorized file IDs; invalid inputs MUST fail before model execution.
- FR-3: Each call MUST use independent messages, execution records, callback state and sandbox resources, with no persistent graph checkpoints.
- FR-4: The agent MUST use sequential inspect_data and run_code tools; each code execution MUST start a fresh Python process.
- FR-5: supports/refutes MUST require inspected authorized evidence and a successful, non-truncated code execution. Missing evidence MUST return inconclusive. Returned status, verdict and error_code MUST be consistent; failed checks MUST carry an error code.
- FR-6: Success, failure, timeout and cancellation MUST clean task resources; cancellation MUST propagate.
- FR-7: The module MUST preserve parent request correlation and MUST NOT log raw code, data or output to application stdout logs.

## Non-Functional Requirements

- NFR-1: Tasks MUST have a 1–60 second execution budget including queuing, at most six model calls, three executions and 20 graph steps. Cleanup has independent bounded Docker operations.
- NFR-2: Inputs MUST be at most five regular files of 10 MiB each; links and junctions MUST be rejected.
- NFR-3: Containers MUST have a read-only root/input, non-root user, no external network, no added capabilities, no privilege escalation, 1 CPU, 512 MiB memory without additional swap and 64 processes.
- NFR-4: Temporary work MUST be limited to 64 MiB; stdout/stderr MUST each be limited to 32 KiB; inspections MUST be limited to five sample records and 16 KiB.
- NFR-5: A single event-loop service process MUST allow at most two active tasks, including cleanup.

## Acceptance Criteria

### AC-1: Request validation (FR-1, FR-2, NFR-2)
Given invalid fields, unknown IDs or oversized files, When validation runs, Then the call raises an explicit validation error.

### AC-2: Task isolation (FR-3, NFR-5)
Given concurrent tasks, When they run, Then their correlation IDs and mutable state stay independent; queued timeout/cancellation does not release another task's slot.

### AC-3: Calculation (FR-4, FR-5)
Given known CSV/JSON records, When real code executes, Then arithmetic results are correct, code errors can be corrected and Python variables do not persist.

### AC-4: Cleanup (FR-6, NFR-1, NFR-4)
Given infinite loops, cancellation or output overflow, When execution ends, Then task containers and staged files are removed.

### AC-5: Expiry (FR-6)
Given an expired owner-labeled container, When initialization cleans it, Then the container and its generated staging directory are removed.

### AC-6: Real model (FR-5)
Given real model tasks with supported, contradicted or missing-period evidence, When the graph completes, Then the verdict and cited file IDs match the case.

### AC-7: Logs (FR-7)
Given runtime events, When logged, Then parent and check IDs remain correlated without raw payloads.

### AC-8: Container restrictions (NFR-3)
Given a real task container, When inspected and exercised, Then resource settings, immutable inputs and isolated temporary storage are verified.

## Edge Cases

- EC-1: Docker unavailable, missing image or unexpected container termination: explicit environment_error, no fake execution.
- EC-2: Model API failure or missing key: model_error; malformed final JSON or unsupported evidence: result_invalid.
- EC-3: Malformed UTF-8 CSV/JSON or excessive preview: data_invalid/output_limit, inconclusive.
- EC-4: Cleanup failure: failed/inconclusive and cleanup_failed; log and retain diagnostic identifiers.
- EC-5: Service crash: no graph recovery; expired task containers are reclaimed at the next runtime initialization.

## API Contracts

Python entry point: async run_check(request: CheckRequest, *, authorized_files: Mapping[str, Path]) -> CheckResult.

LangChain tool entry point: make_check_tool(*, authorized_files: Mapping[str, Path]) -> StructuredTool.
The returned check_claim tool accepts the CheckRequest fields and returns a JSON-compatible CheckResult object.
Each tool captures a snapshot of one review's trusted file mapping; file paths are excluded from its input schema.
Every invocation delegates to run_check and therefore retains its independent state, budgets and cleanup.
The parent must provide allowed file IDs, formats and existing flag IDs in its context, correlate returned IDs,
and treat incomplete/failed outcomes as unverified even when the tool invocation itself succeeds.
The current chat service does not register this review-specific tool. Cross-team integration is not yet verified.

CheckRequest: request_id, flag_id, claim, question, data_refs: [{file_id, format: csv|json}], timeout_s=60.

CheckResult: request_id, flag_id, status: completed|incomplete|failed, verdict: supports|refutes|inconclusive, explanation, evidence, executions, runtime_ms, error_code.

Validation errors propagate; expected operational errors return explicit results; cancellation and unexpected programming errors propagate after cleanup.

## Data Models

| Object | Fields and constraints |
| --- | --- |
| Evidence | file_id, nonempty location, JSON observed_value |
| Execution | code, stdout, stderr, nullable exit_code, runtime_ms >= 0, output_truncated |
| Graph State | request, messages with add_messages, model_calls, executions, nullable result |
| Runtime Context | Per-call sandbox and tool objects; not checkpointed |
| Attachment | UTF-8 CSV with unique headers or JSON array of record objects |

The public entry and task lifecycle live in agent.py. The main-agent tool adapter lives in integration.py. Graph nodes, state and model-result validation live in graph.py. Internal tool declarations live in tools.py; Docker and material lifecycle live in sandbox.py. Elapsed durations must be finite and nonnegative. Inspector output is UTF-8 with one LF newline, included in the 16 KiB budget.

## Out of Scope

- OS-1: Direct user chat, HTTP routes, databases, external fact verification, code suggestions, automatic execution of uploaded scripts and Excel/R/SQL execution are excluded.
- OS-2: Persistent memory, service recovery and a confirmed Databricks executor are excluded. No database migration applies. Final records are persisted by the parent service.

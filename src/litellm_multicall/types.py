from dataclasses import dataclass, field
from typing import Any, List, Optional

@dataclass
class RequestContext:
    logical_request_id: str
    requested_public_model: str
    policy_id: str
    policy_version: int = 1
    authenticated_principal_ref: Optional[str] = None
    deadline_ms: Optional[int] = None

@dataclass
class ChildContext:
    logical_request_id: str
    child_call_id: str
    stage: str
    candidate_id: Optional[str] = None
    expansion_depth: int = 1

@dataclass
class Candidate:
    candidate_id: str
    model_group: str
    complete_assistant_message: str = ""
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)
    status: str = "pending"

@dataclass
class ExecutionResult:
    final_assistant_message: str = ""
    finish_reason: str = ""
    selected_candidate_id: Optional[str] = None
    aggregate_usage: dict = field(default_factory=dict)


"""Backend-constrained intermediate formats, independent of the public answer."""


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


STRING_LIST = {"type": "array", "items": {"type": "string"}}
RECORD = object_schema({"quote": {"type": "string"}, "fact": {"type": "string"}})
EVIDENCE = object_schema(
    {
        "status": {"type": "string", "enum": ["complete", "incomplete"]},
        "records": {"type": "array", "items": RECORD},
        "needs": STRING_LIST,
    }
)
VERIFICATION = object_schema(
    {
        "supported": {"type": "boolean"},
        "checked_record_ids": STRING_LIST,
        "issues": STRING_LIST,
    }
)
REVIEW = object_schema({"issues": STRING_LIST, "suggestions": STRING_LIST})
COMPACTION = object_schema(
    {
        "summary": {"type": "string", "minLength": 1},
        "unresolved": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "record_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
    }
)
OBSERVATION = object_schema({"text": {"type": "string"}, "evidence_refs": STRING_LIST})
STATE_FIELDS = (
    "facts",
    "events",
    "claims",
    "hypotheses",
    "conflicts",
    "open_questions",
)
REASONING_STATE = object_schema(
    {
        **{
            key: {"type": "array", "items": OBSERVATION, "maxItems": 32}
            for key in STATE_FIELDS
        },
        "evidence_refs": {"type": "array", "items": {"type": "string"}, "maxItems": 32},
    }
)
EVIDENCE_LOOKUP = object_schema(
    {
        "evidence_refs": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "query": {"type": "string", "maxLength": 256},
    }
)


def artifact_parameters(model, kind="evidence", batched=False):
    schema = {
        "evidence": EVIDENCE,
        "verification": VERIFICATION,
        "review": REVIEW,
        "compaction": COMPACTION,
        "reasoning_state": REASONING_STATE,
        "evidence_lookup": EVIDENCE_LOOKUP,
    }[kind]
    if batched:
        schema = object_schema(
            {
                "partitions": {
                    "type": "array",
                    "items": object_schema(
                        {"partition_id": {"type": "string"}, **EVIDENCE["properties"]}
                    ),
                }
            }
        )
    response_format = (
        {
            "type": "json_schema",
            "json_schema": {"name": "prism_" + kind, "strict": True, "schema": schema},
        }
        if "json_schema" in model.capabilities
        else {"type": "json_object"}
    )
    return {"temperature": 0, "response_format": response_format}

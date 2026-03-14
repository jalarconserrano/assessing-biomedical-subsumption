# -*- coding: utf-8 -*-

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field, confloat
from tqdm import tqdm

from openai import OpenAI


# -----------------------------
# Output Schema (Pydantic)
# -----------------------------
Axis = Literal["semantic", "logical", "instruction_following", "mixed", "uncertain"]
ErrorType = Literal[    
    "semantic_definition_error",
    "semantic_scope_mismatch",
    "semantic_polysemy_or_ambiguity",
    "semantic_domain_mismatch",    
    "logical_directionality_error",
    "logical_overgeneralization",
    "logical_invalid_inference",
    "logical_taxonomic_rule_misapplied"    
]

class QualitativeErrorAnalysis(BaseModel):
    is_error: bool = Field(description="True if the evaluated model made an error on this sample.")
    axis: Axis = Field(description="Primary error axis.")
    error_type: ErrorType = Field(description="Specific error type label.")
    confidence: confloat(ge=0.0, le=1.0) = Field(description="Confidence score in [0, 1].")
    short_explanation: str = Field(description="Short explanation (2–5 sentences).")
    evidence: List[str] = Field(
        default_factory=list,
        description="Up to 6 short evidence bullets grounded in the reasoning and/or term descriptions.",
        max_length=6,
    )

# ---------------------------------------
# 2) Input loading
# ---------------------------------------
def load_input(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(path,sep='|')
    elif ext in (".jsonl", ".ndjson"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .jsonl/.ndjson")
    df.rename(columns={"answer": "gold"}, inplace=True)
    required = {"parentdesc", "childdesc", "gold", "pred", "reasoning"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df

# -----------------------------
# 3) Prompting
# -----------------------------
def build_messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    system = """
You are an expert reviewer in biomedical terminologies (e.g., SNOMED CT) and qualitative evaluation of LLM reasoning.
Your task is to analyze a single model decision given:
- parentdesc (candidate parent concept description)
- childdesc (candidate child concept description)
- gold label for the taxonomical “is-a” relation (true/false)
- model prediction pred (true/false)
- the model’s free-text reasoning

You must classify the principal *type of error* and return a structured output that matches the requested JSON schema.

Use the following error taxonomy (pick the best single label):
- semantic_definition_error: the model uses an incorrect definition for at least one term.
- semantic_scope_mismatch: the model partially understands the terms but misjudges their semantic scope (too broad/too narrow).
- semantic_polysemy_or_ambiguity: the failure is driven by lexical ambiguity/polysemy of key words.
- semantic_domain_mismatch: the model assumes the wrong domain/context (e.g., treats a clinical term as non-clinical, confuses “in SNOMED CT” scope, etc.).

- logical_directionality_error: the model detects relatedness but inverts the parent–child direction.
- logical_overgeneralization: the model incorrectly assumes membership in a broader category.
- logical_invalid_inference: the conclusion does not follow from the stated premises (non sequitur).
- logical_taxonomic_rule_misapplied: the model applies the wrong rule for “is-a” (e.g., part-of vs is-a, site/usage/causality conflations).
 
Return structured output that matches the requested schema.
"""

    user = ("Analyze this case:\n"
        f"- parentdesc: {row['parentdesc']}\n"
        f"- childdesc: {row['childdesc']}\n"
        f"- gold (is-a): {row['gold']}\n"
        f"- pred (is-a): {row['pred']}\n"
        f"- reasoning (evaluated model): {row['reasoning']}\n\n"
        "Notes:\n"
        "1) Set is_error = (pred != gold), unless you detect a formatting/encoding issue in gold/pred.\n"
        "2) Keep the explanation concise (2–5 sentences) and actionable.\n"
        "3) Provide up to 6 evidence bullets grounded in the reasoning.\n"
    )

    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_sleep: float = 1.0
    max_sleep: float = 20.0

def call_with_retry(client: OpenAI, model: str, messages: List[Dict[str, str]], retry: RetryConfig) -> QualitativeErrorAnalysis:
    last_err: Optional[Exception] = None
    for attempt in range(retry.max_retries):
        try:
            # Structured Outputs + parse en Responses API (Pydantic)
            resp = client.responses.parse(
                model=model,
                input=messages,
                text_format=QualitativeErrorAnalysis,
            )
            parsed = resp.output_parsed
            if parsed is None:
                raise RuntimeError("Could not parse model output (output_parsed is None).")
            return parsed
        except Exception as e:
            last_err = e
            # backoff con jitter
            sleep = min(retry.max_sleep, retry.base_sleep * (2 ** attempt))
            sleep = sleep * (0.75 + 0.5 * random.random())
            time.sleep(sleep)
    raise RuntimeError(f"Request failed after retries. Last error: {last_err}") from last_err

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="CSV or JSONL with parentdesc, childdesc, gold, pred, reasoning")
    ap.add_argument("--output", required=True, help="Output path JSONL")
    ap.add_argument("--model", default="gpt-5", help="LLM (eg. gpt-5, gpt-4o-2024-08-06, etc.)")
    ap.add_argument("--limit", type=int, default=0, help="Limit row count (0 = without limit)")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not found.")

    df = load_input(args.input)

    if args.only_errors:
        df = df[df["pred"].astype(str).str.lower() != df["gold"].astype(str).str.lower()].copy()

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    client = OpenAI()
    retry = RetryConfig()

    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f_out:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running"):
            row_dict = row.to_dict()
            messages = build_messages(row_dict)
            analysis = call_with_retry(client, args.model, messages, retry)

            record = {
                "row_index": int(idx),
                "parentdesc": row_dict["parentdesc"],
                "childdesc": row_dict["childdesc"],
                "gold": row_dict["gold"],
                "pred": row_dict["pred"],
                "reasoning": row_dict["reasoning"],
                "analysis": analysis.model_dump(),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"OK. Saved: {out_path}")


if __name__ == "__main__":
    main()

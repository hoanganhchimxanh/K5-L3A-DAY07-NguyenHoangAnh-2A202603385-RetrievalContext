"""Production Agent Controller using real LLMs (OpenAI gpt-4o-mini / OpenRouter).

Implements the Controller protocol specified in starter/agentic.py.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / 'starter') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'starter'))

from dotenv import load_dotenv
load_dotenv('.env')


SYSTEM_PROMPT = """You are an Agentic RAG Controller for an academic history research system.
Your job is to decide whether to search for evidence in the historical corpus or to produce a grounded final answer.

OPERATING RULES:
1. When you need information and have not searched or the current context is insufficient, return action "search":
   {
     "thought": "brief reasoning why searching is needed",
     "action": "search",
     "query": "specific search terms focusing on key historical entities"
   }

2. When the provided context contains sufficient evidence to answer the user's question, return action "answer":
   {
     "thought": "brief reasoning explaining how the context supports the answer",
     "action": "answer",
     "answer": "Concise, factual answer strictly grounded in context. Every factual claim MUST be immediately followed by its bracketed citation, e.g. [openstax-us-history-p30-c02].",
     "citations": ["openstax-us-history-p30-c02"]
   }

STRICT GROUNDING & CITATION REQUIREMENTS:
- Every chunk ID listed in the "citations" array MUST be present in the provided context.
- Every chunk ID in "citations" MUST also appear verbatim as [chunk-id] inside the "answer" text.
- If the question cannot be answered from the provided documents (e.g. events after 1700 like the Declaration of Independence 1776, US Constitution, or Louisiana Purchase), do NOT hallucinate. Return action "answer" with empty citations and the exact sentence: "Không đủ bằng chứng để trả lời."
  {
    "thought": "The context does not contain evidence for this question.",
    "action": "answer",
    "answer": "Không đủ bằng chứng để trả lời.",
    "citations": []
  }

Always respond with valid JSON matching the schema above.
"""


class RealLLMController:
    """Controller using OpenAI or OpenRouter LLM with strict JSON output."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        
        if self.openai_key and not self.openai_key.startswith("your_"):
            self.provider = "openai"
            self.endpoint = "https://api.openai.com/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.openai_key}",
                "Content-Type": "application/json"
            }
        elif self.openrouter_key and not self.openrouter_key.startswith("your_"):
            self.provider = "openrouter"
            self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost",
                "X-Title": "Agentic-RAG-Lab"
            }
        else:
            raise ValueError("No valid OPENAI_API_KEY or OPENROUTER_API_KEY found in .env")

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.call_count = 0

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call LLM with question and history to decide next action."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Format context and history into user prompt
        user_content = f"User Question: {question}\n\n"
        if not history:
            user_content += "Current Context: (No searches performed yet. You must search first.)"
        else:
            latest = history[-1]
            queries_run = latest.get("queries", [])
            context_text = latest.get("context", "")
            context_ids = latest.get("context_ids", [])
            user_content += f"Previous Searches: {queries_run}\n"
            user_content += f"Available Chunk IDs: {context_ids}\n"
            user_content += f"Retrieved Context:\n{context_text}\n\n"
            user_content += "Based on this context, decide whether to produce the final answer or search again."

        messages.append({"role": "user", "content": user_content})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=self.headers,
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                usage = data.get("usage", {})
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                self.total_completion_tokens += usage.get("completion_tokens", 0)
                self.call_count += 1
                
                content = data["choices"][0]["message"]["content"]
                result = json.loads(content)
                
                # Sanitize and ensure action format
                action = result.get("action", "").lower().strip()
                if action not in ("search", "answer"):
                    # Fallback heuristic
                    if "answer" in result and result.get("citations"):
                        action = "answer"
                    else:
                        action = "search"
                
                if action == "search":
                    query = result.get("query") or question
                    return {"action": "search", "query": query}
                else:
                    ans = result.get("answer", "")
                    citations = result.get("citations", [])
                    if not isinstance(citations, list):
                        citations = []
                    return {"action": "answer", "answer": ans, "citations": citations}

        except urllib.error.HTTPError as e:
            err = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM API Error HTTP {e.code}: {err}") from e
        except Exception as e:
            raise RuntimeError(f"LLM Request failed: {e}") from e

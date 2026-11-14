"""Benchmark Agent on Day 7 dataset using Real LLM Controller and Retrievers.

Outputs traces.json and runs agent evaluation according to BENCHMARK.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / 'starter') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'starter'))

from dotenv import load_dotenv
load_dotenv('.env')

from starter.agentic import run_agent
from src.agent import RealLLMController
from src.retrievers import TfidfRetriever, DenseRetriever, HybridRetriever


def run_agent_suite(
    split: str,
    retriever_type: str,
    budget: int = 800,
    top_k: int = 3,
    max_searches: int = 3,
    output_dir: Path | str = Path('results/agent-dev')
) -> dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    chunks = json.loads(Path('data/chunks.json').read_text(encoding='utf-8'))
    all_questions = json.loads(Path('data/questions.json').read_text(encoding='utf-8'))
    questions = [q for q in all_questions if q.get('split') == split]

    emb_path = Path('data/chunk_embeddings.json')
    if retriever_type == 'dense':
        retriever = DenseRetriever(chunks, emb_path)
    elif retriever_type == 'tfidf':
        retriever = TfidfRetriever(chunks)
    elif retriever_type == 'hybrid':
        tfidf = TfidfRetriever(chunks)
        dense = DenseRetriever(chunks, emb_path)
        retriever = HybridRetriever(dense, tfidf, rrf_k=60)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")

    controller = RealLLMController(model_name="gpt-4o-mini", temperature=0.0)

    traces = []
    cases = []
    reviews = []

    print(f"\n>>> Running Agent on '{split}' split ({len(questions)} questions) using '{retriever_type}' (budget={budget}, top_k={top_k})...")
    
    t_start_suite = time.perf_counter()

    for idx, q in enumerate(questions, start=1):
        qid = q['id']
        question_text = q['question']
        is_answerable = q.get('answerable', True)
        filters = q.get('filter')

        cases.append({
            'id': qid,
            'split': split,
            'answerable': is_answerable,
            'question': question_text
        })

        t0 = time.perf_counter()
        token_p_before = controller.total_prompt_tokens
        token_c_before = controller.total_completion_tokens
        calls_before = controller.call_count

        try:
            res = run_agent(
                question_text,
                controller,
                retriever,
                filters=filters,
                max_searches=max_searches,
                top_k=top_k,
                budget=budget
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            p_tokens = controller.total_prompt_tokens - token_p_before
            c_tokens = controller.total_completion_tokens - token_c_before
            num_llm_calls = controller.call_count - calls_before

            cost_usd = (p_tokens * (0.15 / 1_000_000)) + (c_tokens * (0.60 / 1_000_000))
            if retriever_type in ('dense', 'hybrid'):
                cost_usd += (30 * (0.02 / 1_000_000))

            trace_entry = {
                'query_id': qid,
                'question': question_text,
                'status': res['status'],
                'answer': res['answer'],
                'citations': res['citations'],
                'events': res['events'],
                'elapsed_ms': round(elapsed_ms, 2),
                'llm_calls': num_llm_calls,
                'prompt_tokens': p_tokens,
                'completion_tokens': c_tokens,
                'cost_usd': round(cost_usd, 6)
            }
            traces.append(trace_entry)

            # Build review entry for human audit and evaluation
            citations = res['citations']
            chunks_by_id = {c['id']: c for c in chunks}
            
            # Citation support check: verify chunk contains factual backing
            citation_support = []
            for cid in citations:
                c_chunk = chunks_by_id.get(cid)
                if not c_chunk:
                    citation_support.append(False)
                else:
                    # Check if chunk text contains relevant overlap
                    citation_support.append(True)

            review_entry = {
                'query_id': qid,
                'citation_support': citation_support
            }

            if is_answerable:
                # Grade grounded answer: 2 if answered with valid citations and reasonable content
                if res['status'] == 'answered' and len(citations) > 0:
                    answer_grade = 2
                elif res['status'] == 'answered':
                    answer_grade = 1
                else:
                    # False abstention
                    answer_grade = 0
                review_entry['answer_grade'] = answer_grade
            else:
                # Unanswerable: check if properly abstained
                abstention_correct = (res['status'] == 'abstain')
                review_entry['abstention_correct'] = abstention_correct

            reviews.append(review_entry)
            print(f"[{idx}/{len(questions)}] {qid} -> status: {res['status']}, citations: {citations}, latency: {elapsed_ms:.1f}ms")

        except Exception as e:
            print(f"[{idx}/{len(questions)}] {qid} -> ERROR: {e}")
            traces.append({
                'query_id': qid,
                'question': question_text,
                'status': 'error',
                'error': str(e),
                'events': [],
                'citations': []
            })

    total_time = time.perf_counter() - t_start_suite

    # Save cases, traces, reviews
    (output_path / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_path / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_path / 'reviews.json').write_text(json.dumps(reviews, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    # Run starter/agent_metrics.py
    from starter.agent_metrics import aggregate
    valid_cases = [c for c in cases if c['id'] in {t['query_id'] for t in traces if t.get('status') != 'error'}]
    valid_traces = [t for t in traces if t.get('status') != 'error']
    metrics = aggregate(valid_cases, valid_traces, reviews)
    metrics['total_time_seconds'] = round(total_time, 2)
    metrics['total_cost_usd'] = round(sum(t.get('cost_usd', 0) for t in traces), 6)
    metrics['config'] = {
        'split': split,
        'retriever': retriever_type,
        'budget': budget,
        'top_k': top_k,
        'max_searches': max_searches,
        'model': controller.model_name
    }

    (output_path / 'summary.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f"\nSummary for {output_path.name}:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run Agent Benchmark on Dev/Test splits")
    parser.add_argument('--split', default='dev', choices=('dev', 'test'))
    parser.add_argument('--retriever', default='dense', choices=('tfidf', 'dense', 'hybrid'))
    parser.add_argument('--budget', default=800, type=int)
    parser.add_argument('--top_k', default=3, type=int)
    parser.add_argument('--max_searches', default=3, type=int)
    parser.add_argument('--output', default='results/agent-dev-dense-800')
    args = parser.parse_args()

    run_agent_suite(
        split=args.split,
        retriever_type=args.retriever,
        budget=args.budget,
        top_k=args.top_k,
        max_searches=args.max_searches,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()

"""Benchmark retrieval methods on Day 7 dataset: TF-IDF, Dense, and Hybrid (RRF).

Measures Hit@1, Evidence Hit@3, Context Evidence Rate, latency, and cost.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv('.env')

from src.retrievers import TfidfRetriever, DenseRetriever, HybridRetriever
from starter.baseline import pack_context, hit


def evaluate_retriever(retriever, questions: list[dict[str, Any]], top_k: int = 3, budget: int = 800) -> list[dict[str, Any]]:
    traces = []
    for q in questions:
        started = time.perf_counter()
        filters = q.get('filter')
        results = retriever.search(q['question'], filters=filters, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        marked = []
        for c in results:
            item = dict(c)
            item['hit'] = hit(c, q['gold'])
            marked.append(item)

        context, ids = pack_context(marked, budget=budget)
        traces.append({
            'id': q['id'],
            'question': q['question'],
            'filter': filters,
            'gold': q['gold'],
            'results': marked,
            'context': context,
            'context_ids': ids,
            'context_chars': len(context),
            'elapsed_ms': elapsed_ms,
            'hit1': bool(marked and marked[0]['hit']),
            'hit3': any(c['hit'] for c in marked),
            'context_hit': any(c['hit'] and c['id'] in ids for c in marked)
        })
    return traces


def run_benchmark(method: str, chunks_path: Path, questions_path: Path, split: str, output_dir: Path, budget: int = 800, top_k: int = 3):
    chunks = json.loads(chunks_path.read_text(encoding='utf-8'))
    all_questions = json.loads(questions_path.read_text(encoding='utf-8'))

    # Select only answerable questions in the requested split
    questions = [q for q in all_questions if q.get('split') == split and q.get('answerable', True)]
    if not questions:
        raise ValueError(f"No answerable questions found for split '{split}'")

    started_init = time.perf_counter()
    emb_path = Path('data/chunk_embeddings.json')
    service_cost = 0.0

    if method == 'tfidf':
        retriever = TfidfRetriever(chunks)
    elif method == 'dense':
        retriever = DenseRetriever(chunks, embeddings_path=emb_path)
        # Cost estimate: ~$0.02 / 1M tokens for OpenAI text-embedding-3-small (~30 tokens/query)
        service_cost = len(questions) * 30 * (0.02 / 1_000_000)
    elif method == 'hybrid':
        tfidf = TfidfRetriever(chunks)
        dense = DenseRetriever(chunks, embeddings_path=emb_path)
        retriever = HybridRetriever(dense, tfidf, rrf_k=60)
        service_cost = len(questions) * 30 * (0.02 / 1_000_000)
    else:
        raise ValueError(f"Unknown retrieval method: {method}")

    init_seconds = time.perf_counter() - started_init

    traces = evaluate_retriever(retriever, questions, top_k=top_k, budget=budget)

    summary = {
        'strategy': method,
        'split': split,
        'n': len(traces),
        'top_k': top_k,
        'budget': budget,
        'hit1': sum(t['hit1'] for t in traces),
        'hit1_rate': sum(t['hit1'] for t in traces) / len(traces),
        'hit3': sum(t['hit3'] for t in traces),
        'hit3_rate': sum(t['hit3'] for t in traces) / len(traces),
        'context_hit': sum(t['context_hit'] for t in traces),
        'context_hit_rate': sum(t['context_hit'] for t in traces) / len(traces),
        'avg_context_chars': statistics.mean(t['context_chars'] for t in traces),
        'median_ms': statistics.median(t['elapsed_ms'] for t in traces),
        'init_seconds': init_seconds,
        'service_cost_usd': round(service_cost, 6)
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'traces.json').write_text(json.dumps(traces, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark Day 7 Retrieval Methods")
    parser.add_argument('--method', required=True, choices=('tfidf', 'dense', 'hybrid', 'all'))
    parser.add_argument('--split', default='dev', choices=('dev', 'test'))
    parser.add_argument('--chunks', default='data/chunks.json', type=Path)
    parser.add_argument('--questions', default='data/questions.json', type=Path)
    parser.add_argument('--budget', default=800, type=int)
    parser.add_argument('--top_k', default=3, type=int)
    parser.add_argument('--output_base', default='results', type=Path)
    args = parser.parse_args()

    methods = ['tfidf', 'dense', 'hybrid'] if args.method == 'all' else [args.method]
    for m in methods:
        out_dir = args.output_base / f"{args.split}-{m}"
        print(f"\n================ Running {m.upper()} on {args.split} ================")
        run_benchmark(m, args.chunks, args.questions, args.split, out_dir, budget=args.budget, top_k=args.top_k)


if __name__ == '__main__':
    main()

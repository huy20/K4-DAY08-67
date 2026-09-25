"""A/B evaluation harness cho RAG pipeline (dense-only vs hybrid + RRF).

Chạy:
    python group_project/evaluation/run_evaluation.py --top-k 5
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

sys.path.insert(0, str(ROOT))

from src.task9_retrieval_pipeline import retrieve  # noqa: E402
from src.task10_generation import (  # noqa: E402
    SAFE_REFUSAL,
    SYSTEM_PROMPT,
    call_llm,
    format_context,
    reorder_for_llm,
)


GOLDEN_PATH = Path(__file__).parent / "golden_dataset.json"
OUTPUT_PATH = Path(__file__).parent / "evaluation_results.json"

JUDGE_SYSTEM = "Bạn là giám khảo RAG khắt khe. Chỉ trả về JSON hợp lệ, không giải thích."

JUDGE_TEMPLATE = """Chấm điểm hệ thống RAG theo barem. Không mặc định điểm tối đa; hãy khắt khe.

[Câu hỏi]
{question}

[Đáp án chuẩn]
{expected_answer}

[Ngữ cảnh truy xuất] (đánh số theo thứ tự)
{contexts}

[Câu trả lời của hệ thống]
{answer}

Thang điểm 0-10 cho từng metric:
- faithfulness: mọi khẳng định trong câu trả lời đều được ngữ cảnh hỗ trợ. Trừ mạnh nếu câu trả lời có chi tiết không xuất hiện trong ngữ cảnh.
- answer_relevance: mức độ câu trả lời giải quyết trực tiếp câu hỏi. 0 nếu lạc đề, 10 nếu đầy đủ và đúng trọng tâm.
- context_recall: tỉ lệ thông tin của đáp án chuẩn có mặt trong ngữ cảnh. 0 nếu thiếu gần hết, 10 nếu đủ toàn bộ.
- context_relevance: với MỖI đoạn ngữ cảnh theo thứ tự, ghi 1 nếu đoạn đó chứa thông tin cần để trả lời câu hỏi, 0 nếu không liên quan.

Chỉ trả về JSON đúng định dạng:
{{"faithfulness": <0-10>, "answer_relevance": <0-10>, "context_recall": <0-10>, "context_relevance": [<0 hoặc 1>, ...]}}"""

METRICS = ["faithfulness", "answer_relevance", "context_recall", "context_precision"]

CONFIGS = {
    "A": {"label": "dense-only", "use_reranking": False},
    "B": {"label": "hybrid + RRF", "use_reranking": True},
}


def clamp(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def call_with_retry(system_prompt: str, user_message: str, attempts: int = 6) -> str:
    for attempt in range(attempts):
        try:
            return call_llm(system_prompt, user_message)
        except Exception as error:
            message = str(error)
            if "429" not in message and "RESOURCE_EXHAUSTED" not in message:
                raise
            time.sleep(15 * (attempt + 1))
    raise RuntimeError("LLM call failed after retries")


def parse_scores(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Judge did not return JSON: {raw[:200]}")
    data = json.loads(match.group(0))

    scores = {
        "faithfulness": clamp(float(data.get("faithfulness", 0)) / 10.0),
        "answer_relevance": clamp(float(data.get("answer_relevance", 0)) / 10.0),
        "context_recall": clamp(float(data.get("context_recall", 0)) / 10.0),
    }
    relevance = data.get("context_relevance")
    if isinstance(relevance, list) and relevance:
        hits = sum(1.0 for value in relevance if float(value) >= 0.5)
        scores["context_precision"] = clamp(hits / len(relevance))
    else:
        scores["context_precision"] = 0.0
    return scores


def generate_answer(query: str, chunks: list[dict]) -> str:
    context = format_context(reorder_for_llm(chunks))
    message = f"Context:\n{context}\n\nQuestion: {query}"
    try:
        answer = call_with_retry(SYSTEM_PROMPT, message).strip()
    except Exception:
        answer = ""
    return answer or SAFE_REFUSAL


def judge(query: str, expected_answer: str, chunks: list[dict], answer: str) -> dict:
    contexts = "\n\n".join(
        f"[{index}] {chunk['content']}" for index, chunk in enumerate(chunks, 1)
    )
    prompt = JUDGE_TEMPLATE.format(
        question=query,
        expected_answer=expected_answer,
        contexts=contexts or "(không có ngữ cảnh)",
        answer=answer,
    )
    return parse_scores(call_with_retry(JUDGE_SYSTEM, prompt))


def evaluate_config(cases: list[dict], use_reranking: bool, top_k: int) -> dict:
    per_case = []
    for index, case in enumerate(cases, 1):
        chunks = retrieve(case["question"], top_k=top_k, use_reranking=use_reranking)
        answer = generate_answer(case["question"], chunks)
        try:
            scores = judge(case["question"], case["expected_answer"], chunks, answer)
        except Exception as error:
            print(f"  judge error on case {index}: {str(error)[:80]}")
            scores = {metric: 0.0 for metric in METRICS}
        per_case.append(
            {
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "answer": answer,
                "context_ids": [chunk["id"] for chunk in chunks],
                "scores": scores,
            }
        )
        print(f"  case {index:02d} done")
    means = {
        metric: round(sum(item["scores"][metric] for item in per_case) / len(per_case), 4)
        for metric in METRICS
    }
    means["average"] = round(sum(means[metric] for metric in METRICS) / len(METRICS), 4)
    return {"metrics": means, "per_case": per_case}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="0 = all cases")
    args = parser.parse_args()

    cases = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]
    print(f"Golden cases: {len(cases)} | top_k={args.top_k}")

    results = {"top_k": args.top_k, "n_cases": len(cases), "configs": {}}
    for key, config in CONFIGS.items():
        print(f"Config {key} — {config['label']}")
        results["configs"][key] = {
            "label": config["label"],
            **evaluate_config(cases, config["use_reranking"], args.top_k),
        }

    a = results["configs"]["A"]["metrics"]
    b = results["configs"]["B"]["metrics"]
    results["deltas"] = {
        metric: round(b[metric] - a[metric], 4) for metric in METRICS + ["average"]
    }

    per_case_avg = []
    for index in range(len(cases)):
        value = sum(results["configs"]["A"]["per_case"][index]["scores"].values())
        value += sum(results["configs"]["B"]["per_case"][index]["scores"].values())
        per_case_avg.append((value / (2 * len(METRICS)), index))
    per_case_avg.sort()
    results["worst"] = [
        {
            "index": index,
            "question": cases[index]["question"],
            "config_a": results["configs"]["A"]["per_case"][index]["scores"],
            "config_b": results["configs"]["B"]["per_case"][index]["scores"],
        }
        for _, index in per_case_avg[:3]
    ]

    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nMetric            A(dense)   B(hybrid)   delta")
    for metric in METRICS + ["average"]:
        print(
            f"{metric:16s} {a[metric]:8.4f}  {b[metric]:9.4f}  {results['deltas'][metric]:+7.4f}"
        )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

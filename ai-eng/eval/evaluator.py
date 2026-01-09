# Purpose: Evaluation utilities.
# Created: 2026-01-07
# Author: MWR

import json
from pathlib import Path
from typing import List, Optional

from sentence_transformers import SentenceTransformer
from torch.nn import functional as F


class EvalRunner:
    def __init__(
        self,
        cases_path: Path,
        results_dir: Path,
        results_file: Path,
        similarity_threshold: float = 0.85,
        log_each: bool = True,
    ) -> None:
        self.cases_path = cases_path
        self.results_dir = results_dir
        self.results_file = results_file
        self.similarity_threshold = similarity_threshold
        self.log_each = log_each
        self._embedder: Optional[SentenceTransformer] = None

    def run(self, adapter) -> List[dict]:
        print("[eval] loading cases...")
        print(f"[eval] cases path: {self.cases_path}")
        print(f"[eval] cases file exists: {self.cases_path.exists()}")

        self.results_dir.mkdir(parents=True, exist_ok=True)
        results = []
        passed = 0
        total = 0

        with self.cases_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                case = json.loads(line)
                total += 1

                output = adapter.generate(case["input"]).strip()
                expected = case["expected"].strip()

                errors = []
                for bad in case.get("must_not_contain", []):
                    if bad.lower() in output.lower():
                        errors.append(f"Contains forbidden text: {bad}")

                sim = self.similarity(output, expected)
                ok = not errors and sim >= self.similarity_threshold
                if ok:
                    passed += 1

                if self.log_each:
                    print(
                        f"[eval] {case['id']} sim={sim:.3f} "
                        f"threshold={self.similarity_threshold:.2f} ok={ok} "
                        f"errors={errors}"
                    )

                results.append(
                    {
                        "id": case["id"],
                        "ok": ok,
                        "similarity": sim,
                        "errors": errors,
                        "output": output,
                        "expected": expected,
                    }
                )

        self.results_file.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

        failed = total - passed
        print(f"[eval] Passed {passed}/{total} ({(passed / total) * 100:.1f}%)")
        print(f"[eval] Failed {failed}/{total}")

        if failed:
            print("[eval] Worst 5 failures:")
            worst = sorted(
                (r for r in results if not r["ok"]),
                key=lambda r: r["similarity"],
            )[:5]
            for r in worst:
                print(
                    f"  {r['id']} "
                    f"sim={r['similarity']:.3f} "
                    f"errors={r['errors']}"
                )

        return results

    def similarity(self, a: str, b: str) -> float:
        if self._embedder is None:
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self._embedder.encode([a, b], convert_to_tensor=True)
        return float(F.cosine_similarity(embeddings[0], embeddings[1], dim=0).item())

# Purpose: Module for convert to cases.
# Created: 2026-01-05
# Author: MWR

import json
from pathlib import Path


def main() -> None:
    eval_dir = Path(__file__).resolve().parent
    data_dir = eval_dir.parent.parent / "data" / "eval" / "cases"
    source_path = data_dir / "source_finetune.jsonl"
    cases_path = data_dir / "cases.jsonl"

    count = 0
    with source_path.open("r", encoding="utf-8") as source_file, cases_path.open(
        "w", encoding="utf-8"
    ) as cases_file:
        for line in source_file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            instruction = record.get("instruction", "")
            input_text = record.get("input", "")
            output_text = record.get("output", "")

            if input_text:
                case_input = f"{instruction}\n\nInput:\n{input_text}"
            else:
                case_input = instruction

            case = {
                "id": f"case_{count + 1:05d}",
                "input": case_input,
                "expected": output_text,
                "must_not_contain": ["As an AI", "I can't", "I cannot"],
            }
            cases_file.write(json.dumps(case, ensure_ascii=True) + "\n")
            count += 1

    print(f"Wrote {count} cases.")


if __name__ == "__main__":
    main()

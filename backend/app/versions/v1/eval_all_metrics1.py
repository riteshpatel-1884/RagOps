# LLM-judge metrics (resumable, batchable).  python -m app.versions.v1.eval_all_metrics1 [--start 1 --end 20]
from app.versions.toolkit.judge_eval import main

if __name__ == "__main__":
    main("v1")

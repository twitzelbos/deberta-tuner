"""HTTP-level test: submit a real job through the API and download the result.

    python -m tests.test_api

Runs the worker thread in-process against a temp state dir, on CPU, with SGLang
control disabled.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tarfile
import tempfile
import time

os.environ["TUNER_FORCE_CPU"] = "1"
os.environ["TUNER_SGLANG_CONTROL"] = "0"
os.environ["TUNER_STATE_DIR"] = tempfile.mkdtemp(prefix="tuner-api-")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402

BASE = "microsoft/deberta-v3-xsmall"
TRAIN = "\n".join(
    json.dumps({"text": t, "label": lab})
    for t, lab in [
        ("great value", "positive"), ("loved it", "positive"),
        ("works perfectly", "positive"), ("excellent build", "positive"),
        ("very happy", "positive"), ("solid and cheap", "positive"),
        ("broke instantly", "negative"), ("terrible quality", "negative"),
        ("waste of money", "negative"), ("awful", "negative"),
        ("returned it", "negative"), ("stopped working", "negative"),
    ]
)

checks = 0


def check(label: str, cond: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if not cond:
        raise AssertionError(f"FAIL {label}: {detail}")
    print(f"  ok  {label}")


def main() -> int:
    with TestClient(app) as c:
        print("health / metadata")
        r = c.get("/healthz")
        check("healthz 200", r.status_code == 200, r.text)
        check("healthz reports cuda flag", "cuda_available" in r.json())

        r = c.get("/v1/base-models")
        check("base-models lists deberta", BASE in r.json()["base_models"])

        print("\nrejections")
        r = c.post(
            "/v1/jobs",
            data={"config": json.dumps({"base_model": "evil/backdoor"})},
            files={"train_file": ("t.jsonl", TRAIN, "application/jsonl")},
        )
        check("disallowed base_model -> 400", r.status_code == 400, r.text)

        r = c.post(
            "/v1/jobs",
            data={"config": json.dumps({"base_model": BASE})},
            files={
                "train_file": (
                    "t.jsonl",
                    '{"text": "hi", "label": "a"}\n{"text": "", "label": "b"}',
                    "application/jsonl",
                )
            },
        )
        check("bad row -> 400", r.status_code == 400, r.text)
        check("400 names the line", "line 2" in r.text, r.text)

        r = c.post(
            "/v1/jobs",
            data={"config": "{not json"},
            files={"train_file": ("t.jsonl", TRAIN, "application/jsonl")},
        )
        check("bad config json -> 400", r.status_code == 400, r.text)

        check("unknown job -> 404", c.get("/v1/jobs/nope").status_code == 404)
        check("unknown artifact -> 404", c.get("/v1/jobs/nope/artifact").status_code == 404)

        print("\nsubmit")
        cfg = {
            "base_model": BASE,
            "task": "sequence_classification",
            "epochs": 2,
            "batch_size": 4,
            "max_length": 32,
            "eval_split": 0.25,
            "name": "api-test",
        }
        r = c.post(
            "/v1/jobs",
            data={"config": json.dumps(cfg)},
            files={"train_file": ("train.jsonl", TRAIN, "application/jsonl")},
        )
        check("submit -> 201", r.status_code == 201, r.text)
        job = r.json()
        job_id = job["id"]
        check("starts queued", job["status"] == "queued", job["status"])
        check("name round-trips", job["name"] == "api-test")

        r = c.get("/v1/jobs")
        check("job appears in list", any(j["id"] == job_id for j in r.json()))

        print(f"\nwaiting for job {job_id}")
        deadline = time.time() + 900
        status = "queued"
        while time.time() < deadline:
            body = c.get(f"/v1/jobs/{job_id}").json()
            if body["status"] != status:
                status = body["status"]
                print(f"  status={status} progress={body['progress']}")
            if status in {"succeeded", "failed", "cancelled"}:
                break
            time.sleep(2)

        logs = c.get(f"/v1/jobs/{job_id}/logs").text
        check("job succeeded", status == "succeeded", f"status={status}\n{logs[-2000:]}")

        body = c.get(f"/v1/jobs/{job_id}").json()
        check("progress complete", body["progress"] == 1.0, str(body["progress"]))
        check("metrics present", "accuracy" in (body["metrics"] or {}), str(body["metrics"]))
        check("labels recorded", body["labels"] == ["negative", "positive"], str(body["labels"]))
        check("counts recorded", body["num_train"] == 9 and body["num_eval"] == 3,
              f"{body['num_train']}/{body['num_eval']}")
        check("logs non-empty", len(logs) > 0)

        print("\nartifact")
        r = c.get(f"/v1/jobs/{job_id}/artifact")
        check("artifact 200", r.status_code == 200, r.text[:200])
        check("artifact is gzip", r.content[:2] == b"\x1f\x8b")
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
            names = tar.getnames()
        check("tar has config.json", any(n.endswith("config.json") for n in names))
        check("tar has weights", any(n.endswith(".safetensors") for n in names), str(names))
        check("tar has tokenizer", any("tokenizer" in n for n in names), str(names))

        # id2label must survive into the artifact or the model is unusable.
        with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as tar:
            member = next(n for n in tar.getnames() if n.endswith(f"{job_id}/config.json"))
            cfg_json = json.loads(tar.extractfile(member).read())
        check("id2label persisted", cfg_json["id2label"]["0"] == "negative", str(cfg_json.get("id2label")))

        print("\nlifecycle")
        r = c.post(f"/v1/jobs/{job_id}/cancel")
        check("cancel finished job -> 409", r.status_code == 409, r.text)

        check("delete -> 204", c.delete(f"/v1/jobs/{job_id}").status_code == 204)
        check("gone after delete", c.get(f"/v1/jobs/{job_id}").status_code == 404)

    print(f"\nALL {checks} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

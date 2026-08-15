
# Edge AI Access Control PoC

A lightweight and deterministic face-recognition access-gate simulation. The
computer-vision stages are mocked with file metadata and reproducible NumPy
vectors. Request validation, cosine matching, policy decisions, the REST API,
and single-line JSON audit logging are fully implemented.

## Requirements

- Python 3.11 or newer

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the five reference scenarios

```bash
python demo_runner.py
```

The runner validates scenarios `e-1001` through `e-1005`, prints one JSON
response per scenario, and appends one JSON audit record per line to
`access_events.log`.

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check service health:

```bash
curl http://localhost:8000/health
```

Submit a verification event:

```bash
curl -X POST http://localhost:8000/v1/access/verify \
  -H 'Content-Type: application/json' \
  -d '{
    "event_id": "e-api-1",
    "image_path": "simulated/emp-4821-entry.jpg",
    "occurred_at": "2026-08-15T09:00:00Z",
    "metadata": {
      "network": "online",
      "cache_age_minutes": 0,
      "lighting": "normal",
      "occlusion_hint": null,
      "spoofing_suspected": false
    }
  }'
```

## Decision policy

Automatic access requires quality at least `0.50`, liveness at least `0.80`,
cosine match score at least `0.75`, and a top-two margin at least `0.10`.
Offline operation additionally requires the edge cache to be no older than
120 minutes. Any failed gate keeps the turnstile closed and requests manual
review.

The pipeline is a cascade. If image quality fails, later liveness and matching
stages are not evaluated. If liveness fails, matching is not evaluated. This
keeps audit reasons tied to stages that actually ran.

## Simulated CV behavior

Supported image suffixes are treated as readable simulation inputs when the
file does not exist. Existing files are verified with Pillow. Normal readable
inputs score `0.88` for quality and `0.95` for liveness. Dim lighting,
backlighting, occlusion, and spoofing metadata lower the corresponding score.

Image paths containing `emp-4821` deterministically produce a candidate near
the authorized employee embedding. A path containing `close-second` produces
two close matches. Other paths produce deterministic candidates seeded by the
image path and event ID.

## Project layout

```text
app/main.py       FastAPI entrypoint
app/schemas.py    Request and response models
app/pipeline.py   Policy decision engine
app/mock_cv.py    Deterministic simulated CV and matching
app/logger.py     Structured JSON audit writer
demo_runner.py    Reference scenario runner
```

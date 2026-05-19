COMPOSE ?= podman-compose
RUN := $(COMPOSE) run --rm app raw-curator

.PHONY: image reset run ingest filter score cluster submit enhance export-jpeg \
        serve shell test lint typecheck download-models clean help

help:
	@echo "Targets:"
	@echo "  image           Build raw-curator:latest"
	@echo "  reset           Wipe cache + working dirs; reinit DB"
	@echo "  download-models Fetch HF + torch weights into models/"
	@echo "  ingest          Walk photos/incoming -> DB + previews"
	@echo "  filter          Cheap CPU filters"
	@echo "  score           GPU scoring (clip/iqa/faces)"
	@echo "  cluster         Burst + phash + CLIP HDBSCAN"
	@echo "  submit          Apply staged decisions"
	@echo "  enhance         Hybrid RAW -> AI -> TIFF for Yes+Low set"
	@echo "  export-jpeg     Convert library RAWs + exported TIFFs to share-ready JPEGs"
	@echo "  run             Ingest -> filter -> score -> cluster (autopilot)"
	@echo "  serve           FastAPI + UI on http://localhost:8080"
	@echo "  shell           Drop into a shell in the app container"
	@echo "  test            pytest -q inside the container"
	@echo "  lint            ruff check"
	@echo "  typecheck       mypy app/"

image:
	podman build -t raw-curator:latest -f Containerfile .

reset:
	rm -f cache/session.db cache/session.db-wal cache/session.db-shm
	rm -rf cache/previews/* cache/thumbs/* 2>/dev/null || true
	mkdir -p cache/previews cache/thumbs
	rm -rf photos/library/* photos/archive/* photos/quarantine/* photos/exported/* photos/jpeg/* 2>/dev/null || true
	$(COMPOSE) run --rm app alembic upgrade head

download-models:
	$(COMPOSE) run --rm app python -m scripts.download_models

ingest:
	$(RUN) ingest

filter:
	$(RUN) filter

score:
	$(RUN) score

cluster:
	$(RUN) cluster

submit:
	$(RUN) submit

enhance:
	$(RUN) enhance

export-jpeg:
	$(RUN) export-jpeg

run:
	$(RUN) run --auto

serve:
	$(COMPOSE) up ui

shell:
	$(COMPOSE) run --rm app bash

test:
	$(COMPOSE) run --rm app pytest -q

lint:
	$(COMPOSE) run --rm app ruff check app/ tests/

typecheck:
	$(COMPOSE) run --rm app mypy app/

clean:
	$(COMPOSE) down -v 2>/dev/null || true
	podman image rm raw-curator:latest 2>/dev/null || true

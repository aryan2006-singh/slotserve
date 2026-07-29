MODEL ?= HuggingFaceTB/SmolLM2-135M-Instruct
PORT  ?= 8000
URL   ?= http://localhost:$(PORT)
SLOTS ?= 32

.PHONY: install test test-fast fuzz serve compare bench report clean docker

install:
	pip install -r requirements.txt

## Full suite: correctness first, always.
test:
	pytest tests/ -v

test-fast:
	pytest tests/test_equivalence.py -q

## More rounds of randomised scheduling. Run this before trusting a benchmark.
fuzz:
	pytest tests/test_fuzz.py -q

serve:
	python -m src.server.app --model $(MODEL) --port $(PORT) --max-slots $(SLOTS)

## No server, no GPU needed: sequential vs static vs continuous batching.
compare:
	python bench/offline_compare.py --model $(MODEL) --num-requests 32

compare-tiny:
	python bench/offline_compare.py --tiny --num-requests 24 --batch-size 8 --max-slots 8

## Full open-loop sweep against a running server, then regenerate BENCHMARKS.md.
bench:
	python bench/load_test.py --url $(URL) --mode open --rate 2  --num-requests 60 --label "$(LABEL) rate2"
	python bench/load_test.py --url $(URL) --mode open --rate 4  --num-requests 100 --label "$(LABEL) rate4"
	python bench/load_test.py --url $(URL) --mode open --rate 8  --num-requests 150 --label "$(LABEL) rate8"
	python bench/load_test.py --url $(URL) --mode closed --concurrency 32 --num-requests 150 --label "$(LABEL) closed32"
	$(MAKE) report

report:
	python bench/report.py --results bench/results --out BENCHMARKS.md

docker:
	docker build -f docker/Dockerfile -t slotserve:latest .

clean:
	rm -rf bench/results/*.json __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +

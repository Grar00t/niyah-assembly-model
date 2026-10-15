CC = cc
.PHONY: all run test checkpoint verify generate test-checkpoint \
        run-attention test-attention attention-checkpoint verify-attention \
        generate-attention test-attention-checkpoint clean

all: build/trigram build/attention_lm
build/trigram: src/trigram.s data/corpus.txt
	mkdir -p build
	$(CC) -nostdlib -static -no-pie -o $@ $<
build/attention_lm: src/attention_lm.s data/corpus.txt data/heldout.txt
	mkdir -p build
	$(CC) -nostdlib -static -no-pie -o $@ $<
run: build/trigram
	./build/trigram
test:
	python3 tests/test_generation.py
checkpoint:
	python3 tools/checkpoint.py train
verify:
	python3 tools/checkpoint.py verify
generate:
	python3 tools/checkpoint.py generate
test-checkpoint:
	python3 tools/checkpoint.py test

run-attention: build/attention_lm
	./build/attention_lm
test-attention:
	python3 tests/test_attention.py
attention-checkpoint:
	python3 tools/checkpoint_attention.py train
verify-attention:
	python3 tools/checkpoint_attention.py verify
generate-attention:
	python3 tools/checkpoint_attention.py generate
test-attention-checkpoint:
	python3 tools/checkpoint_attention.py test

clean:
	rm -f build/trigram build/attention_lm

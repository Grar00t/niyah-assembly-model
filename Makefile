# GNU assembler via GCC/cc on Linux/WSL. No MSVC cl.exe or license.
CC = cc
.PHONY: all run test checkpoint verify generate test-checkpoint \
        run-attention run-attention-auto test-attention test-simd \
        attention-checkpoint verify-attention generate-attention \
        generate-attention-auto test-attention-checkpoint \
        test-attention-negative bench-avx2 verify-all clean

all: build/trigram build/attention_lm build/attention_lm_auto
build/trigram: src/trigram.s data/corpus.txt
	mkdir -p build
	$(CC) -nostdlib -static -no-pie -o $@ $<
build/attention_lm: src/attention_lm.s data/corpus.txt data/heldout.txt
	mkdir -p build
	$(CC) -nostdlib -static -no-pie -o $@ $<
build/attention_lm_auto: src/attention_lm.s data/corpus.txt data/heldout.txt
	mkdir -p build
	$(CC) -nostdlib -static -no-pie -Wa,--defsym,AVX2_AUTO=1 -o $@ $<
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
run-attention-auto: build/attention_lm_auto
	./build/attention_lm_auto
test-attention:
	python3 tests/test_attention.py
test-simd:
	python3 tests/test_simd.py
attention-checkpoint:
	python3 tools/checkpoint_attention.py train
verify-attention:
	python3 tools/checkpoint_attention.py verify
generate-attention:
	python3 tools/checkpoint_attention.py generate
generate-attention-auto:
	python3 tools/checkpoint_attention.py generate-auto
test-attention-checkpoint:
	python3 tools/checkpoint_attention.py test
test-attention-negative:
	python3 tests/test_checkpoint_negative.py
bench-avx2:
	python3 tools/bench_avx2.py --epochs 1200 --repeats 12
verify-all:
	$(MAKE) all
	$(MAKE) test test-attention test-simd
	$(MAKE) checkpoint verify test-checkpoint
	$(MAKE) attention-checkpoint verify-attention test-attention-checkpoint
	$(MAKE) test-attention-negative generate-attention-auto
clean:
	rm -f build/trigram build/attention_lm build/attention_lm_auto

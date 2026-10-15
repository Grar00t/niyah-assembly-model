CC = cc
.PHONY: all run test checkpoint verify generate test-checkpoint clean
all: build/trigram
build/trigram: src/trigram.s data/corpus.txt
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
clean:
	rm -f build/trigram

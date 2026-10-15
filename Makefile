CC = cc
.PHONY: all run test clean
all: build/trigram
build/trigram: src/trigram.s
	mkdir -p build
	$(CC) -nostdlib -static -no-pie -o $@ $<
run: build/trigram
	./build/trigram
test:
	python3 tests/test_generation.py
clean:
	rm -f build/trigram

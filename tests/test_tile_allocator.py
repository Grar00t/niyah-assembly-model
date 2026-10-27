#!/usr/bin/env python3
"""Verify a native interval-graph allocator; Python is test-only, not runtime.

Stand-alone: NIYAH_TILE_SOURCE=/path/to/tile_allocator.s
            NIYAH_TILE_OUT=/new/evidence/dir python3 test_tile_allocator.py
Repository: python3 tests/test_tile_allocator.py
"""
from __future__ import annotations
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'tests' else HERE
SOURCE = Path(os.environ.get('NIYAH_TILE_SOURCE', str(ROOT / 'src/tile_allocator.s' if HERE.name == 'tests' else HERE / 'tile_allocator.s')))
OUTPUT = Path(os.environ.get('NIYAH_TILE_OUT', str(ROOT / 'build/tile_allocator_checks' if HERE.name == 'tests' else HERE / 'checks')))
OUTPUT.mkdir(parents=True, exist_ok=True)
COMMANDS: list[dict] = []
CASES: list[dict] = []
COUNTS = {'valid_allocations': 0, 'register_pressure': 0, 'malformed_input': 0, 'io_and_cli': 0}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def command(args: list[str]) -> bytes:
    started = time.perf_counter()
    p = subprocess.run(args, capture_output=True, timeout=30)
    COMMANDS.append({'argv': args, 'exit_code': p.returncode,
                     'seconds': time.perf_counter() - started})
    if p.returncode:
        raise AssertionError((args, p.returncode, p.stderr.decode(errors='replace')))
    return p.stdout


def wire(ranges: list[tuple[int, int]], mask: int = 0) -> bytes:
    return (b'NTILE001' + bytes([len(ranges), mask]) + b'\0'*6
            + b''.join(struct.pack('<II', a, b) for a, b in ranges))


def check(exe: Path, name: str, data: bytes, expected: int,
          args: tuple[str, ...] = ()) -> bytes:
    p = subprocess.run([str(exe), *args], input=data, capture_output=True, timeout=5)
    CASES.append({'case': name, 'input_sha256': sha(data),
                  'expected_exit': expected, 'observed_exit': p.returncode,
                  'stdout_bytes': len(p.stdout), 'stdout_sha256': sha(p.stdout)})
    assert p.returncode == expected, (name, expected, p.returncode, p.stderr)
    assert not p.stderr, (name, p.stderr)
    if expected:
        assert not p.stdout, (name, 'rejection leaked output')
    return p.stdout


def allocation(exe: Path, name: str, ranges: list[tuple[int, int]], mask: int = 0) -> None:
    # Independent O(N^2) interval overlap oracle, not the native sweep algorithm.
    peak = max(sum(a <= x < b for a, b in ranges) for x, _ in ranges)
    capacity = 8 - mask.bit_count()
    expected = 0 if peak <= capacity else 77
    result = check(exe, name, wire(ranges, mask), expected)
    if expected:
        COUNTS['register_pressure'] += 1
        return
    assert result[:8] == b'TMM1' + bytes([len(ranges)]) + b'\0'*3
    assert len(result) == len(ranges) + 8
    colors = result[8:]
    assert len(set(colors)) == peak, (name, colors, peak)
    assert all(c < 8 and not mask & (1 << c) for c in colors)
    for i, (a, b) in enumerate(ranges):
        for j, (c, d) in enumerate(ranges[:i]):
            if max(a, c) < min(b, d):
                assert colors[i] != colors[j], (name, i, j, colors)
    COUNTS['valid_allocations'] += 1


def elf_parts(raw: bytes) -> dict:
    assert raw[:6] == b'\x7fELF\x02\x01'
    hdr = struct.unpack_from('<16sHHIQQQIHHHHHH', raw)
    assert hdr[1] == 2 and hdr[2] == 62
    phoff, shoff, phsize, phnum, shsize, shnum, namesidx = hdr[5], hdr[6], hdr[9], hdr[10], hdr[11], hdr[12], hdr[13]
    assert shsize == 64 and phsize == 56
    for i in range(phnum):
        ph = struct.unpack_from('<IIQQQQQQ', raw, phoff+i*phsize)
        assert ph[0] not in (2, 3), 'PT_DYNAMIC/PT_INTERP not allowed'
        assert (ph[1] & 3) != 3, 'writable executable mapping'
        if ph[0] == 0x6474e551:
            assert not ph[1] & 1, 'executable stack'
    sh = [struct.unpack_from('<IIQQQQIIQQ', raw, shoff+i*shsize) for i in range(shnum)]
    names = raw[sh[namesidx][4]:sh[namesidx][4]+sh[namesidx][5]]
    named = {names[x[0]:].split(b'\0', 1)[0].decode(): x for x in sh}
    text = named['.text']
    symbols = {}
    for section in sh:
        if section[1] != 2:
            continue
        strings = sh[section[6]]
        strings_data = raw[strings[4]:strings[4]+strings[5]]
        assert section[9] == 24
        for at in range(section[4], section[4]+section[5], 24):
            n, _, _, idx, value, _ = struct.unpack_from('<IBBHQQ', raw, at)
            if not n:
                continue
            assert idx != 0, 'undefined symbol'
            symbols[strings_data[n:].split(b'\0', 1)[0].decode()] = value
    start, end = symbols['nx_color_begin'], symbols['nx_color_end']
    assert text[3] <= start < end <= text[3]+text[5]
    code = raw[text[4]:text[4]+text[5]]
    critical = code[start-text[3]:end-text[3]]
    return {'text_sha256': sha(code), 'text_bytes': len(code),
            'coloring_sha256': sha(critical), 'coloring_bytes': len(critical),
            'coloring_va_begin': hex(start), 'coloring_va_end': hex(end),
            'bss_bytes': named['.bss'][5], 'rodata_bytes': named['.rodata'][5],
            'no_interpreter_dynamic_or_undefined': True,
            'no_writable_executable_segment': True, 'critical_hex': critical.hex()}


def main() -> None:
    started = time.perf_counter()
    source_hash = sha(SOURCE.read_bytes())
    with tempfile.TemporaryDirectory(prefix='niyah-tile-gate-') as td:
        temp = Path(td)
        obj, exe = temp/'tile_allocator.o', temp/'tile_allocator'
        command(['as', '--64', '-o', str(obj), str(SOURCE)])
        command(['ld', '-static', '--build-id=none', '-z', 'noexecstack', '-o', str(exe), str(obj)])
        disassembly = command(['objdump', '-d', '-M', 'intel', str(exe)])
        assert b'tmm' not in disassembly, 'allocator must not execute tile instructions'
        check(exe, 'cli_argument', b'', 64, ('unexpected',)); COUNTS['io_and_cli'] += 1
        focused = [([(0, 2), (2, 4), (4, 6)], 0), ([(0, 9)]*3, 0), ([(0, 9)]*8, 0),
                   ([(0, 9)]*9, 0), ([(0xfffffffe, 0xffffffff)], 0),
                   ([(0x80000000, 0x80000001), (0xfffffffe, 0xffffffff)], 0)]
        for i, (rs, mask) in enumerate(focused):
            allocation(exe, f'focused_{i}', rs, mask)
        expected = b'TMM1\x03\x00\x00\x00\x00\x01\x02'
        for i in range(20):
            assert check(exe, f'deterministic_{i}', wire([(0, 9)]*3), 0) == expected
            COUNTS['valid_allocations'] += 1
        for mask in range(256):
            capacity = 8-mask.bit_count()
            allocation(exe, f'mask_{mask}_reuse', [(i, i+1) for i in range(64)], mask)
            allocation(exe, f'mask_{mask}_fit', [(0, 5)]*max(1, capacity), mask)
            allocation(exe, f'mask_{mask}_pressure', [(0, 5)]*(capacity+1), mask)
        choices = [(a, b) for a in range(3) for b in range(a+1, 4)]
        for count in range(1, 5):
            for i, combo in enumerate(itertools.combinations_with_replacement(choices, count)):
                for mask in (0xf8, 0xfe):
                    allocation(exe, f'exhaustive_{count}_{i}_{mask}', sorted(combo), mask)
        rng = random.Random(20261027)
        for i in range(600):
            span = 512 if i%2 == 0 else 12
            starts = sorted(rng.randrange(span) for _ in range(rng.randint(1, 64)))
            rs = [(a, a+rng.randrange(1, 29)) for a in starts]
            allocation(exe, f'random_{i}', rs, rng.randrange(256))
        good = wire([(0, 9)]*8)
        invalid = [good[:i] for i in range(len(good))]
        invalid += [good+b'x', b'XTILE001'+good[8:]]
        invalid += [good[:8]+bytes([n])+good[9:] for n in (0, 65, 255)]
        invalid += [good[:k]+b'\x01'+good[k+1:] for k in range(10, 16)]
        invalid += [wire([(2, 2)]), wire([(2, 1)]), wire([(5, 9), (4, 6)]),
                    wire([(0, 9)]*9+[(10, 10)]), wire([(0, 1), (2, 2)], 255),
                    wire([(0xffffffff, 0)]), wire([(0xffffffff, 0xffffffff)])]
        for i, data in enumerate(invalid):
            check(exe, f'malformed_{i}', data, 65); COUNTS['malformed_input'] += 1
        for name in ('dev_full', 'closed_pipe'):
            if name == 'dev_full':
                fd = os.open('/dev/full', os.O_WRONLY)
            else:
                reader, fd = os.pipe(); os.close(reader)
            try:
                p = subprocess.run([str(exe)], input=wire([(0, 1)]), stdout=fd,
                                   stderr=subprocess.PIPE, timeout=5)
            finally:
                os.close(fd)
            assert p.returncode == 74 and not p.stderr, (name, p.returncode)
            CASES.append({'case': name, 'expected_exit': 74, 'observed_exit': p.returncode})
            COUNTS['io_and_cli'] += 1
        with open('/dev/null', 'rb') as stream:
            p = subprocess.run([str(exe)], stdin=stream, capture_output=True, timeout=5)
        assert p.returncode == 65 and not p.stdout
        COUNTS['io_and_cli'] += 1
        CASES.append({'case': 'empty_stdin', 'expected_exit': 65, 'observed_exit': p.returncode})
        p = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert p.stdin is not None and p.stdout is not None and p.stderr is not None
        data = wire([(0, 9)]*3)
        try:
            for at in range(0, len(data), 3):
                p.stdin.write(data[at:at+3]); p.stdin.flush(); time.sleep(.001)
            p.stdin.close()
            assert p.wait(timeout=5) == 0 and p.stdout.read() == expected and not p.stderr.read()
        finally:
            if p.poll() is None:
                p.kill(); p.wait()
            p.stdout.close(); p.stderr.close()
        COUNTS['io_and_cli'] += 1
        CASES.append({'case': 'fragmented_reads', 'expected_exit': 0, 'observed_exit': 0})
        for suffix in ('second',):
            obj2, exe2 = temp/(suffix+'.o'), temp/suffix
            command(['as', '--64', '-o', str(obj2), str(SOURCE)])
            command(['ld', '-static', '--build-id=none', '-z', 'noexecstack', '-o', str(exe2), str(obj2)])
            assert exe2.read_bytes() == exe.read_bytes(), 'non-reproducible build'
        raw = exe.read_bytes()
        parts = elf_parts(raw)
        assert sha(SOURCE.read_bytes()) == source_hash
        (OUTPUT/'tile_allocator').write_bytes(raw)
        (OUTPUT/'tile_allocator').chmod(0o755)
        (OUTPUT/'coloring.bin').write_bytes(bytes.fromhex(parts.pop('critical_hex')))
        (OUTPUT/'disassembly.txt').write_bytes(disassembly)
        result = {'schema': 'niyah-tile-interval-coloring-v2', 'status': 'PASS',
                  'exit_code': 0, 'source_sha256': source_hash,
                  'binary_sha256': sha(raw), 'binary_bytes': len(raw), **parts,
                  'counts': COUNTS, 'tests': sum(COUNTS.values()),
                  'observed_cases': len(CASES),
                  'elapsed_seconds': time.perf_counter()-started,
                  'rebuild_byte_identical': True,
                  'amx_hardware_executed': False,
                  'compiler_ir_integration': False,
                  'scope': 'sorted half-open intervals, 1..64 values, 8 uniform-shape resources, no spills',
                  'commands': COMMANDS}
        (OUTPUT/'case_receipt.json').write_text(json.dumps(CASES, indent=2)+'\n')
        (OUTPUT/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

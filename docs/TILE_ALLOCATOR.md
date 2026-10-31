# Tile-register interval allocator

`src/tile_allocator.s` assigns register IDs to sorted half-open live intervals. It is a static Linux executable using syscalls. It does not execute AMX instructions. `Makefile.ir` connects it to the external niyah-lang compiler bridge.

```sh
make tile-allocator test-tile-allocator
```

## Contract

Input uses the `NTILE001` header, a reservation mask and unsigned start/end pairs. Output uses the `TMM1` header and tile IDs. The input domain is 1..64 intervals and 8 physical registers. The source header, sizes and error codes are recorded in [`evidence/docs-refresh-20261031/contract-excerpts.json`](../evidence/docs-refresh-20261031/contract-excerpts.json). Input must reach EOF; validation precedes allocation.

For each interval in start order, reuse the lowest unreserved register whose prior interval has ended. If all available registers remain live at the new interval's start, that instant requires another register. This is standard interval-graph coloring on the declared sorted-interval domain. The caller supplies liveness; the allocator does not infer it.

There are no spills, control-flow joins, tile reconfiguration or arbitrary interference graphs. A successful assignment is not hardware AMX execution. Linux permission, tile configuration, packing and CPU dispatch belong to the execution path.

`tests/test_tile_allocator.py` contains an independent overlap oracle, reservation masks, malformed streams, output failures and reproducible-build checks. No finished allocator execution receipt was found in the evidence directories at inspection; the previous test-count claim was omitted. [`evidence/docs-refresh-20261031/claims.json`](../evidence/docs-refresh-20261031/claims.json)

Linux xstate and Intel AMX references are retained in [SOURCES.md](SOURCES.md). [Bridge contract](TENSOR_IR.md).

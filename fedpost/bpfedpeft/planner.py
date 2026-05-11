from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence


@dataclass(frozen=True)
class BlockSpec:
    """Inclusive, zero-based transformer-layer interval."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("block start must be non-negative")
        if self.end < self.start:
            raise ValueError("block end must be >= start")

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.end + 1))

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end, "size": self.size}


def blocks_from_end_layers(
    end_layers: Sequence[int],
    overlap_layers: int = 1,
    one_based: bool = True,
) -> list[BlockSpec]:
    """Build overlapping blocks from configured end-layer indices."""

    if not end_layers:
        raise ValueError("end_layers must not be empty")
    if overlap_layers < 0:
        raise ValueError("overlap_layers must be non-negative")

    ends = [int(x) - 1 if one_based else int(x) for x in end_layers]
    if sorted(ends) != ends:
        raise ValueError("end_layers must be sorted")
    if len(set(ends)) != len(ends):
        raise ValueError("end_layers must be unique")
    if ends[0] < 0:
        raise ValueError("end_layers must be positive when one_based=True")

    blocks: list[BlockSpec] = []
    start = 0
    for end in ends:
        blocks.append(BlockSpec(start=start, end=end))
        start = max(0, end - overlap_layers + 1)
    return blocks


def plan_equal_blocks(
    num_layers: int,
    num_blocks: int,
    overlap_layers: int = 1,
) -> list[BlockSpec]:
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if num_blocks > num_layers:
        raise ValueError("num_blocks cannot exceed num_layers")

    ends = []
    for idx in range(1, num_blocks + 1):
        end = round(idx * num_layers / num_blocks) - 1
        if ends and end <= ends[-1]:
            end = ends[-1] + 1
        ends.append(min(end, num_layers - 1))
    ends[-1] = num_layers - 1
    return blocks_from_end_layers(ends, overlap_layers=overlap_layers, one_based=False)


def solve_cka_partition(
    similarity: Sequence[Sequence[float]],
    num_blocks: int,
    max_block_layers: int | None = None,
    overlap_layers: int = 1,
) -> list[BlockSpec]:
    """Dynamic-programming solver for Eq. (3)."""

    num_layers = len(similarity)
    if num_layers == 0:
        raise ValueError("similarity matrix must not be empty")
    if any(len(row) != num_layers for row in similarity):
        raise ValueError("similarity matrix must be square")
    if num_blocks <= 0 or num_blocks > num_layers:
        raise ValueError("num_blocks must be in [1, num_layers]")
    if overlap_layers < 0:
        raise ValueError("overlap_layers must be non-negative")

    if num_blocks == 1:
        return [BlockSpec(0, num_layers - 1)]

    max_len = max_block_layers or num_layers
    if max_len <= 0:
        raise ValueError("max_block_layers must be positive")

    dp = [[inf] * (num_blocks + 1) for _ in range(num_layers)]
    prev = [[None] * (num_blocks + 1) for _ in range(num_layers)]

    for end in range(num_layers):
        if end + 1 <= max_len:
            dp[end][1] = 0.0

    for k in range(2, num_blocks + 1):
        for end in range(num_layers):
            best_cost = inf
            best_prev = None
            for prev_end in range(end):
                if dp[prev_end][k - 1] == inf:
                    continue
                start = max(0, prev_end - overlap_layers + 1)
                if end - start + 1 > max_len:
                    continue
                if prev_end + 1 >= num_layers:
                    continue
                cost = dp[prev_end][k - 1] + float(similarity[prev_end][prev_end + 1])
                if cost < best_cost:
                    best_cost = cost
                    best_prev = prev_end
            dp[end][k] = best_cost
            prev[end][k] = best_prev

    if dp[num_layers - 1][num_blocks] == inf:
        raise ValueError("no feasible BP-FedPEFT partition satisfies the constraints")

    ends = []
    end = num_layers - 1
    for k in range(num_blocks, 0, -1):
        ends.append(end)
        end = prev[end][k]
        if end is None and k > 1:
            raise RuntimeError("partition backtracking failed")
    ends.reverse()
    return blocks_from_end_layers(ends, overlap_layers=overlap_layers, one_based=False)


def linear_cka(x: Sequence[Sequence[float]], y: Sequence[Sequence[float]]) -> float:
    """Dependency-free Linear CKA for planning smoke tests."""

    x_centered = _center_columns(x)
    y_centered = _center_columns(y)
    xy = _matmul_transpose_left(x_centered, y_centered)
    xx = _matmul_transpose_left(x_centered, x_centered)
    yy = _matmul_transpose_left(y_centered, y_centered)

    numerator = _frobenius_sq(xy)
    denominator = (_frobenius_sq(xx) ** 0.5) * (_frobenius_sq(yy) ** 0.5)
    return 0.0 if denominator == 0.0 else numerator / denominator


def _center_columns(values: Sequence[Sequence[float]]) -> list[list[float]]:
    rows = [list(map(float, row)) for row in values]
    if not rows:
        raise ValueError("matrix must have at least one row")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("all rows must have the same length")

    means = [sum(row[j] for row in rows) / len(rows) for j in range(width)]
    return [[row[j] - means[j] for j in range(width)] for row in rows]


def _matmul_transpose_left(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    if len(a) != len(b):
        raise ValueError("matrices must have the same number of rows")
    a_width = len(a[0])
    b_width = len(b[0])
    out = [[0.0 for _ in range(b_width)] for _ in range(a_width)]
    for row_a, row_b in zip(a, b):
        for i in range(a_width):
            for j in range(b_width):
                out[i][j] += row_a[i] * row_b[j]
    return out


def _frobenius_sq(matrix: Sequence[Sequence[float]]) -> float:
    return sum(value * value for row in matrix for value in row)

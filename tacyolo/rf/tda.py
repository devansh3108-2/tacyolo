from __future__ import annotations

import numpy as np


def connected_components(points: np.ndarray, eps: float = 3.0) -> list[list[int]]:
    """Union-find clustering = Betti-0 on a Vietoris–Rips graph at scale eps."""
    n = len(points)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) <= eps:
                union(i, j)
    buckets: dict[int, list[int]] = {}
    for i in range(n):
        buckets.setdefault(find(i), []).append(i)
    return list(buckets.values())


def persistence_1d(values: np.ndarray) -> list[tuple[float, float]]:
    """Sublevel-set 1D persistence on a sorted 1D signal (cheap TDA, not full Rips)."""
    x = np.asarray(values, dtype=np.float32).ravel()
    if x.size == 0:
        return []
    order = np.argsort(x)
    parent = list(range(len(x)))
    born = np.full(len(x), -1.0, dtype=np.float32)
    alive = np.zeros(len(x), dtype=bool)
    pairs: list[tuple[float, float]] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for idx in order:
        alive[idx] = True
        born[idx] = x[idx]
        parent[idx] = idx
        for nb in (idx - 1, idx + 1):
            if 0 <= nb < len(x) and alive[nb]:
                a, b = find(idx), find(nb)
                if a == b:
                    continue
                # older component (lower birth) survives
                if born[a] <= born[b]:
                    pairs.append((float(born[b]), float(x[idx])))
                    parent[b] = a
                else:
                    pairs.append((float(born[a]), float(x[idx])))
                    parent[a] = b
    # essential class
    roots = {find(i) for i in range(len(x)) if alive[i]}
    death = float(x.max())
    for r in roots:
        pairs.append((float(born[r]), death))
    pairs.sort(key=lambda p: p[1] - p[0], reverse=True)
    return pairs[:8]


def tda_summary(peaks: list[tuple[int, int, float]], eps: float = 3.0) -> tuple[int, list[tuple[float, float]]]:
    if not peaks:
        return 0, []
    pts = np.array([[p[0], p[1]] for p in peaks], dtype=np.float32)
    comps = connected_components(pts, eps=eps)
    energies = np.array([p[2] for p in peaks], dtype=np.float32)
    return len(comps), persistence_1d(energies)

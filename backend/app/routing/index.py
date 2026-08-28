"""Integer-indexed view of a request graph, for the hot search loop.

Yen's algorithm runs a lot of Dijkstras -- one per spur node, per iteration,
per weighting. Doing that against dictionaries keyed by string node ids spends
most of its time hashing strings. This module flattens the request graph into
arrays once per request; the search then touches nothing but integers, floats
and Python lists.

Nothing here changes what is computed, only how fast it is computed.
"""

from __future__ import annotations

import numpy as np

KIND_ROAD, KIND_TRANSIT, KIND_TRANSFER, KIND_RIDE, KIND_ACCESS = 0, 1, 2, 3, 4
_KIND_CODE = {"road": KIND_ROAD, "transit": KIND_TRANSIT, "transfer": KIND_TRANSFER,
              "ride": KIND_RIDE, "access": KIND_ACCESS}

NO_ROUTE = -1


class SearchIndex:
    """Flattened, integer-keyed request graph plus its cost planes."""

    __slots__ = ("node_ids", "node_of", "out", "e_u", "e_v", "e_kind", "e_route",
                 "money", "travel", "boarding_wait", "ride_wait", "bucket_starts",
                 "n_nodes", "n_edges", "n_buckets", "n_routes", "bucket_lut",
                 "_plane_cache")

    def __init__(self, graph, costs, bucket_starts: tuple[float, ...]):
        self.node_ids: list[str] = list(graph.nodes)
        self.node_of: dict[str, int] = {n: i for i, n in enumerate(self.node_ids)}
        self.n_nodes = len(self.node_ids)
        self.n_edges = len(graph.edges)
        self.bucket_starts = bucket_starts
        self.n_buckets = len(bucket_starts)

        route_of: dict[str, int] = {}
        e_u = [0] * self.n_edges
        e_v = [0] * self.n_edges
        e_kind = [0] * self.n_edges
        e_route = [NO_ROUTE] * self.n_edges

        for i, e in enumerate(graph.edges):
            e_u[i] = self.node_of[e.u]
            e_v[i] = self.node_of[e.v]
            e_kind[i] = _KIND_CODE.get(e.kind, KIND_ROAD)
            if e.kind == "transit" and e.route_id:
                e_route[i] = route_of.setdefault(e.route_id, len(route_of))

        self.e_u, self.e_v, self.e_kind, self.e_route = e_u, e_v, e_kind, e_route

        out: list[list[int]] = [[] for _ in range(self.n_nodes)]
        for node_id, edge_list in graph.out_adj.items():
            ni = self.node_of.get(node_id)
            if ni is not None:
                out[ni] = list(edge_list)
        self.out = out

        # Cost planes as plain Python lists: indexing a list element-by-element
        # is markedly cheaper than indexing a NumPy array the same way.
        self.travel: list[list[float]] = [
            (row.tolist() if isinstance(row, np.ndarray) else list(row))
            for row in costs.travel
        ]
        self.boarding_wait: list[list[float]] = [
            (row.tolist() if isinstance(row, np.ndarray) else list(row))
            for row in costs.boarding_wait
        ]
        self.ride_wait: list[float] = costs.ride_wait.tolist()
        self.money: list[float] = costs.money.tolist()
        self.n_routes = len(route_of)

        # Elapsed-minute -> bucket, as a flat lookup so the inner loop never
        # scans the bucket boundaries. Anything past the last bucket clamps.
        span = int(bucket_starts[-1]) + 60
        self.bucket_lut: list[int] = []
        for m in range(span):
            b = 0
            for i in range(self.n_buckets):
                if m >= bucket_starts[i]:
                    b = i
                else:
                    break
            self.bucket_lut.append(b)
        self._plane_cache: dict[float, tuple] = {}

    def bucket_for(self, elapsed_min: float) -> int:
        i = int(elapsed_min)
        lut = self.bucket_lut
        return lut[i] if 0 <= i < len(lut) else lut[-1]

    def planes(self, money_weight: float):
        """Pre-add everything that does not depend on search state.

        Returns four [bucket][edge] planes:
          min_stay / w_stay   staying on the vehicle you are already on
          min_board / w_board  boarding, so the headway wait applies
        `w_*` are search weights (minutes + money_weight x rupees); `min_*` are
        pure minutes, which is what advances the clock.
        """
        cached = self._plane_cache.get(money_weight)
        if cached is not None:
            return cached
        n = self.n_edges
        ride_extra = [self.ride_wait[i] if self.e_kind[i] == KIND_RIDE else 0.0
                      for i in range(n)]
        money_term = [money_weight * self.money[i] for i in range(n)]
        min_stay, w_stay, min_board, w_board = [], [], [], []
        for b in range(self.n_buckets):
            tb, wb = self.travel[b], self.boarding_wait[b]
            ms = [tb[i] + ride_extra[i] for i in range(n)]
            mb = [ms[i] + wb[i] for i in range(n)]
            min_stay.append(ms)
            min_board.append(mb)
            w_stay.append([ms[i] + money_term[i] for i in range(n)])
            w_board.append([mb[i] + money_term[i] for i in range(n)])
        out = (min_stay, w_stay, min_board, w_board)
        self._plane_cache[money_weight] = out
        return out

    def step_cost(self, edge_i: int, cur_route: int, elapsed: float) -> float:
        """Minutes to traverse `edge_i`, including any wait charged on boarding."""
        b = self.bucket_for(elapsed)
        m = self.travel[b][edge_i]
        k = self.e_kind[edge_i]
        if k == KIND_TRANSIT:
            if self.e_route[edge_i] != cur_route:
                m += self.boarding_wait[b][edge_i]
        elif k == KIND_RIDE:
            m += self.ride_wait[edge_i]
        return m

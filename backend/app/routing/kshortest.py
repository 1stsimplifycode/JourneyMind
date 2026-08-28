"""Candidate journey generation: Yen's k-shortest paths on the multimodal graph.

Three details make this different from a textbook Yen's:

1. **State, not just node.** Waiting for a bus is charged when you *board*, not
   at every stop, so the search state is `(node, route you are sitting on,
   boardings used)`. Without the route in the state, a path that stays on one
   metro line would be charged a fresh wait at every station. Without the
   boarding count, a transfer cap could not be enforced -- two ways of reaching
   the same station with different numbers of changes are genuinely different
   states.

2. **Time-dependent weights.** An edge entered 20 minutes into the journey is
   priced from the 15-30 minute bucket, not from the departure-time prediction.

3. **Several weightings, not one.** k-shortest by time returns k variations on
   the fastest trip -- all expensive, none cheap, and a Pareto frontier built
   from them would collapse to a single point. So the search runs under a
   family of blended time/money weightings and the results are pooled.

This is explicitly an approximation. The exact problem -- cheapest path that
also fits a time limit -- is the Resource Constrained Shortest Path Problem,
which is NP-hard. We generate a good candidate set and say so.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from ..graph.builder import DESTINATION_ID, ORIGIN_ID
from .costs import BUCKET_STARTS, CostTable
from .index import KIND_RIDE, KIND_TRANSIT, NO_ROUTE, SearchIndex

# Single-mode reference journeys. One Dijkstra each, restricted to walking plus
# one vehicle mode. Two jobs at once: they guarantee the candidate set contains
# the options a user would have compared by hand (bus vs Rapido vs auto vs cab
# vs metro), and they are baseline 5 from the documentation -- "does multi-modal
# planning actually beat what the apps do today?"
REFERENCE_MODES = ("metro", "bus", "bike_taxi", "auto", "cab")

# (weight on money, label). Money weight 0 is pure time.
BLENDS: tuple[tuple[float, str], ...] = (
    (0.00, "fastest"),
    (0.06, "time-leaning"),
    (0.18, "balanced"),
    (0.45, "cost-leaning"),
    (1.20, "cheapest"),
)

MAX_EXPANSIONS = 40_000
# Yen's explores a spur from every position along the previous path. On long
# walking paths that is dozens of Dijkstras for very little added diversity,
# so the spur sweep is capped. A documented truncation, not a silent one.
MAX_SPUR_POSITIONS = 14


@dataclass(frozen=True)
class PathResult:
    edges: tuple[int, ...]
    weight: float
    origin_blend: str


def _dijkstra(ix: SearchIndex, money_weight: float, source: int, target: int,
              banned_edges: frozenset, banned_nodes: frozenset,
              start_route: int, start_elapsed: float,
              max_boardings: int, start_boardings: int = 0):
    """Dijkstra over states `(node, route you are sitting on, boardings used)`.

    The state is packed into a single integer. Yen's calls this function
    hundreds of times per request, and integer keys avoid allocating and
    hashing a tuple on every one of roughly a million edge relaxations.
    """
    out, e_v, e_kind, e_route = ix.out, ix.e_v, ix.e_kind, ix.e_route
    lut, n_lut = ix.bucket_lut, len(ix.bucket_lut)
    min_stay, w_stay, min_board, w_board = ix.planes(money_weight)

    n_slots = ix.n_routes + 1            # route slot 0 means "on foot"
    n_board_slots = max_boardings + 1
    stride = n_slots * n_board_slots

    def pack(node: int, route: int, boards: int) -> int:
        return node * stride + (route + 1) * n_board_slots + boards

    start = pack(source, start_route, start_boardings)
    dist = {start: 0.0}
    elapsed = {start: start_elapsed}
    prev: dict[int, tuple[int, int] | None] = {start: None}
    pq = [(0.0, 0, start)]
    seen = set()
    tick = 0
    expansions = 0

    while pq:
        d, _, state = heapq.heappop(pq)
        if state in seen:
            continue
        seen.add(state)
        node = state // stride
        rest = state - node * stride
        cur_route = rest // n_board_slots - 1
        n_board = rest - (cur_route + 1) * n_board_slots

        if node == target:
            path: list[int] = []
            s: int | None = state
            while prev.get(s) is not None:
                s, edge_i = prev[s]        # type: ignore[misc]
                path.append(edge_i)
            path.reverse()
            return path, d

        expansions += 1
        if expansions > MAX_EXPANSIONS:
            break

        t_now = elapsed[state]
        bi = int(t_now)
        b = lut[bi] if bi < n_lut else lut[-1]
        ms_b, ws_b, mb_b, wb_b = min_stay[b], w_stay[b], min_board[b], w_board[b]

        for edge_i in out[node]:
            if edge_i in banned_edges:
                continue
            v = e_v[edge_i]
            if v in banned_nodes:
                continue
            kind = e_kind[edge_i]

            if kind == KIND_TRANSIT:
                route = e_route[edge_i]
                if route == cur_route:
                    minutes, weight = ms_b[edge_i], ws_b[edge_i]
                    n_board2 = n_board
                else:
                    minutes, weight = mb_b[edge_i], wb_b[edge_i]
                    n_board2 = n_board + 1
                    if n_board2 > max_boardings:
                        continue
                carry = route
            elif kind == KIND_RIDE:
                n_board2 = n_board + 1
                if n_board2 > max_boardings:
                    continue
                minutes, weight = ms_b[edge_i], ws_b[edge_i]
                carry = NO_ROUTE
            else:
                minutes, weight = ms_b[edge_i], ws_b[edge_i]
                n_board2 = n_board
                carry = cur_route      # walking does not end your seat on a line

            nd = d + weight
            nxt = v * stride + (carry + 1) * n_board_slots + n_board2
            if nd < dist.get(nxt, 1e18):
                dist[nxt] = nd
                elapsed[nxt] = t_now + minutes
                prev[nxt] = (state, edge_i)
                tick += 1
                heapq.heappush(pq, (nd, tick, nxt))
    return None


def replay(ix: SearchIndex, path) -> tuple[float, int, int]:
    """Walk a path and return (elapsed minutes, route you are on, boardings used)."""
    t = 0.0
    cur_route = NO_ROUTE
    boardings = 0
    for edge_i in path:
        kind = ix.e_kind[edge_i]
        route = ix.e_route[edge_i]
        if kind == KIND_RIDE:
            boardings += 1
        elif kind == KIND_TRANSIT and route != cur_route:
            boardings += 1
        t += ix.step_cost(edge_i, cur_route, t)
        if kind == KIND_TRANSIT:
            cur_route = route
        elif kind == KIND_RIDE:
            cur_route = NO_ROUTE
    return t, cur_route, boardings


def path_weight(ix: SearchIndex, money_weight: float, path) -> float:
    """Re-evaluate a whole path with correct elapsed times and boarding waits."""
    total = 0.0
    t = 0.0
    cur_route = NO_ROUTE
    for edge_i in path:
        m = ix.step_cost(edge_i, cur_route, t)
        total += m + money_weight * ix.money[edge_i]
        t += m
        kind = ix.e_kind[edge_i]
        if kind == KIND_TRANSIT:
            cur_route = ix.e_route[edge_i]
        elif kind == KIND_RIDE:
            cur_route = NO_ROUTE
    return total


def yen_k_shortest(ix: SearchIndex, k: int, money_weight: float, max_boardings: int,
                   source: int, target: int, blend_label: str = "") -> list[PathResult]:
    first = _dijkstra(ix, money_weight, source, target, frozenset(), frozenset(),
                      NO_ROUTE, 0.0, max_boardings)
    if first is None:
        return []
    accepted = [PathResult(tuple(first[0]),
                           path_weight(ix, money_weight, first[0]), blend_label)]
    candidates: list[tuple[float, tuple[int, ...]]] = []
    seen = {accepted[0].edges}

    while len(accepted) < k:
        prev_path = list(accepted[-1].edges)
        for i in range(min(len(prev_path), MAX_SPUR_POSITIONS)):
            root = prev_path[:i]
            spur_node = ix.e_u[prev_path[i]]

            banned_e = frozenset(
                p.edges[i] for p in accepted
                if len(p.edges) > i and list(p.edges[:i]) == root
            )
            banned_n = frozenset({ix.e_u[e] for e in root} - {spur_node})

            t, cur_route, used = replay(ix, root)
            spur = _dijkstra(ix, money_weight, spur_node, target, banned_e, banned_n,
                             cur_route, t, max_boardings, start_boardings=used)
            if spur is None:
                continue
            full = tuple(root + spur[0])
            if full in seen:
                continue
            seen.add(full)
            heapq.heappush(candidates, (path_weight(ix, money_weight, list(full)), full))

        if not candidates:
            break
        w, best = heapq.heappop(candidates)
        accepted.append(PathResult(best, w, blend_label))

    return accepted


def _direct_ride_edge(graph, mode: str, source: int, target: int, ix):
    """The one hailed edge from the rider's door to their destination."""
    src_id, dst_id = ix.node_ids[source], ix.node_ids[target]
    for i, e in enumerate(graph.edges):
        if e.kind == "ride" and e.mode == mode and e.u == src_id and e.v == dst_id:
            return i
    return None


def single_mode_paths(graph, ix: SearchIndex, source: int, target: int,
                      modes=REFERENCE_MODES) -> list[PathResult]:
    """The best journey using walking plus exactly one vehicle mode.

    These are the options a person would otherwise have compared by opening
    four apps. Including them means the recommendation is measured against
    them rather than asserted to be better.
    """
    out: list[PathResult] = []
    for mode in modes:
        # A hailed mode's reference journey is the DOOR-TO-DOOR ride, full
        # stop. Letting a search find it meant that on a real road network a
        # two-hop chain through a hub could come out shorter than the direct
        # edge's straight-line estimate -- so every "bike taxi only" reference
        # was two bike taxis, the validator rightly threw it out, and the mode
        # lost its card entirely. One vehicle is what the row means.
        direct = _direct_ride_edge(graph, mode, source, target, ix)
        if direct is not None:
            out.append(PathResult((direct,), path_weight(ix, 0.0, [direct]),
                                  f"{mode}-direct"))
            continue

        banned = frozenset(
            i for i, e in enumerate(graph.edges)
            if e.mode not in ("walk", mode)
        )
        got = _dijkstra(ix, 0.0, source, target, banned, frozenset(),
                        NO_ROUTE, 0.0, max_boardings=4)
        if got is None:
            continue
        out.append(PathResult(tuple(got[0]), path_weight(ix, 0.0, got[0]),
                              f"{mode}-only"))
    # walking the whole way, for completeness and as the true cost floor
    banned_walk = frozenset(i for i, e in enumerate(graph.edges) if e.mode != "walk")
    got = _dijkstra(ix, 0.0, source, target, banned_walk, frozenset(),
                    NO_ROUTE, 0.0, max_boardings=0)
    if got is not None:
        out.append(PathResult(tuple(got[0]), path_weight(ix, 0.0, got[0]), "walk-only"))
    return out


#: Hailed vehicles, for the first/last-mile searches below.
RIDE_ONLY = ("bike_taxi", "auto", "cab")


def access_transit_paths(graph, ix, source: int, target: int,
                         max_boardings: int) -> list[PathResult]:
    """Ride to the transit, transit across the city, ride from the transit.

    The single most useful shape this product has, and pooled k-shortest was
    not reliably finding it. A generic search spends its k on variations of
    whatever is currently winning; on a long trip that is the direct ride, so
    the cheap "bike taxi to the bus, bus to the metro, bike taxi to the door"
    family never surfaced and low budgets came back as "nothing fits" while a
    ninety-rupee journey existed.

    One Dijkstra per transit mode per weighting, restricted to that mode plus
    the hailed vehicles plus walking. Four extra searches, and the family is
    guaranteed to be in the candidate set rather than lucky to be.

    Nothing here forces the result to WIN -- these are candidates like any
    other, and a direct ride beats them whenever it is genuinely better.
    """
    out: list[PathResult] = []
    for transit in ("metro", "bus"):
        allowed = {"walk", transit, *RIDE_ONLY}
        banned = frozenset(i for i, e in enumerate(graph.edges)
                           if e.mode not in allowed)
        for money_weight, label in ((1.20, "cheapest"), (0.06, "time-leaning")):
            got = _dijkstra(ix, money_weight, source, target, banned, frozenset(),
                            NO_ROUTE, 0.0, max_boardings=max_boardings)
            if got is None:
                continue
            out.append(PathResult(tuple(got[0]),
                                  path_weight(ix, money_weight, got[0]),
                                  f"{transit}+ride/{label}"))
    return out


def generate_candidates(graph, costs: CostTable, k_per_blend: int,
                        max_transfers: int, blends=BLENDS,
                        include_single_mode: bool = True) -> list[PathResult]:
    """Pool k-shortest results across several time/money weightings, plus one
    single-mode reference journey per mode."""
    ix = SearchIndex(graph, costs, BUCKET_STARTS)
    source = ix.node_of[ORIGIN_ID]
    target = ix.node_of[DESTINATION_ID]
    max_boardings = max(1, max_transfers + 1)

    pooled: dict[tuple[int, ...], PathResult] = {}
    for money_weight, label in blends:
        for r in yen_k_shortest(ix, k_per_blend, money_weight, max_boardings,
                                source, target, label):
            pooled.setdefault(r.edges, r)
    if include_single_mode:
        for r in single_mode_paths(graph, ix, source, target):
            pooled.setdefault(r.edges, r)
        for r in access_transit_paths(graph, ix, source, target, max_boardings):
            pooled.setdefault(r.edges, r)
    return list(pooled.values())

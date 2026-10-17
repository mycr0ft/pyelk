"""Brandes-Köpnig horizontal coordinate assignment.

ELK layered ``nodePlacement.strategy = BRANDES_KOZEN`` (alias accepted:
``BRANDES``).  Four vertical-alignment scans (align up/down x traverse
left/right) greedily build non-crossing alignment chains over edge-chain
segments (source -> dummy nodes -> target), resolving conflicts by
endpoint degree (segments with a higher combined degree steal the slot).
Each scan then packs the chains so every node in a chain shares one
centre — long edges run straight through their dummy nodes — using
per-layer cursors so nothing overlaps.  The scan with the smallest total
width wins; ties prefer more aligned segments.  Deterministic.

Reference: Ulrik Brandes and Boris Köpnig, "Fast and Simple Horizontal
Coordinate Assignment", in DAG Drawing: Beyond Layered Approach (2002).
"""

from collections import defaultdict


def align_and_pack(layers, all_nodes, chains, node_spacing, layer_spacing,
                   padding, horizontal):
    """Return {'step': {id(n): coord}, 'along': {id(n): coord}}.

    ``step`` is the within-layer coordinate (x for DOWN/UP layouts, y for
    RIGHT/LEFT layouts); ``along`` the layer stacking coordinate.
    """
    nodes = list(all_nodes)
    if not nodes:
        return {'step': {}, 'along': {}}

    nbr_up, nbr_down, degree = _neighbours(layers, chains)
    positions = [{id(n): i for i, n in enumerate(layer)} for layer in layers]

    best_key = None
    placed = None
    for adj in ('down', 'up'):
        for within in ('left', 'right'):
            nbr = nbr_down if adj == 'down' else nbr_up
            segs, unaligned = _scan(layers, positions, nbr, degree, adj,
                                    within)
            result = _pack(layers, segs, node_spacing, layer_spacing,
                           padding, horizontal)
            width = _extent(result['step'])
            key = (unaligned, round(width, 3))
            if best_key is None or key < best_key:
                best_key = key
                placed = result

    for n in nodes:
        placed['step'].setdefault(id(n), 0.0)
        placed['along'].setdefault(id(n), 0.0)
    return placed


def _neighbours(layers, chains):
    """Adjacent-layer neighbours via chain segments; degree = incident
    chain segments +2 per node (the BK alignment priority)."""
    nbr_up = defaultdict(list)      # id -> [(id, node)] in layer li-1
    nbr_down = defaultdict(list)    # id -> [(id, node)] in layer li+1
    deg = defaultdict(int)
    for layer in layers:
        for n in layer:
            deg[id(n)] += 0
    for chain in chains:
        for (la, a), (lb, b) in zip(chain, chain[1:]):
            if lb == la + 1:
                nbr_down[id(a)].append((id(b), b))
                nbr_up[id(b)].append((id(a), a))
            elif la == lb + 1:
                nbr_down[id(b)].append((id(a), a))
                nbr_up[id(a)].append((id(b), b))
            deg[id(a)] += 1
            deg[id(b)] += 1
    return nbr_up, nbr_down, deg


def _scan(layers, positions, nbr, degree, adj, within):
    """One vertical-alignment pass; returns (segments, unaligned_count).

    ``segments`` = list of ((layer_i, u), (layer_j, w)) with
    |li - lj| == 1.  A candidate crossing an already-aligned segment is
    dropped unless its combined endpoint degree is strictly higher (then
    it steals the slot)."""
    adj_off = 1 if adj == 'down' else -1
    layer_order = (list(range(len(layers))) if adj == 'down'
                   else list(range(len(layers) - 1, -1, -1)))

    cands = []
    for li in layer_order:
        if li + adj_off < 0 or li + adj_off >= len(layers):
            continue
        nj = li + adj_off
        side = 'down' if adj == 'down' else 'up'
        for v in layers[li]:
            if positions[li].get(id(v)) is None:
                continue
            for wid, w in nbr.get(id(v), []):
                if w.layer != nj or w is v:
                    continue
                if positions[nj].get(id(w)) is None:
                    continue
                pv = positions[li][id(v)]
                pw = positions[nj][id(w)]
                if within == 'right':
                    pv, pw = -pv, -pw
                cands.append((min(pv, pw), ((li, v), (nj, w))))
    cands.sort(key=lambda t: t[0])

    # group candidates per node (already free-node-first via the greedy
    # scan); even neighbour counts offer the r AND l median candidates —
    # the scan's within-direction decides the preference, and both are
    # attempted before giving up (Brandes-Köpnig algorithm 1)
    segs = []
    endpoints = set()
    by_node = defaultdict(list)
    for _, seg in cands:
        (li, v), (lj, w) = seg
        key = (li, id(v))
        by_node[key].append(seg)
    for li, layer in enumerate(layers):
        order = (layer if within == 'left' else list(reversed(layer)))
        for v in order:
            group = sorted(by_node.get((li, id(v)), []),
                           key=lambda s: (positions[s[1][0]].get(id(s[1][1]),
                                                                0)
                                          * (-1 if within == 'right' else 1)))
            for seg in group:
                (lj, w) = seg[1]
                if id(v) in endpoints:
                    break           # v already aligned
                if id(w) in endpoints:
                    continue
                crossing = [s for s in segs if _crosses(seg, s, positions)]
                if crossing:
                    my = degree[id(v)] + degree[id(w)]
                    their = max(degree[id(a1)] + degree[id(a2)]
                                for s in crossing for a1, a2 in
                                [(s[0][1], s[1][1])])
                    if my <= their:
                        continue
                    for s in crossing:
                        segs.remove(s)
                        endpoints.discard(id(s[0][1]))
                        endpoints.discard(id(s[1][1]))
                segs.append(seg)
                endpoints.add(id(v))
                endpoints.add(id(w))

    unaligned = sum(1 for layer in layers for n in layer
                    if id(n) not in endpoints and nbr.get(id(n)))
    return segs, unaligned


def _crosses(seg, other, positions):
    """Two segments spanning the SAME adjacent-layer pair cross when
    their position orders disagree between the two layers."""
    (la1, a1), (la2, a2) = seg
    (lb1, b1), (lb2, b2) = other
    if {la1, la2} != {lb1, lb2} or la1 == la2:
        return False
    pa1 = positions[la1].get(id(a1))
    pa2 = positions[la2].get(id(a2))
    pb1 = positions[lb1].get(id(b1))
    pb2 = positions[lb2].get(id(b2))
    if None in (pa1, pa2, pb1, pb2):
        return False
    # orient both segments upper-layer-first
    if la2 < la1:
        (la1, a1), (la2, a2) = (la2, a2), (la1, a1)
        pa1, pa2 = pa2, pa1
    if lb2 < lb1:
        (lb1, b1), (lb2, b2) = (lb2, b2), (lb1, b1)
        pb1, pb2 = pb2, pb1
    if la1 != lb1:
        return False
    return ((pa1 - pb1) > 0) != ((pa2 - pb2) > 0)


def _build_chains(layers, segs):
    """Chains = maximal alignment paths over the segment set.  Within one
    scan every node belongs to at most one segment, so paths are simple;
    process nodes in layer order so each path is assembled from its
    topmost node."""
    nxt = defaultdict(dict)      # id(node) -> {layer: other node}
    for (li, v), (lj, w) in segs:
        nxt[id(v)][lj] = w
        nxt[id(w)][li] = v
    chains = []
    seen = set()
    for li, layer in enumerate(layers):
        for n in layer:
            if id(n) in seen:
                continue
            left, right, cur, cl = [], [], n, li
            while id(cur) in nxt and (cl - 1) in nxt[id(cur)]:
                cur = nxt[id(cur)][cl - 1]
                cl -= 1
                left.append((cl, cur))
            cur, cl = n, li
            while id(cur) in nxt and (cl + 1) in nxt[id(cur)]:
                cur = nxt[id(cur)][cl + 1]
                cl += 1
                right.append((cl, cur))
            chain = list(reversed(left)) + [(li, n)] + right
            ok = True
            for cl2, c in chain:
                if id(c) in seen:
                    ok = False
                    break
            if not ok:
                continue
            for _, c in chain:
                seen.add(id(c))
            chains.append(chain)
    return chains


def _pack(layers, segs, node_spacing, layer_spacing, padding, horizontal):
    """Cursor-based packing; chains share one centre x (or y)."""
    step = {}
    along = {}

    cur = padding['top' if not horizontal else 'left']
    for layer in layers:
        # DOWN/UP stack along y: the layer's y-extent is node HEIGHT;
        # RIGHT/LEFT stack along x: node WIDTH
        max_other = max(((n.width if horizontal else n.height)
                         for n in layer), default=0.0)
        for n in layer:
            along[id(n)] = cur
        cur += max_other + layer_spacing

    chains = _build_chains(layers, segs)
    in_chain = {id(m) for chain in chains for _, m in chain}
    for li, layer in enumerate(layers):
        for n in layer:
            if id(n) not in in_chain:
                chains.append([(li, n)])

    start_limit = padding['left' if not horizontal else 'top'] - node_spacing
    cursors = [start_limit for _ in layers]

    def key(chain):
        li = min(c[0] for c in chain)
        idx = {id(n): i for i, n in enumerate(layers[li])}
        pos = min(idx.get(id(m), 0) for c, m in chain if c == li)
        return (li, pos)

    seg_map = defaultdict(list)
    for s in segs:
        for node in (s[0][1], s[1][1]):
            seg_map[id(node)].append(s)

    for chain in sorted(chains, key=key):
        want = []
        for (li, node) in chain:
            for s in seg_map.get(id(node), []):
                for cn in (s[0][1], s[1][1]):
                    if cn is not node and id(cn) in step:
                        want.append(step[id(cn)])
        desired = None
        if want:
            want.sort()
            k = len(want)
            desired = (want[k // 2] if k % 2
                       else (want[k // 2 - 1] + want[k // 2]) / 2.0)
        req = None
        for (li, node) in chain:
            half = (node.height if horizontal else node.width) / 2.0
            lo = cursors[li] + node_spacing + half
            req = lo if req is None else max(req, lo)
        c = desired if desired is not None else (req if req is not None
                                                 else start_limit)
        c = max(c, req) if req is not None else c
        for (li, node) in chain:
            step[id(node)] = c
            half = (node.height if horizontal else node.width) / 2.0
            cursors[li] = max(cursors[li], c + half)

    return {'step': step, 'along': along}


def _extent(step):
    vals = list(step.values())
    return (max(vals) - min(vals)) if vals else 0.0
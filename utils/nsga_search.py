# nsga2_transformer_backbone.py
# Minimal deps: Python 3.9+, no external libraries required.

from __future__ import annotations
import random, math, time, json, hashlib
from typing import Any, Dict, List, Tuple, Optional

# -----------------------------
# Problem definition (EDIT ME)
# -----------------------------
class SearchSpace:
    """
    Mixed search-space with typed genes:
      - "int":   bounded integer
      - "float": bounded float
      - "cat":   categorical
      - "perm":  permutation over a label list (e.g., block schedule)
    You may add more types (e.g., 'bool', 'set') as needed.
    """

    def __init__(self):
        # Example: Transformer SW/arch knobs (EDIT to match your project)
        self.genes = {
            "n_layers":     {"type": "int",   "low": 4, "high": 24},
            "d_model":      {"type": "int",   "low": 128, "high": 1024, "step": 32},
            "n_heads":      {"type": "int",   "low": 2, "high": 16},
            "mlp_ratio":    {"type": "float", "low": 1.5, "high": 6.0},
            "attn_type":    {"type": "cat",   "choices": ["scaled_dot", "gqa", "mha_linear", "flash"]},
            "n_kv_groups":  {"type": "int",   "low": 1, "high": 8},
            "seq_len":      {"type": "int",   "low": 128, "high": 8192, "step": 128},
            "quant_bits":   {"type": "int",   "low": 4, "high": 16},
            "act_sparsity": {"type": "float", "low": 0.0, "high": 0.9},
            # Example permutation gene: order of block types or pipeline stages
            "layer_order":  {"type": "perm",  "items": ["attn", "mlp", "ln"]},
        }

    def sample(self) -> Dict[str, Any]:
        x = {}
        for k, g in self.genes.items():
            t = g["type"]
            if t == "int":
                step = g.get("step", 1)
                lo = g["low"]; hi = g["high"]
                v = random.randrange(lo, hi + 1, step)
            elif t == "float":
                v = random.uniform(g["low"], g["high"])
            elif t == "cat":
                v = random.choice(g["choices"])
            elif t == "perm":
                v = g["items"][:]
                random.shuffle(v)
            else:
                raise ValueError(f"Unknown gene type {t}")
            x[k] = v
        return x

    def repair(self, x: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp/round to keep feasibility; also enforce derived constraints."""
        y = dict(x)
        for k, g in self.genes.items():
            t = g["type"]
            if t == "int":
                lo, hi = g["low"], g["high"]
                step = g.get("step", 1)
                v = max(lo, min(hi, int(round(y[k]/step)*step)))
                y[k] = v
            elif t == "float":
                lo, hi = g["low"], g["high"]
                y[k] = float(max(lo, min(hi, y[k])))
            elif t == "cat":
                if y[k] not in g["choices"]:
                    y[k] = g["choices"][0]
            elif t == "perm":
                # Ensure it's a permutation of the items
                base = g["items"]
                y[k] = _repair_perm(y[k], base)
        # Example coupled constraints:
        if y["d_model"] % y["n_heads"] != 0:
            # snap n_heads to a divisor of d_model
            divisors = [h for h in range(1, y["n_heads"]+16) if y["d_model"] % h == 0]
            if divisors:
                y["n_heads"] = min(divisors, key=lambda h: abs(h - y["n_heads"]))
        return y


def _repair_perm(p: List[str], items: List[str]) -> List[str]:
    seen = set()
    out = []
    for e in p:
        if e in items and e not in seen:
            out.append(e); seen.add(e)
    for e in items:
        if e not in seen:
            out.append(e)
    return out


# -----------------------------
# Objectives & constraints
# -----------------------------
class EvaluationResult:
    def __init__(self, objs: List[float], cons: List[float], aux: Dict[str, Any]):
        self.objs = objs   # all minimized
        self.cons = cons   # <=0 is satisfied, >0 is violated
        self.aux  = aux    # any extra logging (params, FLOPs, etc.)


class Problem:
    """
    Define objectives and constraints here.
    - All objectives are minimized. For "maximize", negate it.
    - Constraints: return positive value if violated (<=0 means OK).
    """
    def __init__(self, space: SearchSpace):
        self.space = space

        # Example objective directions:
        #   obj0: validation loss (min)
        #   obj1: -throughput (maximize throughput) -> minimize(-TP)
        #   obj2: energy per token (min)
        #   obj3: TTFT (min)
        self.n_objs = 4

        # Example hard limits (EDIT):
        self.mem_budget_bytes = 1_500_000_000  # ~1.5 GB
        self.param_budget = 120_000_000
        self.latency_limit_ms = 200.0

        self._cache: Dict[str, EvaluationResult] = {}

    def eval(self, x: Dict[str, Any]) -> EvaluationResult:
        key = self._hash(x)
        if key in self._cache:
            return self._cache[key]

        # ---- TODO: plug in your pipeline below ----
        # 1) Derive model size/FLOPs from knobs
        params = estimate_params(x)                  # int
        mem_bytes = estimate_mem_footprint(x)       # int
        flops = estimate_flops(x)                   # float

        # 2) Run training/inference quick proxy or a cached measurement
        #    Replace with your own fast proxy or async runner.
        #    Ensure returned metrics are deterministic for caching.
        val_loss, throughput, e_per_token, ttft = proxy_measure(x, params, mem_bytes, flops)

        # 3) Constraints (<= 0 satisfied)
        c1 = params - self.param_budget
        c2 = mem_bytes - self.mem_budget_bytes
        c3 = (latency_from_tp(throughput) - self.latency_limit_ms)

        # 4) Objectives (all minimized)
        objs = [
            float(val_loss),
            float(-throughput),        # maximize throughput
            float(e_per_token),
            float(ttft),
        ]
        res = EvaluationResult(objs=objs, cons=[c1, c2, c3], aux={
            "params": params,
            "mem_bytes": mem_bytes,
            "FLOPs": flops,
            "val_loss": val_loss,
            "throughput": throughput,
            "energy/token": e_per_token,
            "TTFT": ttft,
        })
        self._cache[key] = res
        return res

    @staticmethod
    def _hash(x: Dict[str, Any]) -> str:
        s = json.dumps(x, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()


# -----------------------------
# Variation operators
# -----------------------------
def sbx_crossover(a: float, b: float, low: float, high: float, eta: float = 15.0) -> Tuple[float, float]:
    """Simulated Binary Crossover for floats."""
    if random.random() < 0.5:
        if abs(a - b) < 1e-12:
            return a, b
        x1, x2 = sorted([a, b])
        rand = random.random()
        beta = 1.0 + (2.0*(x1 - low) / (x2 - x1))
        alpha = 2.0 - beta**-(eta + 1.0)
        if rand <= 1.0/alpha:
            betaq = (rand * alpha)**(1.0/(eta+1.0))
        else:
            betaq = (1.0/(2.0 - rand*alpha))**(1.0/(eta+1.0))
        c1 = 0.5*((x1 + x2) - betaq*(x2 - x1))
        beta = 1.0 + (2.0*(high - x2)/(x2 - x1))
        alpha = 2.0 - beta**-(eta + 1.0)
        if rand <= 1.0/alpha:
            betaq = (rand * alpha)**(1.0/(eta+1.0))
        else:
            betaq = (1.0/(2.0 - rand*alpha))**(1.0/(eta+1.0))
        c2 = 0.5*((x1 + x2) + betaq*(x2 - x1))
        return (min(max(c1, low), high), min(max(c2, low), high))
    return a, b


def pm_mutation(v: float, low: float, high: float, eta: float = 20.0, p: float = 0.1) -> float:
    """Polynomial mutation for floats."""
    if random.random() >= p:
        return v
    delta1 = (v - low) / (high - low)
    delta2 = (high - v) / (high - low)
    rand = random.random()
    mut_pow = 1.0/(eta + 1.0)
    if rand < 0.5:
        xy = 1.0 - delta1
        val = 2.0*rand + (1.0 - 2.0*rand)*(xy**(eta + 1.0))
        deltaq = val**mut_pow - 1.0
    else:
        xy = 1.0 - delta2
        val = 2.0*(1.0 - rand) + 2.0*(rand - 0.5)*(xy**(eta + 1.0))
        deltaq = 1.0 - val**mut_pow
    v = v + deltaq*(high - low)
    return min(max(v, low), high)


def one_point_crossover_int(a: int, b: int) -> Tuple[int, int]:
    return a, b  # integers usually recombine by picking parents; tune if needed


def mutate_int(v: int, low: int, high: int, step: int = 1, p: float = 0.1) -> int:
    if random.random() < p:
        span = ((high - low) // step) + 1
        delta = random.randint(-max(1, span//10), max(1, span//10)) * step
        v = v + delta
    return max(low, min(high, v))


def cx_cat(a: Any, b: Any) -> Tuple[Any, Any]:
    return (a if random.random() < 0.5 else b,
            b if random.random() < 0.5 else a)


def mut_cat(v: Any, choices: List[Any], p: float = 0.1) -> Any:
    if random.random() < p:
        c = random.choice(choices)
        return c if c != v else random.choice([x for x in choices if x != v] or choices)
    return v


def pmx_permutation(p1: List[Any], p2: List[Any]) -> Tuple[List[Any], List[Any]]:
    """Partially Mapped Crossover (PMX) for permutations."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    c1, c2 = p1[:], p2[:]
    map1 = {p2[i]: p1[i] for i in range(a, b+1)}
    map2 = {p1[i]: p2[i] for i in range(a, b+1)}
    c1[a:b+1] = p2[a:b+1]
    c2[a:b+1] = p1[a:b+1]
    for i in list(range(0, a)) + list(range(b+1, n)):
        while c1[i] in map1:
            c1[i] = map1[c1[i]]
        while c2[i] in map2:
            c2[i] = map2[c2[i]]
    return c1, c2


def mut_swap_perm(p: List[Any], p_mut: float = 0.2) -> List[Any]:
    q = p[:]
    if random.random() < p_mut:
        i, j = random.sample(range(len(q)), 2)
        q[i], q[j] = q[j], q[i]
    return q


def crossover(space: SearchSpace, x: Dict[str, Any], y: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    a, b = dict(x), dict(y)
    for k, g in space.genes.items():
        if random.random() > 0.5:
            continue
        t = g["type"]
        if t == "float":
            a[k], b[k] = sbx_crossover(a[k], b[k], g["low"], g["high"])
        elif t == "int":
            a[k], b[k] = one_point_crossover_int(a[k], b[k])
        elif t == "cat":
            a[k], b[k] = cx_cat(a[k], b[k])
        elif t == "perm":
            a[k], b[k] = pmx_permutation(a[k], b[k])
    return space.repair(a), space.repair(b)


def mutate(space: SearchSpace, x: Dict[str, Any], p_float=0.1, p_int=0.1, p_cat=0.1, p_perm=0.2) -> Dict[str, Any]:
    y = dict(x)
    for k, g in space.genes.items():
        t = g["type"]
        if t == "float":
            y[k] = pm_mutation(y[k], g["low"], g["high"], p=p_float)
        elif t == "int":
            y[k] = mutate_int(y[k], g["low"], g["high"], g.get("step", 1), p=p_int)
        elif t == "cat":
            y[k] = mut_cat(y[k], g["choices"], p=p_cat)
        elif t == "perm":
            y[k] = mut_swap_perm(y[k], p_mut=p_perm)
    return space.repair(y)


# -----------------------------
# NSGA-II core
# -----------------------------
def fast_non_dominated_sort(F: List[List[int]], objs: List[List[float]], cons: List[List[float]]) -> List[List[int]]:
    """
    Constraint-domination: feasible dominates infeasible.
    Among feasible or among infeasible, standard Pareto domination.
    """
    N = len(objs)
    S = [[] for _ in range(N)]
    n = [0]*N
    rank = [None]*N
    F.clear()
    first = []
    for p in range(N):
        S[p].clear()
        n[p] = 0
        for q in range(N):
            if p == q: continue
            d = dominates(objs[p], cons[p], objs[q], cons[q])
            if d == 1:
                S[p].append(q)
            elif d == -1:
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            first.append(p)
    F.append(first)
    i = 0
    while F[i]:
        Q = []
        for p in F[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    Q.append(q)
        i += 1
        F.append(Q)
    return F[:-1]


def dominates(o1: List[float], c1: List[float], o2: List[float], c2: List[float]) -> int:
    """Return 1 if 1 dominates 2, -1 if 2 dominates 1, 0 otherwise."""
    feas1 = all(c <= 0 for c in c1)
    feas2 = all(c <= 0 for c in c2)
    if feas1 and not feas2:
        return 1
    if feas2 and not feas1:
        return -1
    if feas1 and feas2:
        better = False; worse = False
        for a, b in zip(o1, o2):
            if a < b - 1e-12: better = True
            elif a > b + 1e-12: worse = True
        if better and not worse:  return 1
        if worse and not better:  return -1
        return 0
    # both infeasible: compare total violation
    v1 = sum(max(0.0, c) for c in c1)
    v2 = sum(max(0.0, c) for c in c2)
    if v1 < v2 - 1e-12: return 1
    if v1 > v2 + 1e-12: return -1
    return 0


def crowding_distance(front: List[int], objs: List[List[float]]) -> Dict[int, float]:
    if not front: return {}
    m = len(objs[0])
    dist = {i: 0.0 for i in front}
    for k in range(m):
        front_sorted = sorted(front, key=lambda i: objs[i][k])
        fmin = objs[front_sorted[0]][k]
        fmax = objs[front_sorted[-1]][k]
        dist[front_sorted[0]] = float('inf')
        dist[front_sorted[-1]] = float('inf')
        denom = (fmax - fmin) if abs(fmax - fmin) > 1e-12 else 1.0
        for i in range(1, len(front_sorted)-1):
            prev = objs[front_sorted[i-1]][k]
            nxt  = objs[front_sorted[i+1]][k]
            dist[front_sorted[i]] += (nxt - prev)/denom
    return dist


def tournament_select(pop: List[Dict[str, Any]],
                      evals: List[EvaluationResult],
                      k: int = 2) -> int:
    i = random.randrange(len(pop))
    for _ in range(k-1):
        j = random.randrange(len(pop))
        d = dominates(evals[i].objs, evals[i].cons, evals[j].objs, evals[j].cons)
        if d == -1 or (d == 0 and random.random() < 0.5):
            i = j
    return i


def nsga2(problem: Problem,
          pop_size: int = 64,
          n_gen: int = 20,
          seed_solutions: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], List[EvaluationResult]]:
    random.seed(42)

    space = problem.space
    pop: List[Dict[str, Any]] = []

    # Seeding: e.g., baseline GPT-2 small, your current best configs, etc.
    if seed_solutions:
        for s in seed_solutions[:pop_size]:
            pop.append(space.repair(s))
    while len(pop) < pop_size:
        pop.append(space.sample())

    evals = [problem.eval(ind) for ind in pop]

    for gen in range(n_gen):
        # Generate offspring
        offspring = []
        while len(offspring) < pop_size:
            p1 = pop[tournament_select(pop, evals)]
            p2 = pop[tournament_select(pop, evals)]
            c1, c2 = crossover(space, p1, p2)
            c1 = mutate(space, c1)
            c2 = mutate(space, c2)
            offspring.append(c1)
            if len(offspring) < pop_size:
                offspring.append(c2)

        offspring_evals = [problem.eval(ind) for ind in offspring]

        # Environmental selection
        union = pop + offspring
        union_evals = evals + offspring_evals

        fronts: List[List[int]] = []
        _ = fast_non_dominated_sort(fronts,
                                    [e.objs for e in union_evals],
                                    [e.cons for e in union_evals])

        new_pop: List[Dict[str, Any]] = []
        new_evals: List[EvaluationResult] = []
        for front in fronts:
            if len(new_pop) + len(front) <= pop_size:
                for idx in front:
                    new_pop.append(union[idx]); new_evals.append(union_evals[idx])
            else:
                cd = crowding_distance(front, [e.objs for e in union_evals])
                front_sorted = sorted(front, key=lambda i: cd[i], reverse=True)
                needed = pop_size - len(new_pop)
                for idx in front_sorted[:needed]:
                    new_pop.append(union[idx]); new_evals.append(union_evals[idx])
                break

        pop, evals = new_pop, new_evals

        # SIMPLE LOGGING: top front summary
        best_front = fronts[0]
        bf_objs = [union_evals[i].objs for i in best_front]
        print(f"[Gen {gen+1}/{n_gen}] Pareto size={len(best_front)}  "
              f"avg objs={tuple(round(sum(col)/len(col), 4) for col in zip(*bf_objs))}")

    return pop, evals


# -----------------------------
# Proxy metrics (EDIT ME)
# -----------------------------
def estimate_params(x: Dict[str, Any]) -> int:
    # Very rough proxy: params ~ 12 * d_model^2 * n_layers / mlp_ratio scaling
    d = x["d_model"]; L = x["n_layers"]; r = x["mlp_ratio"]
    base = 12 * d * d * L
    head_adj = max(1, x["n_heads"]) / 8
    return int(base * (1.0/r) * head_adj)

def estimate_mem_footprint(x: Dict[str, Any]) -> int:
    params = estimate_params(x)
    bytes_per_param = max(1, x["quant_bits"] // 8)
    kv_groups = max(1, x["n_kv_groups"])
    seq = x["seq_len"]
    # params + KV cache proxy
    return int(params*bytes_per_param + 4*seq*kv_groups*x["d_model"])

def estimate_flops(x: Dict[str, Any]) -> float:
    # FLOPs/token proxy
    d = x["d_model"]; L = x["n_layers"]; seq = x["seq_len"]
    attn_type = x["attn_type"]
    attn_cost = {"scaled_dot": d*seq, "gqa": d*seq/2, "mha_linear": d*math.log2(max(2, seq)), "flash": d*seq*0.7}
    return float(L*(2*d*d + attn_cost.get(attn_type, d*seq)))

def latency_from_tp(tp_tok_s: float) -> float:
    # turn throughput into ms/token lat proxy
    if tp_tok_s <= 1e-9: return 1e9
    return 1000.0 / tp_tok_s

def proxy_measure(x: Dict[str, Any], params: int, mem_bytes: int, flops: float) -> Tuple[float,float,float,float]:
    """
    Stand-in for your real measurement (training curve proxy, microbench, or emulator).
    Replace with calls into your pipeline (or sockets/SSH runners).
    """
    # Example monotone-ish tradeoffs to guide the search:
    scale = 1e-9
    throughput = max(1.0, 40_000_000 / (flops*scale + 1.0))
    ttft = 150.0 + 0.02*(params/1e6) + (0 if x["attn_type"] in ("flash","mha_linear") else 30.0)
    e_per_token = 0.1 + 5e-12*flops + 2e-10*mem_bytes + 0.02*(1.0 - x["act_sparsity"])
    # Make val_loss depend on capacity and regularization knobs
    capacity = math.log10(params + 10.0)
    reg = 0.2 + (x["quant_bits"] <= 8)*0.15 + x["act_sparsity"]*0.1
    val_loss = max(2.0, 5.0 - 0.35*capacity + 0.4*reg)

    return val_loss, throughput, e_per_token, ttft


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    space = SearchSpace()
    problem = Problem(space)

    # Optional: seed with your baselines
    seeds = [
        {"n_layers":12,"d_model":512,"n_heads":8,"mlp_ratio":4.0,"attn_type":"scaled_dot","n_kv_groups":1,
         "seq_len":2048,"quant_bits":8,"act_sparsity":0.2,"layer_order":["attn","mlp","ln"]},
        {"n_layers":16,"d_model":768,"n_heads":12,"mlp_ratio":3.0,"attn_type":"flash","n_kv_groups":2,
         "seq_len":4096,"quant_bits":6,"act_sparsity":0.4,"layer_order":["mlp","attn","ln"]},
    ]

    pop, evals = nsga2(problem, pop_size=64, n_gen=25, seed_solutions=seeds)

    # Extract final Pareto front
    fronts: List[List[int]] = []
    _ = fast_non_dominated_sort(fronts,
                                [e.objs for e in evals],
                                [e.cons for e in evals])
    pareto = fronts[0]
    print(f"Final Pareto size={len(pareto)}")
    for i in pareto[:10]:
        print(json.dumps({"x": pop[i], "objs": evals[i].objs, "cons": evals[i].cons, "aux": evals[i].aux}, indent=2))

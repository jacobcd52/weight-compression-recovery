"""MAP-Elites / Pareto-grid archive for the compression-method search.

Objectives (both minimized): compression ratio (bytes) and recovery cost (retrain fraction).
Cells are keyed by (family, ratio-bin) to preserve diversity; each cell keeps the best entry.
Parent sampling favours the Pareto front plus a diverse pick, and shorter programs on ties.
Persisted as JSON (programs stored as text) so a run survives restarts.
"""
import json
import math
import os


class Archive:
    def __init__(self, path, budget_fraction=0.05):
        self.path = path
        self.budget_fraction = budget_fraction
        self.entries = []          # list of dicts (see add())

    # ---- io ----
    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"budget_fraction": self.budget_fraction, "entries": self.entries}, f, indent=2)
        os.replace(tmp, self.path)

    def load(self):
        if os.path.exists(self.path):
            d = json.load(open(self.path))
            self.budget_fraction = d.get("budget_fraction", self.budget_fraction)
            self.entries = d["entries"]
        return self

    # ---- core ----
    def _ratio_bin(self, ratio):
        # log-spaced bins (0.001 .. 1.0) -> integer bin index
        return int(round(2 * math.log10(max(ratio, 1e-4))))

    def add(self, rec):
        """rec: {name, code, family, ratio, recovered, recovery_fraction, best_lr, gen, parents}.
        recovery_fraction for DNR = budget_fraction (the cap)."""
        rec = dict(rec)
        rec["cell"] = (rec["family"], self._ratio_bin(rec["ratio"]))
        self.entries.append(rec)
        return rec

    def _dominates(self, a, b):
        # a dominates b if no worse on both objectives and strictly better on one
        ra, rb = a["ratio"], b["ratio"]
        ca, cb = a["recovery_fraction"], b["recovery_fraction"]
        return (ra <= rb and ca <= cb) and (ra < rb or ca < cb)

    def pareto_front(self):
        rec = [e for e in self.entries if e.get("recovered")]
        front = []
        for e in rec:
            if not any(self._dominates(o, e) for o in rec if o is not e):
                front.append(e)
        # de-dup by (ratio, recovery_fraction), prefer shorter code
        front.sort(key=lambda e: (e["ratio"], e["recovery_fraction"], len(e["code"])))
        return front

    def best_per_cell(self):
        cells = {}
        for e in self.entries:
            key = tuple(e["cell"])
            cur = cells.get(key)
            # prefer recovered, then lower recovery_fraction, then lower ratio, then shorter code
            score = (0 if e.get("recovered") else 1, e["recovery_fraction"], e["ratio"], len(e["code"]))
            if cur is None or score < cur[0]:
                cells[key] = (score, e)
        return [v[1] for v in cells.values()]

    def sample_parents(self, k=3, rng=None):
        import random
        rng = rng or random
        pool = self.best_per_cell()
        front = self.pareto_front()
        picks = []
        if front:
            picks.append(rng.choice(front))                # exploit a frontier point
        # diverse inspirations from distinct cells
        rng.shuffle(pool)
        for e in pool:
            if len(picks) >= k:
                break
            if e not in picks:
                picks.append(e)
        return picks

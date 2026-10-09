#!/usr/bin/env python3
"""Unified Betting Edge Finder: NBA + NFL + Premier League.

Layers: (1) pattern analysis, (2) logistic-regression ML, (3) walk-forward strategy testing.
Every prediction for a game uses only games with an earlier date (no look-ahead).
Prices are ASSUMED: the data holds no-vig consensus probabilities, so offered odds
= 1 / (fair_prob * (1 + margin)). See CFG["margin"].

Usage:  python edge_finder.py run --data DIR --out DIR [--upcoming FILE ...]
        python edge_finder.py selftest --data DIR
"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

CFG = dict(
    margin={"SPREAD": .045, "TOTAL": .045, "AH": .045, "OU25": .045, "BTTS": .045, "ML": .04, "1X2": .05},
    edge_thr=0.02, kelly_frac=0.25, kelly_cap=0.03, bankroll0=100.0,
    min_train={"nba": 400, "nfl": 100, "epl": 100}, refit_days=7,
    shrink_k=40, min_seg_n=15, ml_C=0.1, seed=7)
LINEM = ("SPREAD", "TOTAL", "AH")
FILES = {"nba": "nba_2025_26_closing_lines_ats.csv", "nfl": "nfl_2025_closing_lines_ats.csv",
         "epl": "premier_league_2025_26_closing_odds.csv"}
SIDES = {"SPREAD": ("home", "away"), "AH": ("home", "away"), "ML": ("home", "away"),
         "TOTAL": ("over", "under"), "OU25": ("over", "under"), "BTTS": ("yes", "no")}


# ---------------------------------------------------------------- settlement helpers
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def line_res(diff, line):
    """Result r in [-1,1] for a side with margin `diff` and handicap `line`. Quarter lines split in two."""
    diff = np.asarray(diff, float); line = np.asarray(line, float)
    quarter = np.abs(line * 2 - np.round(line * 2)) > 1e-9
    r = np.where(quarter, 0.5 * (np.sign(diff + line + .25) + np.sign(diff + line - .25)), np.sign(diff + line))
    return np.where(np.isnan(diff) | np.isnan(line), np.nan, r)


def _c(d, n):
    return d[n] if n in d else pd.Series(np.nan, index=d.index)


# ---------------------------------------------------------------- loading
def detect(d):
    return "epl" if "match_id" in d else ("nfl" if "week" in d else "nba")


def build_games(sport, d):
    kc = "kickoff_utc" if sport == "epl" else "start_time_utc"
    d = d.assign(_k=pd.to_datetime(d[kc], utc=True)).sort_values("_k", kind="stable").drop(columns="_k").reset_index(drop=True)
    if sport == "epl":
        g = pd.DataFrame({"sport": sport, "gid": d.match_id, "kick": pd.to_datetime(d.kickoff_utc, utc=True),
                          "date": pd.to_datetime(d.date_uk), "home": d.home_team, "away": d.away_team,
                          "hs": _c(d, "home_goals"), "as_": _c(d, "away_goals"),
                          "ats_line": _c(d, "close_ah_line_home"), "tot_line": 2.5})
    else:
        g = pd.DataFrame({"sport": sport, "gid": d.game_id, "kick": pd.to_datetime(d.start_time_utc, utc=True),
                          "date": pd.to_datetime(d.date_et), "home": d.home_team, "away": d.away_team,
                          "hs": _c(d, "home_score"), "as_": _c(d, "away_score"),
                          "ats_line": _c(d, "close_spread_home"), "tot_line": _c(d, "close_total")})
    return g, d


def build_frames(sport, g, d):
    """One row per (game, market, selection) with prices and settled results at open and close."""
    diff = (g.hs - g.as_).to_numpy(float); tot = (g.hs + g.as_).to_numpy(float)
    base = g[["sport", "gid", "date", "kick", "home", "away"]]
    out = []

    def mk(market, sel, lo, lc, po, pc, ro, rc, two=True):
        f = base.copy()
        f["market"], f["sel"], f["two_way"] = market, sel, two
        f["line_o"], f["line_c"], f["pref_o"], f["pref_c"], f["r_o"], f["r_c"] = lo, lc, po, pc, ro, rc
        out.append(f)

    nan = np.nan
    if sport == "epl":
        lo, lc = _c(d, "open_ah_line_home").to_numpy(float), _c(d, "close_ah_line_home").to_numpy(float)
        mk("AH", "home", lo, lc, .5, _c(d, "close_ah_home_prob"), line_res(diff, lo), line_res(diff, lc))
        r_ou = np.where(np.isnan(tot), nan, np.sign(tot - 2.5))
        mk("OU25", "over", 2.5, 2.5, _c(d, "open_over25_prob"), _c(d, "close_over25_prob"), r_ou, r_ou)
        both = (g.hs > 0) & (g.as_ > 0)
        r_bt = np.where(np.isnan(tot), nan, np.where(both, 1.0, -1.0))
        mk("BTTS", "yes", 2.5, 2.5, _c(d, "open_btts_prob"), _c(d, "close_btts_prob"), r_bt, r_bt)
        for sel, col, win in (("home", "home", diff > 0), ("draw", "draw", diff == 0), ("away", "away", diff < 0)):
            r = np.where(np.isnan(diff), nan, np.where(win, 1.0, -1.0))
            mk("1X2", sel, nan, nan, _c(d, f"open_{col}_prob"), _c(d, f"close_{col}_prob"), r, r, two=False)
    else:
        lo, lc = _c(d, "open_spread_home").to_numpy(float), _c(d, "close_spread_home").to_numpy(float)
        mk("SPREAD", "home", lo, lc, .5, _c(d, "close_home_cover_prob"), line_res(diff, lo), line_res(diff, lc))
        to, tc = _c(d, "open_total").to_numpy(float), _c(d, "close_total").to_numpy(float)
        mk("TOTAL", "over", to, tc, .5, _c(d, "close_over_prob"), line_res(tot, -to), line_res(tot, -tc))
        r_ml = np.sign(diff)
        mk("ML", "home", nan, nan, _c(d, "open_home_win_prob"), _c(d, "close_home_win_prob"), r_ml, r_ml)
    F = pd.concat(out, ignore_index=True)
    for c in ("pref_o", "pref_c"):
        F[c] = pd.to_numeric(F[c], errors="coerce")
    return F


# ---------------------------------------------------------------- pre-game features (strictly prior dates)
def build_feats(g, sport):
    K = dict(nba=(7, 1), nfl=(14, 5), epl=(10, 3))[sport]
    st, lg_tot, rows = {}, [], {}
    shr = lambda v, k: (sum(v) / (len(v) + k)) if v else 0.0

    def tf(team, d, lgavg):
        s = st.get(team)
        if not s:
            return dict(ats=0, ats5=0, marg=0, tot=0, pace=0, rest=K[0] / 2, short=0, gp=0)
        rest = min((d - s["last"]).days, K[0])
        pace = (sum(s["tp"]) - lgavg * len(s["tp"])) / (len(s["tp"]) + 10) if lgavg == lgavg else 0
        return dict(ats=shr(s["ats"], 10), ats5=sum(s["ats"][-5:]) / (len(s["ats"][-5:]) + 3), marg=shr(s["pm"], 10),
                    tot=shr(s["tot"], 10), pace=pace, rest=rest, short=float(rest <= K[1]), gp=len(s["ats"]))

    for d, grp in g.groupby("date", sort=True):
        lgavg = float(np.mean(lg_tot)) if lg_tot else np.nan
        for i, r in grp.iterrows():
            h, a = tf(r.home, d, lgavg), tf(r.away, d, lgavg)
            rows[i] = dict(gp=min(h["gp"], a["gp"]), ats_d=h["ats"] - a["ats"], ats5_d=h["ats5"] - a["ats5"],
                           marg_d=h["marg"] - a["marg"], tot_s=h["tot"] + a["tot"], pace_s=h["pace"] + a["pace"],
                           rest_d=h["rest"] - a["rest"], short_h=h["short"], short_a=a["short"])
        for i, r in grp.iterrows():          # update state AFTER the whole day is featurised
            if np.isnan(r.hs) or np.isnan(r.as_):
                continue
            dd, tt = r.hs - r.as_, r.hs + r.as_
            ra = dd + r.ats_line if r.ats_line == r.ats_line else 0.0
            rt = tt - r.tot_line if r.tot_line == r.tot_line else 0.0
            for team, sgn, pf, pa in ((r.home, 1, r.hs, r.as_), (r.away, -1, r.as_, r.hs)):
                s = st.setdefault(team, dict(ats=[], tp=[], pm=[], tot=[], last=d))
                s["ats"].append(sgn * ra); s["tp"].append(tt); s["pm"].append(pf - pa); s["tot"].append(rt); s["last"] = d
            lg_tot.append(tt)
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


# ---------------------------------------------------------------- pattern segments
EDGES = {("nba", "SPREAD"): [2, 5, 8, 11], ("nfl", "SPREAD"): [1.5, 3.5, 6.5, 10], ("epl", "AH"): [.25, .75, 1.25],
         ("nba", "TOTAL"): [216, 222, 228, 234], ("nfl", "TOTAL"): [41, 44, 47, 50]}
PEDGES = [.2, .35, .5, .65, .8]


def segs(r, mode):
    m, o = r.market, []
    line = r.line_o if mode == "open" else r.line_c
    pref = r.pref_o if mode == "open" else r.pref_c
    pmv = int(np.sign(round(r.pref_c - r.pref_o, 2))) if abs(r.pref_c - r.pref_o) > .02 else 0
    if m in ("SPREAD", "AH"):
        b = int(np.digitize(abs(line), EDGES[(r.sport, m)])); fav = "H" if line < 0 else ("A" if line > 0 else "P")
        o = [(f"line:{fav}{b}", 1), (f"tm:{r.home}", 1), (f"tm:{r.away}", -1), (f"rest:{int(r.short_h)}{int(r.short_a)}", 1)]
        if mode == "close": o.append((f"mv:{int(np.sign(r.line_c - r.line_o))}", 1))
    elif m == "TOTAL":
        o = [(f"tot:{int(np.digitize(line, EDGES[(r.sport, m)]))}", 1), (f"tm:{r.home}", 1), (f"tm:{r.away}", 1)]
        if mode == "close": o.append((f"mv:{int(np.sign(r.line_c - r.line_o))}", 1))
    elif m == "ML":
        o = [(f"p:{int(np.digitize(pref, PEDGES))}", 1), (f"tm:{r.home}", 1), (f"tm:{r.away}", -1),
             (f"rest:{int(r.short_h)}{int(r.short_a)}", 1)]
        if mode == "close": o.append((f"mv:{pmv}", 1))
    elif m in ("OU25", "BTTS"):
        o = [(f"p:{int(np.digitize(pref, PEDGES))}", 1), (f"tm:{r.home}", 1), (f"tm:{r.away}", 1)]
        if mode == "close": o.append((f"mv:{pmv}", 1))
    elif m == "1X2":
        o = [(f"p:{int(np.digitize(pref, PEDGES))}", 1), (f"H:{r.home}", 1), (f"A:{r.away}", 1)]
        if mode == "close": o.append((f"mv:{pmv}", 1))
    return o


FC = {"SPREAD": ["line", "lp", "ats_d", "ats5_d", "marg_d", "rest_d", "short_h", "short_a"],
      "AH": ["line", "lp", "ats_d", "ats5_d", "marg_d", "rest_d", "short_h", "short_a"],
      "TOTAL": ["line", "lp", "tot_s", "pace_s", "rest_d"],
      "ML": ["lp", "ats_d", "marg_d", "rest_d", "short_h", "short_a"],
      "OU25": ["lp", "tot_s", "pace_s"], "BTTS": ["lp", "tot_s", "pace_s"], "1X2": ["lp", "marg_d", "ats_d", "rest_d"]}


def feature_cols(mk, mode):
    c = list(FC[mk])
    if mode == "close":
        c.append("move" if mk in LINEM else "lmove")
    return c


def fit_predict(Xtr, ytr, Xte):
    if len(ytr) < 60 or ytr.min() == ytr.max():
        return np.full(len(Xte), ytr.mean() if len(ytr) else .5)
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(C=CFG["ml_C"], max_iter=500).fit(sc.transform(Xtr), ytr)
    return m.predict_proba(sc.transform(Xte))[:, 1]


# ---------------------------------------------------------------- walk-forward predictions
def walk_forward(F, mode, t0):
    F = F.copy()
    o = mode == "open"
    F["line"] = F.line_o if o else F.line_c
    F["pref"] = F.pref_o if o else F.pref_c
    F["r"] = F.r_o if o else F.r_c
    F["y"] = np.where(F.r > 0, 1.0, np.where(F.r < 0, 0.0, np.nan))
    F["lp"] = logit(F.pref)
    F["move"] = 0.0 if o else (F.line_c - F.line_o)
    F["lmove"] = 0.0 if o else (logit(F.pref_c) - logit(F.pref_o))
    F = F[F.pref.notna() & ~(F.market.isin(LINEM) & F.line.isna())]
    if not o:
        F = F[F.pref_o.notna()]
    Fs = F.sort_values(["date", "kick"]).reset_index(drop=True)
    Fs["blk"] = np.where(Fs.date >= t0, (Fs.date - t0).dt.days // CFG["refit_days"], -1)
    dates, y, pref = Fs.date.to_numpy(), Fs.y.to_numpy(), Fs.pref.to_numpy()
    grp = (Fs.market + "|" + Fs.sel).to_numpy()
    sl = [segs(r, mode) for r in Fs.itertuples()]
    n = len(Fs)
    p_pat, p_ml, pat_n, pat_res = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n), np.zeros(n)
    pat_key = np.array([""] * n, dtype=object)
    stt, ptr = {}, 0
    blocks = sorted(Fs.blk[Fs.blk >= 0].unique())
    Xs = {}
    for g_ in set(grp):
        mk = g_.split("|")[0]
        Xs[g_] = Fs[feature_cols(mk, mode)].fillna(0).to_numpy(float)
    for b in blocks:
        bstart = (t0 + pd.Timedelta(days=int(b) * CFG["refit_days"])).to_datetime64()
        while ptr < n and dates[ptr] < bstart:                 # add newly-resolved rows to pattern stats
            if not np.isnan(y[ptr]):
                for k, s in sl[ptr]:
                    yt, pt = (y[ptr], pref[ptr]) if s == 1 else (1 - y[ptr], 1 - pref[ptr])
                    c0 = stt.get((grp[ptr], k), (0, 0.0)); stt[(grp[ptr], k)] = (c0[0] + 1, c0[1] + yt - pt)
            ptr += 1
        test = np.where(Fs.blk.to_numpy() == b)[0]
        for i in test:                                          # pattern prediction
            best = None
            for k, s in sl[i]:
                cn, cs = stt.get((grp[i], k), (0, 0.0))
                if cn >= CFG["min_seg_n"]:
                    rs = cs / (cn + CFG["shrink_k"])
                    if best is None or abs(rs) > abs(best[0]): best = (rs, s, k, cn)
            if best:
                p_pat[i] = np.clip(pref[i] + best[1] * best[0], .01, .99); pat_key[i] = best[2]; pat_n[i] = best[3]; pat_res[i] = best[0] * best[1]
            else:
                p_pat[i] = pref[i]
        for g_ in set(grp[test]):                               # ML prediction
            idx = np.where(grp == g_)[0]
            tr = idx[(dates[idx] < bstart) & ~np.isnan(y[idx])]
            te = test[grp[test] == g_]
            p_ml[te] = fit_predict(Xs[g_][tr], y[tr], Xs[g_][te])
    Fs["p_pat"], Fs["p_ml"], Fs["pat_key"], Fs["pat_n"], Fs["pat_res"] = p_pat, p_ml, pat_key, pat_n, pat_res
    Fs = Fs[Fs.blk >= 0].copy()
    m3 = Fs.market == "1X2"                                      # normalise 3-way probabilities
    for c in ("p_pat", "p_ml"):
        s = Fs[m3].groupby("gid")[c].transform("sum")
        Fs.loc[m3, c] = Fs.loc[m3, c] / s
    Fs["mode"] = mode
    return Fs


# ---------------------------------------------------------------- selections (pricing, CLV, settlement)
def build_sel(P, mscale=1.0):
    out = []
    for two in (True, False):
        Q = P[P.two_way == two]
        if Q.empty: continue
        sides = [0, 1] if two else [0]
        for s in sides:
            a = Q.copy()
            marg = a.market.map(CFG["margin"]) * mscale
            if s == 0:
                a["side"] = a.sel if not two else a.market.map(lambda m: SIDES[m][0])
                a["p_ref"], a["p_pat"], a["p_ml"], a["r"] = a.pref, a.p_pat, a.p_ml, a.r
                ln = a.line
            else:
                a["side"] = a.market.map(lambda m: SIDES[m][1])
                a["p_ref"], a["p_pat"], a["p_ml"], a["r"] = 1 - a.pref, 1 - a.p_pat, 1 - a.p_ml, -a.r
                ln = np.where(a.market.isin(["SPREAD", "AH"]), -a.line, a.line)
            a["line_side"] = ln
            a["odds"] = 1 / (a.p_ref * (1 + marg))
            a["profit_unit"] = np.where(a.r > 0, a.r * (a.odds - 1), a.r)
            # closing-line value (only meaningful for open-mode bets)
            if Q["mode"].iloc[0] == "open":
                dl = a.line_c - a.line_o
                pts = np.where(a.market.isin(["SPREAD", "AH"]), (-dl if s == 0 else dl), (dl if s == 0 else -dl))
                pr = (a.pref_c if s == 0 else 1 - a.pref_c) / (a.pref_o if s == 0 else 1 - a.pref_o) - 1
                prob_m = a.market.isin(["ML", "OU25", "BTTS", "1X2"])
                a["clv"] = np.where(prob_m, pr, pts)
                a["clv_unit"] = np.where(prob_m, "prob%", "points")
                tilt = (a.pref_c if s == 0 else 1 - a.pref_c) - .5
                a["clv_beat"] = np.where(prob_m, pr > 0, (pts > 0) | ((pts == 0) & (tilt > 0)))
            else:
                a["clv"], a["clv_unit"], a["clv_beat"] = np.nan, "", np.nan
            out.append(a)
    S = pd.concat(out, ignore_index=True)
    keep = ["sport", "market", "mode", "gid", "date", "kick", "home", "away", "side", "line_side", "p_ref", "odds",
            "p_pat", "p_ml", "pat_key", "pat_n", "r", "profit_unit", "clv", "clv_unit", "clv_beat"]
    return S[keep]


def pick(S, approach, thr):
    s = S[S.p_pat.notna() & S.p_ml.notna()].copy()
    be = 1 / s.odds
    if approach == "PATTERN": s["p"] = s.p_pat
    elif approach == "ML": s["p"] = s.p_ml
    else:
        s = s[(s.p_pat > be) & (s.p_ml > be)]; s["p"] = (s.p_pat + s.p_ml) / 2
    s["edge"] = s.p - 1 / s.odds
    s = s[s.edge >= thr]
    return s.sort_values("edge", ascending=False).drop_duplicates(["gid", "market"]).sort_values(["date", "kick"])


def stake(b, method):
    b = b.sort_values(["date", "kick"]).copy()
    bank0 = CFG["bankroll0"]
    if method == "FLAT":
        b["stake"] = 1.0
    else:
        bank, st_, cur, day_bank = bank0, [], None, bank0
        pnl_day = 0.0
        for r in b.itertuples():
            if r.date != cur:
                bank += pnl_day; pnl_day = 0.0; cur = r.date; day_bank = bank
            f = min(CFG["kelly_cap"], max(0.0, CFG["kelly_frac"] * (r.p * r.odds - 1) / (r.odds - 1)))
            s_ = f * day_bank; st_.append(s_)
            pnl_day += s_ * (r.profit_unit if r.profit_unit == r.profit_unit else 0.0)
        b["stake"] = st_
    b["pnl"] = b.stake * b.profit_unit.fillna(0)
    b["bank"] = bank0 + b.pnl.cumsum()
    return b


def summarise(b, label):
    s = b[b.r.notna()]
    if s.empty:
        return dict(**label, bets=0)
    pk = np.maximum.accumulate(np.r_[CFG["bankroll0"], s.bank.to_numpy()])
    flat = bool((s.stake == 1).all())
    dd = (pk[1:] - s.bank.to_numpy()).max() if flat else 100 * ((pk[1:] - s.bank.to_numpy()) / pk[1:]).max()
    pu = s.profit_unit.to_numpy()
    res = dict(**label, bets=len(s), staked=s.stake.sum(), profit=s.pnl.sum(),
               roi_pct=100 * s.pnl.sum() / s.stake.sum() if s.stake.sum() else np.nan,
               hit_pct=100 * (s.r > 0).sum() / max((s.r != 0).sum(), 1), avg_edge_pct=100 * s.edge.mean(),
               final_bank=s.bank.iloc[-1], max_drawdown=dd, dd_unit="units" if flat else "% of bankroll",
               clv_beat_pct=100 * s.clv_beat.astype(float).mean() if s.clv_beat.notna().any() else np.nan)
    if len(s) >= 10 and (s.stake == 1).all():
        rng = np.random.default_rng(CFG["seed"])
        bs = rng.choice(pu, (2000, len(pu))).mean(1) * 100
        res.update(roi_lo90=np.percentile(bs, 5), roi_hi90=np.percentile(bs, 95),
                   t_stat=pu.mean() / (pu.std(ddof=1) / np.sqrt(len(pu))) if pu.std() > 0 else np.nan)
    return res


# ---------------------------------------------------------------- exploratory pattern scan (full sample)
def pattern_scan(F, mode, t0):
    o = mode == "open"
    rows = []
    for r in F.itertuples():
        pref, rr = (r.pref_o, r.r_o) if o else (r.pref_c, r.r_c)
        if np.isnan(pref) or np.isnan(rr) or rr == 0: continue
        if r.market in LINEM and np.isnan(r.line_o if o else r.line_c): continue
        if not o and np.isnan(r.pref_o): continue
        for k, s in segs(r, mode):
            y = 1.0 if rr > 0 else 0.0
            yt, pt, rt = (y, pref, rr) if s == 1 else (1 - y, 1 - pref, -rr)
            odds = 1 / (pt * (1 + CFG["margin"][r.market]))
            rows.append((r.sport, r.market, r.sel, mode, k, yt, pt, rt * (odds - 1) if rt > 0 else rt))
    D = pd.DataFrame(rows, columns=["sport", "market", "sel", "mode", "segment", "yt", "pt", "pu"])
    A = D.groupby(["sport", "market", "sel", "mode", "segment"]).agg(n=("yt", "size"), win_rate=("yt", "mean"),
        expected=("pt", "mean"), var=("pt", lambda x: (x * (1 - x)).sum()), roi_pct=("pu", lambda x: 100 * x.mean())).reset_index()
    A = A[A.n >= 20].copy()
    A["z"] = (A.win_rate - A.expected) * A.n / np.sqrt(A["var"])
    A["p_value"] = 2 * (1 - stats.norm.cdf(np.abs(A.z)))
    A["q_bh"] = np.nan
    srt = A.sort_values("p_value"); q = (srt.p_value.values * len(srt) / np.arange(1, len(srt) + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]; A.loc[srt.index, "q_bh"] = np.minimum(q, 1)
    return A.drop(columns="var").sort_values("p_value")


# ---------------------------------------------------------------- orchestration
def predict_sport(sport, d):
    g, d = build_games(sport, d)
    F = build_frames(sport, g, d)
    feats = build_feats(g, sport)
    F = F.merge(feats.assign(gid=g.gid.values), on="gid", how="left")
    res = g[g.hs.notna()].sort_values("date")
    t0 = res.date.iloc[CFG["min_train"][sport]] if len(res) > CFG["min_train"][sport] else res.date.iloc[-1]
    P = pd.concat([walk_forward(F, m, t0) for m in ("open", "close")], ignore_index=True)
    return F, P, t0, feats, g


def load_all(data, upcoming=()):
    raw = {}
    for sp, fn in FILES.items():
        raw[sp] = pd.read_csv(Path(data) / fn)
    for u in upcoming:
        d = pd.read_csv(u); sp = detect(d); idc = "match_id" if sp == "epl" else "game_id"
        raw[sp] = pd.concat([raw[sp], d[~d[idc].isin(raw[sp][idc])]], ignore_index=True)
    return raw


def run(data, out, upcoming=()):
    out = Path(out); (out / "results").mkdir(parents=True, exist_ok=True)
    raw = load_all(data, upcoming)
    Fall, Pall, scans = {}, [], []
    for sp, d in raw.items():
        print(f"[{sp}] {len(d)} rows -> walk-forward ...", flush=True)
        F, P, t0, feats, g = predict_sport(sp, d)
        print(f"   first test date {t0.date()}, predictions: {len(P)}")
        Pall.append(P); Fall[sp] = (F, t0)
        for m in ("open", "close"):
            scans.append(pattern_scan(F[F.date.notna()], m, t0))
    P = pd.concat(Pall, ignore_index=True)
    pd.concat(scans).to_csv(out / "results/pattern_scan_fullsample.csv", index=False)
    summ, summ0, bets_all, base_rows = [], [], [], []
    for scale, tag, store in ((1.0, "assumed_margin", summ), (0.0, "zero_margin", summ0)):
        for mode in ("open", "close"):
            S = build_sel(P[P["mode"] == mode], scale)
            for ap in ("PATTERN", "ML", "BOTH"):
                pk = pick(S, ap, CFG["edge_thr"] if scale else CFG["edge_thr"])
                for stk in ("FLAT", "KELLY"):
                    for sp in list(FILES) + ["ALL"]:
                        sub = pk if sp == "ALL" else pk[pk.sport == sp]
                        if sub.empty: continue
                        b = stake(sub, stk)
                        store.append(summarise(b, dict(pricing=tag, sport=sp, mode=mode, approach=ap, staking=stk)))
                        if scale and sp == "ALL":
                            bets_all.append(b.assign(approach=ap, staking=stk))
            if scale:   # naive baselines (flat, every game) to show margin drag
                for nm, f in (("ALL_HOME/OVER/YES", lambda s: s[s.side.isin(["home", "over", "yes"])]),
                              ("ALL_AWAY/UNDER/NO", lambda s: s[s.side.isin(["away", "under", "no"])])):
                    sub = f(S[S.r.notna()]); sub = sub.assign(p=sub.p_ref, edge=0.0)
                    for sp in list(FILES):
                        for mk in sub[sub.sport == sp].market.unique():
                            q = sub[(sub.sport == sp) & (sub.market == mk)]
                            base_rows.append(summarise(stake(q, "FLAT"), dict(strategy=nm, sport=sp, market=mk, mode=mode)))
    ll = []
    for (sp, mk, sel, mode), q in P[P.r.notna() & (P.r != 0)].groupby(["sport", "market", "sel", "mode"]):
        y = (q.r > 0).astype(float).to_numpy(); e = 1e-4
        L = lambda p: -np.mean(y * np.log(np.clip(p, e, 1 - e)) + (1 - y) * np.log(np.clip(1 - p, e, 1 - e)))
        ll.append(dict(sport=sp, market=mk, sel=sel, mode=mode, n=len(q), logloss_market=L(q.pref.to_numpy()),
                       logloss_ml=L(q.p_ml.to_numpy()), logloss_pattern=L(q.p_pat.to_numpy()),
                       ml_beats_market=L(q.p_ml.to_numpy()) < L(q.pref.to_numpy())))
    pd.DataFrame(ll).to_csv(out / "results/model_vs_market_logloss.csv", index=False)
    pd.DataFrame(summ).to_csv(out / "results/summary_assumed_margin.csv", index=False)
    pd.DataFrame(summ0).to_csv(out / "results/summary_zero_margin.csv", index=False)
    pd.DataFrame(base_rows).to_csv(out / "results/baselines_naive.csv", index=False)
    B = pd.concat(bets_all, ignore_index=True); B.to_csv(out / "results/all_bets.csv", index=False)
    # by-market and CLV tables (flat staking, open mode CLV)
    byM = []
    for ap in ("PATTERN", "ML", "BOTH"):
        S = build_sel(P, 1.0)
        pk = pick(S, ap, CFG["edge_thr"])
        for (sp, mk, mode), sub in pk.groupby(["sport", "market", "mode"]):
            byM.append(summarise(stake(sub, "FLAT"), dict(sport=sp, market=mk, mode=mode, approach=ap)))
    pd.DataFrame(byM).to_csv(out / "results/summary_by_market.csv", index=False)
    clv = B[(B["mode"] == "open") & (B.staking == "FLAT") & B.r.notna()].groupby(["sport", "market", "approach", "clv_unit"]).agg(
        bets=("clv", "size"), avg_clv=("clv", "mean"), beat_close_pct=("clv_beat", lambda x: 100 * x.astype(float).mean())).reset_index()
    clv.to_csv(out / "results/clv_by_market.csv", index=False)
    # threshold sensitivity (flat, ALL) - reported, NOT used to choose the default
    sens = []
    for mode in ("open", "close"):
        S = build_sel(P[P["mode"] == mode], 1.0)
        for ap in ("PATTERN", "ML", "BOTH"):
            for thr in (0.0, 0.02, 0.04, 0.06):
                sub = pick(S, ap, thr)
                if len(sub): sens.append(summarise(stake(sub, "FLAT"), dict(mode=mode, approach=ap, threshold=thr)))
    pd.DataFrame(sens).to_csv(out / "results/threshold_sensitivity.csv", index=False)
    # paper picks: unresolved games (real upcoming) + latest settled slate as pseudo-live
    picks = []
    for mode in ("open", "close"):
        S = build_sel(P[P["mode"] == mode], 1.0)
        for ap in ("PATTERN", "ML", "BOTH"):
            pk = pick(S, ap, CFG["edge_thr"])
            if pk.empty: continue
            pk = pk.assign(approach=ap, kelly_stake_pct=100 * np.minimum(CFG["kelly_cap"], np.maximum(0, CFG["kelly_frac"] * (pk.p * pk.odds - 1) / (pk.odds - 1))))
            for sp, sub in pk.groupby("sport"):
                last = P[(P.sport == sp) & P.r.notna()].date.max()
                up = sub[sub.r.isna()]
                lt = sub[(sub.date == last)]
                picks += [up.assign(slate="UPCOMING"), lt.assign(slate="LATEST_SETTLED_SLATE (pseudo-live)")]
    cols = ["slate", "sport", "market", "mode", "approach", "date", "home", "away", "side", "line_side", "odds", "p_ref", "p", "edge",
            "pat_key", "pat_n", "kelly_stake_pct", "r", "profit_unit"]
    if picks:
        pd.concat(picks)[cols].to_csv(out / "results/paper_picks.csv", index=False)
    equity_plot(B, out / "equity_curves.png")
    print("done ->", out)
    return P


def equity_plot(B, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(18, 7), sharey=False)
    for i, mode in enumerate(("open", "close")):
        for j, sp in enumerate(["nba", "nfl", "epl", "ALL"]):
            a = ax[i, j]
            for ap, col in (("PATTERN", "tab:orange"), ("ML", "tab:blue"), ("BOTH", "tab:green")):
                s = B[(B["mode"] == mode) & (B.approach == ap) & (B.staking == "FLAT") & B.r.notna()]
                if sp != "ALL": s = s[s.sport == sp]
                if s.empty: continue
                s = s.sort_values(["date", "kick"]); a.plot(s.date, s.pnl.cumsum().values, label=f"{ap} ({len(s)})", color=col)
            a.axhline(0, color="k", lw=.6); a.set_title(f"{sp.upper()} - bet at {mode}"); a.tick_params(axis="x", rotation=30)
            if j == 0: a.set_ylabel("cumulative units (flat 1u)")
            a.legend(fontsize=7)
    fig.suptitle("Walk-forward flat-stake equity (assumed margin, edge >= 2%)"); fig.tight_layout(); fig.savefig(path, dpi=110)


# ---------------------------------------------------------------- self tests
def selftest(data):
    raw = load_all(data)
    ok = True
    # 1. settlement vs file labels
    d = raw["epl"]; g, _ = build_games("epl", d); diff = (d.home_goals - d.away_goals).to_numpy(float)
    for col, line in (("ah_result", "close_ah_line_home"), ("ah_result_open", "open_ah_line_home")):
        r = line_res(diff, d[line].to_numpy(float))
        lab = np.where(r > 0, "home", np.where(r < 0, "away", "push"))
        file = d[col].replace({"away-half": "away"}).to_numpy()
        mism = (lab != file).sum(); print(f"AH settle vs {col}: mismatches {mism} (half-win/half-loss lines may differ in label)")
    for sp in ("nba", "nfl"):
        d = raw[sp]; r = line_res((d.home_score - d.away_score).to_numpy(float), d.close_spread_home.to_numpy(float))
        lab = np.where(r > 0, "home", np.where(r < 0, "away", "push")); print(sp, "ATS mismatches:", (lab != d.ats_result.to_numpy()).sum())
        rt = line_res((d.home_score + d.away_score).to_numpy(float), -d.close_total.to_numpy(float))
        lab = np.where(rt > 0, "over", np.where(rt < 0, "under", "push")); print(sp, "total mismatches:", (lab != d.total_result.to_numpy()).sum())
    # 2. leakage: perturb the last 30% of results; predictions before the cut must not change
    for sp in ("nba", "epl"):
        d = raw[sp].copy(); dc = "date_et" if sp != "epl" else "date_uk"
        cut = d[dc].sort_values().iloc[int(len(d) * .7)]
        F1, P1, *_ = predict_sport(sp, d)
        d2 = d.copy(); m = d2[dc] >= cut; rng = np.random.default_rng(1)
        hc, ac = ("home_score", "away_score") if sp != "epl" else ("home_goals", "away_goals")
        d2.loc[m, hc] = rng.integers(0, 120 if sp != "epl" else 5, m.sum()); d2.loc[m, ac] = rng.integers(0, 120 if sp != "epl" else 5, m.sum())
        F2, P2, *_ = predict_sport(sp, d2)
        key = ["gid", "market", "sel", "mode"]
        A = P1[P1.date < pd.Timestamp(cut)].merge(P2[P2.date < pd.Timestamp(cut)], on=key, suffixes=("_1", "_2"))
        dm = max(np.nanmax(np.abs(A.p_ml_1 - A.p_ml_2)), np.nanmax(np.abs(A.p_pat_1 - A.p_pat_2)))
        print(f"{sp}: leakage test on {len(A)} predictions, max |diff| = {dm:.2e} ->", "PASS" if dm < 1e-9 else "FAIL"); ok &= dm < 1e-9
    # 3. null test: simulate outcomes from the market's own fair probabilities -> ROI should be about -margin
    F, P, *_ = predict_sport("nba", raw["nba"])
    rng = np.random.default_rng(3); P = P.copy()
    P["r"] = np.where(rng.random(len(P)) < P.pref, 1.0, -1.0)
    S = build_sel(P[P["mode"] == "close"]); pk = pick(S, "ML", 0.0)
    print("null (outcomes simulated from fair probs) flat ROI %:", round(100 * pk.profit_unit.mean(), 2),
          "bets", len(pk), "(expect roughly -4%, noise +/- ~3)")
    print("SELFTEST", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "selftest"])
    ap.add_argument("--data", default=".")
    ap.add_argument("--out", default="out")
    ap.add_argument("--upcoming", nargs="*", default=[])
    a = ap.parse_args()
    run(a.data, a.out, a.upcoming) if a.cmd == "run" else selftest(a.data)

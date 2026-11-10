"""Descarga 5 temporadas de football-data.co.uk (nuestra 'base de datos' en CSV), calcula fuerzas de equipos,
ratings Elo, un modelo estadístico (Poisson) y un modelo de IA (gradient boosting), los evalúa con backtesting
y prepara los próximos partidos.  Genera data/datos.json y data/partidos.json.
Uso: python scripts/actualizar_datos.py [--csv-dir DIR]"""
import json, math, sys
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

LIGAS = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
BASE = "https://www.football-data.co.uk"
DATA = Path(__file__).resolve().parent.parent / "data"
CORE = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
ODDS = ["B365H", "B365D", "B365A"]
HL, K0, RHO, HOME_ELO = 120.0, 4.0, -0.10, 60.0   # semivida (días), suavizado, Dixon-Coles, ventaja local Elo
EPOCH = pd.Timestamp(1970, 1, 1)
NAN = float("nan")


def fechas():
    hoy = date.today(); ini = hoy.year if hoy.month >= 7 else hoy.year - 1
    return hoy, ini, [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(ini - 4, ini + 1)]


def leer(src):
    d = pd.read_csv(src, encoding="latin-1")
    d.columns = [str(c).replace("ï»¿", "").replace("﻿", "").strip() for c in d.columns]  # quita el BOM
    for c in ODDS:
        if c not in d.columns:
            d[c] = float("nan")
    return d


def cargar(cod, temps, csv_dir):
    partes = []
    for t in temps:
        copia = DATA / f"{cod}_{t}.csv"
        try:
            d = leer(csv_dir / f"{cod}_{t}.csv" if csv_dir else f"{BASE}/mmz4281/{t}/{cod}.csv")
            d = d[CORE + ODDS].dropna(subset=CORE)
            d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed")
            d[["FTHG", "FTAG"]] = d[["FTHG", "FTAG"]].astype(int)
            d.to_csv(copia, index=False)
        except Exception as ex:
            if csv_dir is None and copia.exists():   # la fuente fallo: se usa la copia guardada en el repositorio
                print("  usando copia guardada", cod, t, "-", str(ex)[:80])
                d = pd.read_csv(copia); d["Date"] = pd.to_datetime(d["Date"])
                for c in ODDS:
                    if c not in d.columns:
                        d[c] = float("nan")
            else:
                print("  sin datos", cod, t, ex); continue
        partes.append(d)
    if not partes:
        raise ValueError("sin datos")
    df = pd.concat(partes).sort_values("Date", kind="stable").reset_index(drop=True)
    df["dia"] = (df["Date"] - EPOCH).dt.days
    return df


def fz(lst, ref, mu_gf, mu_ga):
    """Ataque y defensa relativos (1.0 = promedio), con pesos 0.5^(días/HL) y suavizado a la media."""
    sw = sgf = sga = 0.0
    for d, gf, ga in lst:
        w = 0.5 ** ((ref - d) / HL); sw += w; sgf += w * gf; sga += w * ga
    return ((sgf + K0 * mu_gf) / (sw + K0)) / mu_gf, ((sga + K0 * mu_ga) / (sw + K0)) / mu_ga


def probs(lh, la, n=8):
    ph = [math.exp(-lh) * lh ** k / math.factorial(k) for k in range(n + 1)]
    pa = [math.exp(-la) * la ** k / math.factorial(k) for k in range(n + 1)]
    m = [[x * y for y in pa] for x in ph]
    m[0][0] *= 1 - lh * la * RHO; m[0][1] *= 1 + lh * RHO; m[1][0] *= 1 + la * RHO; m[1][1] *= 1 - RHO
    s = sum(map(sum, m))
    pl = sum(m[i][j] for i in range(n + 1) for j in range(i)) / s
    pe = sum(m[i][i] for i in range(n + 1)) / s
    return (pl, pe, 1 - pl - pe)


def mercado(r):
    o = [r.B365H, r.B365D, r.B365A]
    if any(pd.isna(x) or x <= 1 for x in o):
        return None
    inv = [1 / x for x in o]; s = sum(inv)
    return tuple(x / s for x in inv)


def forma(l, d):
    """Últimos 5 partidos: puntos, goles a favor, goles en contra (promedios) y días de descanso."""
    r = l[-5:]; rest = min(d - l[-1][0], 21) if l else NAN
    if len(r) < 3:
        return [NAN, NAN, NAN, rest]
    return [sum(x[1] for x in r) / len(r), sum(x[2] for x in r) / len(r), sum(x[3] for x in r) / len(r), rest]


def feat(S, h, a, d):
    """Variables del partido h vs a usando SOLO lo que se sabe antes del día d."""
    mh, ma = S["sh"] / S["n"], S["sa"] / S["n"]
    ah, dh = fz(S["home"].get(h, []), d, mh, ma); aa, da = fz(S["away"].get(a, []), d, ma, mh)
    lh, la = mh * ah * da, ma * aa * dh; p = probs(lh, la)
    eh, ea = S["elo"].get(h, 1500.0), S["elo"].get(a, 1500.0)
    x = [lh, la, *p, eh, ea, eh + HOME_ELO - ea, ah, dh, aa, da, *forma(S["last"].get(h, []), d), *forma(S["last"].get(a, []), d)]
    return x, p


def procesar(df):
    S = {"home": {}, "away": {}, "last": {}, "elo": {}, "sh": 0.0, "sa": 0.0, "n": 0}
    hist, rows, cnt = {}, [], [0, 0, 0]
    for r in df.itertuples():
        h, a, hg, ag, d = r.HomeTeam, r.AwayTeam, int(r.FTHG), int(r.FTAG), int(r.dia)
        hh, ha = S["home"].setdefault(h, []), S["away"].setdefault(a, [])
        y = 0 if hg > ag else 1 if hg == ag else 2
        if S["n"] >= 60 and len(hh) + len(S["away"].get(h, [])) >= 4 and len(ha) + len(S["home"].get(a, [])) >= 4:
            x, p = feat(S, h, a, d)
            rows.append({"d": d, "y": y, "x": x, "p": p, "mk": mercado(r),
                         "base": tuple((c + 1) / (sum(cnt) + 3) for c in cnt)})
        hh.append((d, hg, ag)); ha.append((d, ag, hg))
        S["last"].setdefault(h, []).append((d, 3 if y == 0 else 1 if y == 1 else 0, hg, ag))
        S["last"].setdefault(a, []).append((d, 3 if y == 2 else 1 if y == 1 else 0, ag, hg))
        S["sh"] += hg; S["sa"] += ag; S["n"] += 1; cnt[y] += 1
        eh, ea = S["elo"].get(h, 1500.0), S["elo"].get(a, 1500.0)
        ex = 1 / (1 + 10 ** (-(eh + HOME_ELO - ea) / 400)); gd = abs(hg - ag)
        mult = 1 if gd < 2 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = 20 * mult * ((1.0 if y == 0 else 0.5 if y == 1 else 0.0) - ex)
        S["elo"][h], S["elo"][a] = eh + delta, ea - delta
        ymd = int(r.Date.strftime("%Y%m%d"))
        hist.setdefault(h, []).append([ymd, round(eh + delta)]); hist.setdefault(a, []).append([ymd, round(ea - delta)])
    return S, hist, rows


def modelo_ml():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=40,
                                          l2_regularization=1.0, random_state=0)


def evaluar_ml(rows, test_desde):
    """Walk-forward: cada 30 días reentrena con todo lo anterior y predice el bloque siguiente."""
    X = np.array([r["x"] for r in rows], float); y = np.array([r["y"] for r in rows]); d = np.array([r["d"] for r in rows])
    for ini in range(test_desde, int(d.max()) + 1, 30):
        tr, te = d < ini, (d >= ini) & (d < ini + 30)
        if tr.sum() < 300 or not te.any() or len(set(y[tr])) < 3:
            continue
        P = modelo_ml().fit(X[tr], y[tr]).predict_proba(X[te])
        for i, q in zip(np.where(te)[0], P):
            rows[i]["ml"] = tuple(float(v) for v in q)
    return modelo_ml().fit(X, y)


def vista(rs, modo):
    def pr(r):
        return r["p"] if modo == "p" else r["ml"] if modo == "ml" else tuple((a + b) / 2 for a, b in zip(r["p"], r["ml"]))
    return [(r["y"], pr(r), r["mk"], r["base"], r["d"]) for r in rs]


def metricas(rs):
    if not rs:
        return None
    ll = lambda p, y: -math.log(max(p[y], 1e-9))
    br = lambda p, y: sum((p[k] - (k == y)) ** 2 for k in range(3))
    ac = lambda p, y: max(range(3), key=lambda k: p[k]) == y
    n = len(rs); b = [x for x in rs if x[2]]
    out = {"n": n, "acc": sum(ac(x[1], x[0]) for x in rs) / n, "ll": sum(ll(x[1], x[0]) for x in rs) / n,
           "brier": sum(br(x[1], x[0]) for x in rs) / n, "ll0": sum(ll(x[3], x[0]) for x in rs) / n,
           "acc_local": sum(x[0] == 0 for x in rs) / n, "nb": len(b)}
    if b:
        out.update({"accB": sum(ac(x[2], x[0]) for x in b) / len(b), "llB": sum(ll(x[2], x[0]) for x in b) / len(b),
                    "accM": sum(ac(x[1], x[0]) for x in b) / len(b), "llM": sum(ll(x[1], x[0]) for x in b) / len(b)})
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def calibracion(rs):
    bins = [[0.0, 0.0, 0] for _ in range(10)]
    for y, p, *_ in rs:
        for k in range(3):
            i = min(int(p[k] * 10), 9); bins[i][0] += p[k]; bins[i][1] += (k == y); bins[i][2] += 1
    return [[round(s / c, 3), round(o / c, 3), c] for s, o, c in bins if c >= 20]


def proximos(csv_dir):
    out = {c: [] for c in LIGAS}
    try:
        f = leer(csv_dir / "fixtures.csv" if csv_dir else f"{BASE}/fixtures.csv")
        f = f[f.Div.isin(LIGAS)].dropna(subset=["HomeTeam", "AwayTeam", "Date"])
        f["Date"] = pd.to_datetime(f["Date"], dayfirst=True, format="mixed")
        for r in f.sort_values("Date", kind="stable").itertuples():
            m = mercado(r)
            out[r.Div].append({"f": r.Date.strftime("%Y-%m-%d"), "t": str(getattr(r, "Time", "") or ""), "h": r.HomeTeam, "a": r.AwayTeam,
                               "mk": [round(x, 3) for x in m] if m else None})
    except Exception as ex:
        print("  sin próximos partidos:", ex)
    return out


SEL_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
ES = {"Brazil": "Brasil", "Spain": "España", "Germany": "Alemania", "France": "Francia", "England": "Inglaterra", "Italy": "Italia",
      "Netherlands": "Países Bajos", "Belgium": "Bélgica", "Croatia": "Croacia", "Switzerland": "Suiza", "Denmark": "Dinamarca",
      "Sweden": "Suecia", "Norway": "Noruega", "Poland": "Polonia", "Czech Republic": "Chequia", "Turkey": "Turquía", "Greece": "Grecia",
      "Scotland": "Escocia", "Wales": "Gales", "Republic of Ireland": "Irlanda", "Hungary": "Hungría", "Romania": "Rumanía",
      "Ukraine": "Ucrania", "Russia": "Rusia", "Mexico": "México", "United States": "Estados Unidos", "Canada": "Canadá",
      "Peru": "Perú", "Panama": "Panamá", "Haiti": "Haití", "Japan": "Japón", "South Korea": "Corea del Sur", "Iran": "Irán",
      "Saudi Arabia": "Arabia Saudita", "Qatar": "Catar", "Iraq": "Irak", "Morocco": "Marruecos", "Egypt": "Egipto", "Algeria": "Argelia",
      "Tunisia": "Túnez", "Cameroon": "Camerún", "Ivory Coast": "Costa de Marfil", "South Africa": "Sudáfrica", "DR Congo": "RD Congo",
      "Uzbekistan": "Uzbekistán", "Jordan": "Jordania", "China PR": "China", "New Zealand": "Nueva Zelanda", "Slovakia": "Eslovaquia",
      "Slovenia": "Eslovenia", "Iceland": "Islandia", "Finland": "Finlandia", "Northern Ireland": "Irlanda del Norte",
      "North Macedonia": "Macedonia del Norte", "Cape Verde": "Cabo Verde", "Bosnia and Herzegovina": "Bosnia y Herzegovina"}


def k_torneo(t):
    t = str(t)
    return 20 if t == "Friendly" else 60 if t == "FIFA World Cup" else 40 if "qualification" in t else 50 if any(
        w in t for w in ("Euro", "Copa Am", "African Cup", "Asian Cup", "Gold Cup", "Nations League", "Confederations")) else 30


def selecciones(csv_dir, hoy):
    """Selecciones nacionales: Elo + Poisson (goles esperados segun la diferencia de Elo). Cancha neutral."""
    d = pd.read_csv(csv_dir / "results.csv" if csv_dir else SEL_URL)
    d["date"] = pd.to_datetime(d["date"]); d = d.dropna(subset=["home_score", "away_score"])
    d = d[(d.date >= "2014-01-01") & (d.date <= pd.Timestamp(hoy))].sort_values("date", kind="stable").reset_index(drop=True)
    d[["home_score", "away_score"]] = d[["home_score", "away_score"]].astype(int)
    for c in ("home_team", "away_team"):
        d[c] = d[c].replace(ES)
    d[d.date >= "2018-01-01"][["date", "home_team", "away_team", "home_score", "away_score", "tournament", "neutral"]].to_csv(
        DATA / "selecciones.csv", index=False)
    neu = d["neutral"].astype(str).str.upper().eq("TRUE")
    test_desde = (pd.Timestamp(hoy) - pd.Timedelta(days=730) - EPOCH).days
    ini_v = pd.Timestamp(hoy) - pd.Timedelta(days=730); ini_ymd = int(ini_v.strftime("%Y%m%d"))
    elo, n, rows, cnt, hist = {}, {}, [], [0, 0, 0], {}
    for r, ne in zip(d.itertuples(), neu):
        h, a, hg, ag, dia = r.home_team, r.away_team, r.home_score, r.away_score, (r.date - EPOCH).days
        eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)
        dif = eh + (0 if ne else HOME_ELO) - ea; y = 0 if hg > ag else 1 if hg == ag else 2
        if n.get(h, 0) >= 8 and n.get(a, 0) >= 8 and r.date >= pd.Timestamp("2016-01-01"):
            rows.append((dia, y, dif, hg + ag, tuple((c + 1) / (sum(cnt) + 3) for c in cnt)))
        cnt[y] += 1; n[h] = n.get(h, 0) + 1; n[a] = n.get(a, 0) + 1
        gd = abs(hg - ag); mult = 1 if gd < 2 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = k_torneo(r.tournament) * mult * ((1.0 if y == 0 else 0.5 if y == 1 else 0.0) - 1 / (1 + 10 ** (-dif / 400)))
        elo[h], elo[a] = eh + delta, ea - delta
        ymd = int(r.date.strftime("%Y%m%d"))
        hist.setdefault(h, []).append([ymd, round(eh + delta)]); hist.setdefault(a, []).append([ymd, round(ea - delta)])
    mu = float((d.home_score.sum() + d.away_score.sum()) / (2 * len(d)))
    tr = [x for x in rows if test_desde - 1460 <= x[0] < test_desde]; te = [x for x in rows if x[0] >= test_desde]
    if len(tr) < 300 or len(te) < 100:
        raise ValueError("pocos partidos de selecciones")
    pr = lambda dif, b: probs(mu * math.exp(b * dif / 400), mu * math.exp(-b * dif / 400))
    ll = lambda b: sum(-math.log(max(pr(x[2], b)[x[1]], 1e-9)) for x in tr)
    b = min([0.3 + 0.05 * i for i in range(17)], key=ll)
    bt = metricas([(x[1], pr(x[2], b), None, x[4], x[0]) for x in te])
    ven = d[d.date >= ini_v]; cuenta = {}
    for r in ven.itertuples():
        cuenta[r.home_team] = cuenta.get(r.home_team, 0) + 1; cuenta[r.away_team] = cuenta.get(r.away_team, 0) + 1
    orden = sorted(t for t, c in cuenta.items() if c >= 20)
    eq = {}
    for t in orden:
        g_ = ven[(ven.home_team == t) | (ven.away_team == t)]
        gf = ga = w = e = 0; res = []
        for r in g_.itertuples():
            a_, c_ = (r.home_score, r.away_score) if r.home_team == t else (r.away_score, r.home_score)
            gf += a_; ga += c_; w += a_ > c_; e += a_ == c_; res.append("G" if a_ > c_ else "E" if a_ == c_ else "P")
        f = math.exp(b * (elo[t] - 1500) / 400)
        eq[t] = {"pj": len(g_), "g": w, "e": e, "p": len(g_) - w - e, "gf": int(gf), "gc": int(ga), "pts": 3 * w + e, "forma": res[-5:],
                 "ataque_l": round(f, 4), "defensa_l": round(1 / f, 4), "ataque_v": round(f, 4), "defensa_v": round(1 / f, 4),
                 "elo": round(elo[t])}
    L = {"nombre": "Selecciones nacionales", "avg_h": round(mu, 4), "avg_a": round(mu, 4), "n": len(ven), "equipos": eq, "orden": orden,
         "proximos": [], "neutral": True, "desde": ini_ymd, "backtest": bt,
         "nota": "Selecciones: ranking Elo de los últimos 2 años, todo en cancha neutral, sin calendario ni cuotas."}
    t3 = int((pd.Timestamp(hoy) - pd.Timedelta(days=1095)).strftime("%Y%m%d"))
    ix = {t: i for i, t in enumerate(orden)}
    d3 = d[(d.date >= pd.Timestamp(str(t3))) & d.home_team.isin(ix) & d.away_team.isin(ix)]
    P = {"teams": orden, "elo": {str(ix[t]): [q for q in hist[t] if q[0] >= t3] for t in orden},
         "m": [[int(r.date.strftime("%Y%m%d")), ix[r.home_team], ix[r.away_team], int(r.home_score), int(r.away_score)] for r in d3.itertuples()]}
    print("OK INT", len(d), "partidos;", len(te), "en backtest;", len(orden), "selecciones; b =", b)
    return L, P


def main():
    csv_dir = Path(sys.argv[sys.argv.index("--csv-dir") + 1]) if "--csv-dir" in sys.argv else None
    DATA.mkdir(exist_ok=True)
    hoy, ini, temps = fechas()
    hoy_dia = (pd.Timestamp(hoy) - EPOCH).days
    inicio_temp = pd.Timestamp(ini, 7, 1); test_desde = (pd.Timestamp(ini - 1, 7, 1) - EPOCH).days
    tres = (pd.Timestamp(ini - 2, 7, 1) - EPOCH).days; tres_ymd = int(f"{ini - 2}0701")
    prox, base, todas = proximos(csv_dir), {}, []
    for cod in LIGAS:
        try:
            df = cargar(cod, temps, csv_dir)
            S, hist, rows = procesar(df)
        except Exception as ex:
            print("SALTADA", cod, ex); continue
        for r in rows:
            r["lig"] = cod
        base[cod] = (df, S, hist); todas += rows
    if not base:
        sys.exit("No se pudo cargar ninguna liga")
    modelo, ml_ok = None, False
    try:
        modelo = evaluar_ml(todas, test_desde); ml_ok = True
    except Exception as ex:
        print("IA omitida:", ex)
    ev = [r for r in todas if r["d"] >= test_desde and (not ml_ok or "ml" in r)]
    ligas, partidos = {}, {}
    for cod, (df, S, hist) in base.items():
        cur = df[df.Date >= inicio_temp]; mh, ma = S["sh"] / S["n"], S["sa"] / S["n"]; eq = {}
        for t in sorted(set(cur.HomeTeam) | set(cur.AwayTeam)):
            h, a = cur[cur.HomeTeam == t], cur[cur.AwayTeam == t]
            g = int((h.FTHG > h.FTAG).sum() + (a.FTAG > a.FTHG).sum()); e = int((h.FTHG == h.FTAG).sum() + (a.FTAG == a.FTHG).sum())
            ult = pd.concat([h.assign(r=h.FTHG - h.FTAG), a.assign(r=a.FTAG - a.FTHG)]).sort_values("Date").tail(5)
            al, dl = fz(S["home"].get(t, []), hoy_dia, mh, ma); av, dv = fz(S["away"].get(t, []), hoy_dia, ma, mh)
            eq[t] = {"pj": len(h) + len(a), "g": g, "e": e, "p": len(h) + len(a) - g - e,
                     "gf": int(h.FTHG.sum() + a.FTAG.sum()), "gc": int(h.FTAG.sum() + a.FTHG.sum()), "pts": 3 * g + e,
                     "forma": ["G" if r > 0 else "E" if r == 0 else "P" for r in ult.r],
                     "ataque_l": round(al, 4), "defensa_l": round(dl, 4), "ataque_v": round(av, 4), "defensa_v": round(dv, 4),
                     "elo": round(S["elo"].get(t, 1500))}
        orden = list(eq)
        L = {"nombre": LIGAS[cod], "avg_h": round(mh, 4), "avg_a": round(ma, 4), "n": len(cur), "equipos": eq, "orden": orden,
             "proximos": [x for x in prox[cod] if x["h"] in eq and x["a"] in eq]}
        if ml_ok:   # probabilidades de la IA para todos los cruces posibles (local i, visitante j)
            pares = [(i, j) for i in range(len(orden)) for j in range(len(orden)) if i != j]
            X = np.array([feat(S, orden[i], orden[j], hoy_dia)[0] for i, j in pares], float)
            M = [[None] * len(orden) for _ in orden]
            for (i, j), q in zip(pares, modelo.predict_proba(X)):
                M[i][j] = [round(float(v), 3) for v in q]
            L["ml"] = M
        mine = [r for r in ev if r["lig"] == cod]
        L["backtest"] = metricas(vista(mine, "p"))
        if ml_ok:
            L["backtest_ml"] = metricas(vista(mine, "ml")); L["backtest_mix"] = metricas(vista(mine, "mix"))
        ligas[cod] = L
        d3 = df[df.dia >= tres]; nombres = sorted(set(d3.HomeTeam) | set(d3.AwayTeam)); ix = {t: i for i, t in enumerate(nombres)}
        partidos[cod] = {"teams": nombres,
                         "elo": {str(ix[t]): [q for q in v if q[0] >= tres_ymd] for t, v in hist.items() if t in ix},
                         "m": [[int(r.Date.strftime("%Y%m%d")), ix[r.HomeTeam], ix[r.AwayTeam], int(r.FTHG), int(r.FTAG)] for r in d3.itertuples()]}
        print("OK", cod, len(df), "partidos;", len(mine), "en backtest;", len(L["proximos"]), "próximos")
    try:
        ligas["INT"], partidos["INT"] = selecciones(csv_dir, hoy)
    except Exception as ex:
        print("Selecciones omitidas:", ex)
    bt = {"global": metricas(vista(ev, "p")), "desde": f"{ini - 1}-07-01"}
    mejor, vm = "poisson", "p"
    if ml_ok:
        bt["ml"], bt["mix"] = metricas(vista(ev, "ml")), metricas(vista(ev, "mix"))
        for nombre, m in (("ml", bt["ml"]), ("mix", bt["mix"])):
            if m and m["ll"] < bt["global"]["ll"] and m["ll"] < (bt[mejor]["ll"] if mejor != "poisson" else 9):
                mejor = nombre
        vm = {"poisson": "p"}.get(mejor, mejor)
    bt["mejor"] = mejor; bt["cal"] = calibracion(vista(ev, vm)) if ev else []
    out = {"temporada": temps[-1], "actualizado": hoy.isoformat(), "ligas": ligas, "backtest": bt,
           "modelo": {"semivida_dias": HL, "suavizado": K0, "rho": RHO}}
    (DATA / "datos.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    (DATA / "partidos.json").write_text(json.dumps(partidos, separators=(",", ":")), encoding="utf-8")
    print("Modelo principal para el 1X2:", mejor)


if __name__ == "__main__":
    main()

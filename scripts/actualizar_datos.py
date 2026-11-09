"""Descarga 3 temporadas de football-data.co.uk (nuestra 'base de datos' en CSV), calcula fuerzas de equipos
con pesos por recencia, ratings Elo, backtesting del modelo y próximos partidos.
Genera data/datos.json y data/partidos.json.  Uso: python scripts/actualizar_datos.py [--csv-dir DIR]"""
import json, math, sys
from datetime import date
from pathlib import Path
import pandas as pd

LIGAS = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
BASE = "https://www.football-data.co.uk"
DATA = Path(__file__).resolve().parent.parent / "data"
CORE = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
ODDS = ["B365H", "B365D", "B365A"]
HL, K0, RHO, HOME_ELO = 120.0, 4.0, -0.10, 60.0   # semivida (días), suavizado, Dixon-Coles, ventaja local Elo
EPOCH = pd.Timestamp(1970, 1, 1)


def fechas():
    hoy = date.today(); ini = hoy.year if hoy.month >= 7 else hoy.year - 1
    return hoy, ini, [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in (ini - 2, ini - 1, ini)]


def leer(src):
    d = pd.read_csv(src, encoding="latin-1")
    d.columns = [str(c).replace("ï»¿", "").replace("\ufeff", "").strip() for c in d.columns]  # quita el BOM
    for c in ODDS:
        if c not in d.columns:
            d[c] = float("nan")
    return d


def cargar(cod, temps, csv_dir):
    partes = []
    for t in temps:
        try:
            d = leer(csv_dir / f"{cod}_{t}.csv" if csv_dir else f"{BASE}/mmz4281/{t}/{cod}.csv")
            d = d[CORE + ODDS].dropna(subset=CORE)
            d["Date"] = pd.to_datetime(d["Date"], dayfirst=True, format="mixed")
            d[["FTHG", "FTAG"]] = d[["FTHG", "FTAG"]].astype(int)
            d.to_csv(DATA / f"{cod}_{t}.csv", index=False)
            partes.append(d)
        except Exception as ex:
            print("  sin datos", cod, t, ex)
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


def procesar(df, test_desde, hoy_dia):
    home, away, elo, hist, bt = {}, {}, {}, {}, []
    sh = sa = n = 0.0; cnt = [0, 0, 0]
    for r in df.itertuples():
        h, a, hg, ag, d = r.HomeTeam, r.AwayTeam, int(r.FTHG), int(r.FTAG), int(r.dia)
        hh, ha = home.setdefault(h, []), away.setdefault(a, [])
        y = 0 if hg > ag else 1 if hg == ag else 2
        if d >= test_desde and n >= 60 and len(hh) + len(away.get(h, [])) >= 4 and len(ha) + len(home.get(a, [])) >= 4:
            mh, ma = sh / n, sa / n
            ah, dh = fz(hh, d, mh, ma); aa, da = fz(ha, d, ma, mh)
            p = probs(mh * ah * da, ma * aa * dh)
            base = tuple((c + 1) / (sum(cnt) + 3) for c in cnt)
            bt.append((y, p, mercado(r), base, d))
        hh.append((d, hg, ag)); ha.append((d, ag, hg)); sh += hg; sa += ag; n += 1; cnt[y] += 1
        eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)
        ex = 1 / (1 + 10 ** (-(eh + HOME_ELO - ea) / 400)); gd = abs(hg - ag)
        mult = 1 if gd < 2 else 1.5 if gd == 2 else (11 + gd) / 8
        delta = 20 * mult * ((1.0 if y == 0 else 0.5 if y == 1 else 0.0) - ex)
        elo[h], elo[a] = eh + delta, ea - delta
        ymd = int(r.Date.strftime("%Y%m%d"))
        hist.setdefault(h, []).append([ymd, round(eh + delta)]); hist.setdefault(a, []).append([ymd, round(ea - delta)])
    return bt, elo, hist, home, away, sh / n, sa / n


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


def main():
    csv_dir = Path(sys.argv[sys.argv.index("--csv-dir") + 1]) if "--csv-dir" in sys.argv else None
    DATA.mkdir(exist_ok=True)
    hoy, ini, temps = fechas()
    hoy_dia = (pd.Timestamp(hoy) - EPOCH).days
    inicio_temp = pd.Timestamp(ini, 7, 1); test_desde = (pd.Timestamp(ini - 1, 7, 1) - EPOCH).days
    prox, ligas, partidos, todo = proximos(csv_dir), {}, {}, []
    for cod, nombre in LIGAS.items():
        try:
            df = cargar(cod, temps, csv_dir)
            bt, elo, hist, home, away, mh, ma = procesar(df, test_desde, hoy_dia)
        except Exception as ex:
            print("SALTADA", cod, ex); continue
        cur = df[df.Date >= inicio_temp]
        eq = {}
        for t in sorted(set(cur.HomeTeam) | set(cur.AwayTeam)):
            h, a = cur[cur.HomeTeam == t], cur[cur.AwayTeam == t]
            g = int((h.FTHG > h.FTAG).sum() + (a.FTAG > a.FTHG).sum()); e = int((h.FTHG == h.FTAG).sum() + (a.FTAG == a.FTHG).sum())
            ult = pd.concat([h.assign(r=h.FTHG - h.FTAG), a.assign(r=a.FTAG - a.FTHG)]).sort_values("Date").tail(5)
            al, dl = fz(home.get(t, []), hoy_dia, mh, ma); av, dv = fz(away.get(t, []), hoy_dia, ma, mh)
            eq[t] = {"pj": len(h) + len(a), "g": g, "e": e, "p": len(h) + len(a) - g - e,
                     "gf": int(h.FTHG.sum() + a.FTAG.sum()), "gc": int(h.FTAG.sum() + a.FTHG.sum()), "pts": 3 * g + e,
                     "forma": ["G" if r > 0 else "E" if r == 0 else "P" for r in ult.r],
                     "ataque_l": round(al, 4), "defensa_l": round(dl, 4), "ataque_v": round(av, 4), "defensa_v": round(dv, 4),
                     "elo": round(elo.get(t, 1500))}
        ligas[cod] = {"nombre": nombre, "avg_h": round(mh, 4), "avg_a": round(ma, 4), "n": len(cur), "equipos": eq,
                      "proximos": [x for x in prox[cod] if x["h"] in eq and x["a"] in eq]}
        nombres = sorted(set(df.HomeTeam) | set(df.AwayTeam)); ix = {t: i for i, t in enumerate(nombres)}
        partidos[cod] = {"teams": nombres, "elo": {str(ix[t]): v for t, v in hist.items()},
                         "m": [[int(r.Date.strftime("%Y%m%d")), ix[r.HomeTeam], ix[r.AwayTeam], int(r.FTHG), int(r.FTAG)] for r in df.itertuples()]}
        ligas[cod]["backtest"] = metricas(bt); todo += bt
        print("OK", cod, len(df), "partidos;", len(bt), "en backtest;", len(ligas[cod]["proximos"]), "próximos")
    if not ligas:
        sys.exit("No se pudo cargar ninguna liga")
    out = {"temporada": temps[-1], "actualizado": hoy.isoformat(), "ligas": ligas,
           "backtest": {"global": metricas(todo), "cal": calibracion(todo), "desde": f"{ini - 1}-07-01"},
           "modelo": {"semivida_dias": HL, "suavizado": K0, "rho": RHO}}
    (DATA / "datos.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    (DATA / "partidos.json").write_text(json.dumps(partidos, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()

"""Descarga resultados de football-data.co.uk, guarda los CSV (nuestra 'base de datos')
y calcula data/datos.json que consume index.html.  Uso: python scripts/actualizar_datos.py [--csv-dir DIR]"""
import json, sys
from datetime import date
from pathlib import Path
import pandas as pd

LIGAS = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A", "D1": "Bundesliga", "F1": "Ligue 1"}
COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
DATA = Path(__file__).resolve().parent.parent / "data"


def temporada():
    hoy = date.today(); ini = hoy.year if hoy.month >= 7 else hoy.year - 1
    return f"{ini % 100:02d}{(ini + 1) % 100:02d}"


def resumen(df):
    ah, aa = df.FTHG.mean(), df.FTAG.mean(); out = {}
    for t in sorted(set(df.HomeTeam) | set(df.AwayTeam)):
        h, a = df[df.HomeTeam == t], df[df.AwayTeam == t]
        g = int((h.FTHG > h.FTAG).sum() + (a.FTAG > a.FTHG).sum())
        e = int((h.FTHG == h.FTAG).sum() + (a.FTAG == a.FTHG).sum())
        ult = pd.concat([h.assign(r=h.FTHG - h.FTAG), a.assign(r=a.FTAG - a.FTHG)]).sort_values("Date").tail(5)
        out[t] = {"pj": len(h) + len(a), "g": g, "e": e, "p": len(h) + len(a) - g - e,
                  "gf": int(h.FTHG.sum() + a.FTAG.sum()), "gc": int(h.FTAG.sum() + a.FTHG.sum()),
                  "pts": 3 * g + e, "forma": ["G" if r > 0 else "E" if r == 0 else "P" for r in ult.r],
                  "ataque_l": float(h.FTHG.mean() / ah) if len(h) else 1.0,
                  "defensa_l": float(h.FTAG.mean() / aa) if len(h) else 1.0,
                  "ataque_v": float(a.FTAG.mean() / aa) if len(a) else 1.0,
                  "defensa_v": float(a.FTHG.mean() / ah) if len(a) else 1.0}
    return {"avg_h": float(ah), "avg_a": float(aa), "n": len(df), "equipos": out}


def main():
    csv_dir = Path(sys.argv[sys.argv.index("--csv-dir") + 1]) if "--csv-dir" in sys.argv else None
    DATA.mkdir(exist_ok=True); ligas = {}; temp = temporada()
    for cod, nombre in LIGAS.items():
        try:
            src = csv_dir / f"{cod}.csv" if csv_dir else f"https://www.football-data.co.uk/mmz4281/{temp}/{cod}.csv"
            df = pd.read_csv(src)[COLS].dropna()
            df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
            df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].astype(int)
            df = df.sort_values("Date")
            df.to_csv(DATA / f"{cod}_{temp}.csv", index=False)
            ligas[cod] = {"nombre": nombre, **resumen(df)}
            print("OK", cod, len(df))
        except Exception as ex:
            print("SALTADA", cod, ex)
    if not ligas:
        sys.exit("No se pudo cargar ninguna liga")
    (DATA / "datos.json").write_text(json.dumps({"temporada": temp, "actualizado": date.today().isoformat(), "ligas": ligas}), encoding="utf-8")


if __name__ == "__main__":
    main()

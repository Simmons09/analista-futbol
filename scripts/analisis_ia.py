"""Pide a Claude un análisis corto de cada partido próximo, usando SOLO los números del modelo.
Solo funciona si existe el secreto ANTHROPIC_API_KEY; si no, se omite sin error.
Genera data/analisis.json.  Uso: python scripts/analisis_ia.py"""
import hashlib, json, os, sys, urllib.error, urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from actualizar_datos import probs  # mismo modelo Poisson del sitio

DATA = Path(__file__).resolve().parent.parent / "data"
KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
MODELO = os.environ.get("MODELO_IA", "").strip() or "claude-haiku-5-5"
MAX_PARTIDOS = 30


def llamar(prompt):
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", method="POST",
        headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
        data=json.dumps({"model": MODELO, "max_tokens": 400, "messages": [{"role": "user", "content": prompt}]}).encode())
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["content"][0]["text"].strip()


def hechos(L, P, cod, x):
    h, a, eq = x["h"], x["a"], L["equipos"]
    H, A = eq[h], eq[a]
    lh = H["ataque_l"] * A["defensa_v"] * L["avg_h"]; la = A["ataque_v"] * H["defensa_l"] * L["avg_a"]
    pl, pe, pv = probs(lh, la)
    out = {"liga": L["nombre"], "fecha": x["f"], "local": h, "visitante": a,
           "probabilidades_poisson": {"gana_local": round(pl, 3), "empate": round(pe, 3), "gana_visitante": round(pv, 3)},
           "goles_esperados": {"local": round(lh, 2), "visitante": round(la, 2)},
           "local": {"nombre": h, "puntos": H["pts"], "partidos": H["pj"], "gf": H["gf"], "gc": H["gc"], "elo": H["elo"], "forma_ultimos_5": "".join(H["forma"])},
           "visitante": {"nombre": a, "puntos": A["pts"], "partidos": A["pj"], "gf": A["gf"], "gc": A["gc"], "elo": A["elo"], "forma_ultimos_5": "".join(A["forma"])}}
    if L.get("ml") and h in L["orden"] and a in L["orden"]:
        q = L["ml"][L["orden"].index(h)][L["orden"].index(a)]
        if q:
            out["probabilidades_ia"] = {"gana_local": q[0], "empate": q[1], "gana_visitante": q[2]}
    p = P.get(cod)
    if p and h in p["teams"] and a in p["teams"]:
        ih, ia = p["teams"].index(h), p["teams"].index(a)
        out["ultimos_enfrentamientos"] = [f"{p['teams'][m[1]]} {m[3]}-{m[4]} {p['teams'][m[2]]}"
                                          for m in reversed(p["m"]) if {m[1], m[2]} == {ih, ia}][:5]
    return out


def main():
    if not KEY:
        print("Sin ANTHROPIC_API_KEY: se omite el análisis con IA."); return
    D = json.loads((DATA / "datos.json").read_text(encoding="utf-8"))
    P = json.loads((DATA / "partidos.json").read_text(encoding="utf-8"))
    ruta = DATA / "analisis.json"
    previo = json.loads(ruta.read_text(encoding="utf-8")).get("partidos", {}) if ruta.exists() else {}
    hoy, nuevo, llamadas, parar = date.today().isoformat(), {}, 0, False
    for cod, L in D["ligas"].items():
        if parar:
            break
        for x in [p for p in L["proximos"] if p["f"] >= hoy]:
            k = f"{cod}|{x['f']}|{x['h']}|{x['a']}"
            datos = hechos(L, P, cod, x)
            huella = hashlib.sha256(json.dumps(datos, sort_keys=True).encode()).hexdigest()[:12]
            if k in previo and previo[k]["h"] == huella:
                nuevo[k] = previo[k]; continue
            if llamadas >= MAX_PARTIDOS:
                continue
            prompt = ("Eres un analista de fútbol. Usando SOLO los datos JSON de abajo, escribe en español un análisis de 3 a 4 frases "
                      f"sencillas del partido {x['h']} vs {x['a']}. Explica quién llega mejor y por qué, qué se espera en goles y cierra "
                      "recordando brevemente que es estadística y no una garantía. No inventes datos, lesiones, noticias ni alineaciones. "
                      "No des consejos de apuestas. Responde solo con el texto, sin títulos ni listas.\n\nDatos:\n" + json.dumps(datos, ensure_ascii=False))
            try:
                nuevo[k] = {"h": huella, "t": llamar(prompt)}; llamadas += 1
            except urllib.error.HTTPError as ex:
                print("Error de la API:", ex.code, "- se detiene el análisis con IA")
                if ex.code in (401, 403, 404):
                    parar = True; break
            except Exception as ex:
                print("Error en", k, ex)
    ruta.write_text(json.dumps({"generado": hoy, "modelo": MODELO, "partidos": nuevo}, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Análisis con IA: {len(nuevo)} partidos ({llamadas} nuevos)")


if __name__ == "__main__":
    main()

"""Render the immutable participation snapshot as a compact control sheet."""
import argparse
import json
from pathlib import Path


def render(path):
    def read(name):
        source = path / name
        return json.loads(source.read_text()) if source.exists() else {}

    audit = read("participation.json")
    run = read("run.json")
    mini = read("minibench.json")
    treasury = read("treasury.json")
    sha = run["RESEARCH_SHA"]
    base = "https://github.com/xmjunior/mmegalab-futureeval-bot"
    lines = [
        "# Control de competiciones de megalab-mario-bot",
        "",
        f"Observación UTC: {audit.get('observed_at','sin lectura completa')}. "
        f"[Ejecución {run['GITHUB_RUN_ID']}]({base}/actions/runs/{run['GITHUB_RUN_ID']}) "
        f"por `{run['GITHUB_EVENT_NAME']}`. [Código del control]({base}/commit/{sha}).",
        "",
        "| Torneo | Participación en la API | Pronósticos aceptados | Oportunidades desde el alta | Puesto | Peer Score | Comentarios privados vinculados |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for key in ("FutureEval", "MiniBench", "MarketPulse"):
        p = audit.get("projects", {}).get(key, {})
        if "forecasted_questions" not in p or p.get("error") or p.get("mismatching_post_ids") or p.get("own_history_without_post"):
            lines.append(f"| {key} | Lectura incompleta | — | — | — | — | — |")
            continue
        entry = next((b["own_entry"] for b in p["leaderboards"] if b.get("own_entry") and b.get("score_type")=="spot_peer_tournament"), {})
        status = {True: "Sí", False: "No"}.get(p.get("api_has_participated"), "Sin confirmación")
        private = sum(c["is_private"] and c["included_forecast"] for c in p["comments"])
        lines.append(f"| {key} {p['project']} | {status} | {p['forecasted_questions']} | {p['opportunities_overlapping_account_lifetime']} | {entry.get('rank','—')} | {entry.get('score','—')} | {private} |")
    lines.extend(["", "La cobertura compara preguntas aceptadas con ventanas que coincidieron con la vida de la cuenta. Un guion indica que el organizador todavía no asignó ese dato o que la lectura quedó incompleta.",
        "", f"MiniBench C: {mini.get('descriptive',{}).get('pairs','sin lectura')} pares resueltos. Treasury: {treasury.get('summary',{}).get('cases_resolved','sin lectura')} casos propios resueltos. Consulta los JSON de cada registro para ver fechas, resultados de los pasos y fallos del proveedor.",
        "", f"[Diseños, pruebas y recuperación]({base}/tree/{sha}/research) · [Pronósticos prospectivos y auditorías]({base}/tree/research/champion-observations/records)", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    args = parser.parse_args()
    print(render(args.state))

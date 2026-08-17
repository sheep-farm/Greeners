#!/usr/bin/env python3
"""Translate Portuguese user-facing output/error strings in Greeners.

This is a deterministic, exact-replacement pass.  It only changes the listed
string literals; it does not touch comments or unrelated code.
"""

from pathlib import Path

# (old Portuguese string, new English string)
OUTPUT_REPLACEMENTS = [
    # Tobit
    (" Tobit  —  MLE  (censura inferior em {})", " Tobit  —  MLE  (lower censoring at {})"),
    (" Obs: {:<8}  Censuradas: {:<6}  Não-cens.: {:<6}  Iter.: {}", " Obs: {:<8}  Censored: {:<6}  Uncens.: {:<6}  Iter.: {}"),
    ("Variável", "Variable"),
    ("Tobit: y e x têm dimensões incompatíveis", "Tobit: y and x have incompatible dimensions"),
    ("Tobit: dados contêm NaN ou Inf", "Tobit: data contain NaN or Inf"),

    # RD
    ("Local Quadrático", "Local Quadratic"),
    ("Local Cúbico", "Local Cubic"),
    (" Regressão Descontínua  —  {}  —  {} (p={})", " Regression Discontinuity  —  {}  —  {} (p={})"),
    ("rd: y e x têm tamanhos diferentes", "rd: y and x have different sizes"),
    ("rd: dados contêm NaN ou Inf", "rd: data contain NaN or Inf"),
    ("fuzzy_rd: dados contêm NaN ou Inf", "fuzzy_rd: data contain NaN or Inf"),
    ("fuzzy_rd: salto na primeira etapa é praticamente zero (τ_D ≈ 0)", "fuzzy_rd: first-stage jump is practically zero (τ_D ≈ 0)"),
    ("rd: observações insuficientes ({n}) para polinômio de ordem {poly_order} (lado {})", "rd: insufficient observations ({n}) for polynomial of order {poly_order} (side {})"),
    (" Efeito de Tratamento (τ̂):", " Treatment effect (τ̂):"),

    # Synthetic control
    (" Controle Sintético  —  Abadie, Diamond, Hainmueller (2010)", " Synthetic Control  —  Abadie, Diamond, Hainmueller (2010)"),
    (" T₀ (1ª pós-trat): {}   Doadores: {}   T pré: {}   T pós: {}", " T₀ (1st post-treat): {}   Donors: {}   T pre: {}   T post: {}"),
    (" RMSPE pré  : {:.4}", " RMSPE pre  : {:.4}"),
    (" RMSPE pós  : {:.4}   razão pós/pré: {:.3}", " RMSPE post  : {:.4}   post/pre ratio: {:.3}"),
    ("Período", "Period"),
    ("Real", "Actual"),
    ("Sintético", "Synthetic"),
    ("Efeito", "Effect"),
    (" * pós-tratamento", " * post-treatment"),
    ("synth: unidade tratada '{treated_unit}' não encontrada em '{id_col}'", "synth: treated unit '{treated_unit}' not found in '{id_col}'"),
    ("synth: apenas {t_pre} período(s) pré-tratamento (mínimo 2)", "synth: only {t_pre} pre-treatment period(s) (minimum 2)"),

    # PSM
    (" Balanço de covariáveis (SMD = diferença padronizada):", " Covariate balance (SMD = standardized difference):"),
    ("Covariável", "Covariate"),
    (" (!) SMD > 0.10 após matching — covariável mal balanceada", " (!) SMD > 0.10 after matching — covariate poorly balanced"),
    ("psm: y, d, x devem ter o mesmo número de observações", "psm: y, d, and x must have the same number of observations"),
    ("psm: dados contêm NaN ou Inf", "psm: data contain NaN or Inf"),
    ("psm: ATT não calculável — nenhum tratado obteve match", "psm: ATT not calculable — no treated unit got a match"),

    # Heckman
    (" Obs (total): {:<8}  Selecionadas: {}", " Obs (total): {:<8}  Selected: {}"),
    (" Equação de resultado  (y | z=1)", " Outcome equation  (y | z=1)"),
    (" Equação de seleção  (Probit — todos os obs)", " Selection equation  (Probit — all obs)"),

    # Panel (PCSE etc)
    ("Variável", "Variable"),  # already covered; kept for safety
    (" Obs: {:<8}  Entidades: {:<6}  Períodos: {:<6}  df_resid: {}", " Obs: {:<8}  Entities: {:<6}  Periods: {:<6}  df_resid: {}"),
    (" Obs: {:<8}  Entidades: {:<6}  Períodos: {:<6}  df_resid: {}", " Obs: {:<8}  Entities: {:<6}  Periods: {:<6}  df_resid: {}"),
    ("FE2SLS: dimensões de y, x, z e grupos divergem", "FE2SLS: dimensions of y, x, z and groups differ"),
    ("FE2SLS: condição de ordem violada — Z tem {l} instrumentos, X tem {k} regressores", "FE2SLS: order condition violated — Z has {l} instruments, X has {k} regressors"),
    ("FE2SLS: dados contêm NaN ou Inf", "FE2SLS: data contain NaN or Inf"),
    ("PCSE: dados contêm NaN ou Inf", "PCSE: data contain NaN or Inf"),
    ("PanelGLS: dados contêm NaN ou Inf", "PanelGLS: data contain NaN or Inf"),
    ("PanelGLS: σ²_i ≈ 0 para entidade {i} — resíduos perfeitamente ajustados?", "PanelGLS: σ²_i ≈ 0 for entity {i} — perfectly fitted residuals?"),
    ("painel não balanceado: número de períodos difere entre entidades", "unbalanced panel: number of periods differs between entities"),
    ("dimensões de y, x, entity_ids, time_ids divergem", "dimensions of y, x, entity_ids, time_ids differ"),

    # Model selection
    ("Período não encontrado no índice", "Period not found in index"),
    ("ID de entidade não encontrado", "Entity ID not found"),
]


def translate_file(path: Path) -> int:
    content = path.read_text(encoding="utf-8")
    changed = 0
    for old, new in OUTPUT_REPLACEMENTS:
        if old in content:
            content = content.replace(old, new)
            changed += 1
    if changed:
        path.write_text(content, encoding="utf-8")
    return changed


def main():
    root = Path(__file__).resolve().parent / "src"
    total = 0
    for path in sorted(root.rglob("*.rs")):
        n = translate_file(path)
        if n:
            print(f"  {path.name}: {n} replacements")
            total += n
    print(f"\nTotal replacements: {total}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Translate Portuguese string literals in Greeners source using argostranslate.

Only touches string literals that contain diacritics or known Portuguese words.
Preserves {} format placeholders, {{, }}, and Greek/statistical subscripts.
"""

import re
import sys
from pathlib import Path

import argostranslate.translate

DIACRITIC_RE = re.compile(r"[áàãâéêíóôõúçÁÀÃÂÉÊÍÓÔÕÚÇ]")

# Additional Portuguese words without diacritics that may appear in output/error
# strings but should not match common English.
PT_WORDS = {
    "nao", "não", "e", "ou", "de", "da", "do", "em", "para", "por", "com",
    "sem", "sob", "sobre", "entre", "antes", "depois", "agora", "ainda",
    "deve", "devem", "deveria", "deverá", "ser", "seja", "sendo",
    "faz", "fazem", "fazer", "fez", "vai", "vão", "ir",
    "pode", "podem", "posso", "possa", "tem", "têm", "tinha",
    "já", "só", "nem", "também", "mesmo", "tal", "assim", "aqui", "ali",
    "onde", "quando", "quem", "qual", "todos", "todo", "nada", "nenhum",
    "nenhuma", "algum", "alguma", "outro", "outra", "mais", "menos", "muito",
    "pouco", "bem", "mal", "grande", "pequeno", "novo", "velho", "bom",
    "ruim", "próprio", "único", "certo", "real", "falso", "verdadeiro",
    "primeiro", "segundo", "terceiro", "quarto", "quinta", "sexta", "último",
    "início", "fim", "começo", "meio", "ponto", "lado", "parte", "lugar",
    "forma", "modo", "meio", "caso", "exemplo", "razão", "ideia", "palavra",
    "nome", "número", "grupo", "tipo", "classe", "mundo", "país", "cidade",
    "estado", "região", "área", "setor", "mercado", "empresa", "trabalho",
    "emprego", "dinheiro", "custo", "preço", "valor", "quantidade", "total",
    "média", "mediana", "desvio", "mínimo", "máximo", "soma", "produto",
    "resultado", "efeito", "causa", "problema", "solução", "sistema",
    "processo", "método", "modelo", "variável", "coeficiente", "constante",
    "termo", "equação", "função", "dado", "conjunto", "base", "tabela",
    "gráfico", "figura", "imagem", "relatório", "texto", "linha", "coluna",
    "campo", "registro", "variaveis", "coeficientes", "funcao", "equacao",
    "grafico", "relatorio", "imagem", "numero", "observacoes", "observações",
    "tratamento", "selecao", "seleção", "covariavel", "sintetico", "sintético",
    "controle", "periodo", "período", "doador", "tratada", "censurada",
    "verossimilhanca", "assintotica", "assintótica", "padrao", "padrão",
    "media", "minimo", "maximo", "desconhecido", "invalido", "vazio",
    "nula", "esperado", "requerido", "argumento", "retornado", "calculado",
    "encontrado", "diverge", "divergem", "diferem", "diferente", "diferentes",
    "incompativeis", "insuficiente", "insuficientes", "insuficiencia",
    "padronizada", "padronizado", "diferenca", "balanco", "desbalanceado",
    "falhou", "teste", "nivel", "auxiliar", "primeira", "diferença",
    "diferenciamento", "regressores", "resíduos", "residuos", "resíduo",
    "residuo", "sobreidentificação", "sobreident",
}

PT_PATTERN = re.compile(
    r"(?:^|\b)(" + "|".join(map(re.escape, PT_WORDS)) + r")(?=\b|$)",
    re.IGNORECASE,
)

# Placeholder token for things that must survive translation literally.
PLACEHOLDER_PREFIX = "{@PRES"
PLACEHOLDER_RE = re.compile(r"\{@PRES\d+@\}")

# Skip comments that look like they contain math/formulas.  These are too
# risky for automated translation.
MATH_RE = re.compile(
    r"[α-ωΑ-Ω∆∑∏∫√±∞≠≤≥×÷\^_$\u2070-\u209C\u00B2\u00B3\u2074]|"
    r"\\(frac|sum|sqrt|alpha|beta|gamma|delta|rho|sigma|tau|lambda|mu|pi|"
    r"theta|epsilon|hat|bar|tilde|vec|mathbf|boldsymbol|left|right|begin|end)|"
    r"\{[^{}]+\}/[^{}]+|H[₀₁₂₃₄]|R²|d/d|∂"
)


def needs_translation(text: str) -> bool:
    return bool(DIACRITIC_RE.search(text) or PT_PATTERN.search(text))


# Things to protect inside a string literal or comment.
TO_PROTECT = [
    # Format/placeholder patterns
    (r"\{\{|\}\}", "BRACE"),
    (r"\{[^}\n]+?\}", "PLACE"),
    # Greek/statistical symbols and subscripts (base + optional combining marks)
    (r"(?:[α-ωΑ-Ωḁ-ỿ][\u0300-\u036f]*)|\bR²\b|\bH₀\b|\bp̂\b|\bρ̂\b|\bσ̂\b|\bβ̂\b|\bγ̂\b|\bδ̂\b|\bτ̂\b|\bμ̂\b", "GREEK"),
    # Common abbreviations / notation
    (r"\b(?:SE|HC\d?|PCSE|IMR|OLS|GLS|WLS|MLE|GMM|VAR|SVAR|VECM|DFM|LPM|ATT|ATE|LATE|SMD|RMSPE|ADF|KPSS|MSE|RMSE|SSE|SSR|SST|RSS|LL|LR|AIC|BIC|HQ|NaN|Inf|i\.i\.d|iid)\b", "ABBR"),
    # Standalone emoji / symbols used in output
    (r"[⚠✓✗✘✔]", "SYM"),
    # Variable/subscript patterns like y_{t-1}, y_i, x_i
    (r"[a-zA-Z]_(?:\{[^}]+\}|[a-zA-Z0-9]+)", "SUB"),
    # Math expressions like V̂_p, σ̂_ε, or plain β̂
    (r"[A-Za-z][\u0300-\u036f]*_(?:\{[^}]+\}|[A-Za-z0-9]+)", "MATH"),
    # Box-drawing / horizontal rules (with optional leading newline)
    (r"\n[\u2500-\u257F]{2,}|\n══+|\n──+|\n—+", "DECO"),
    # Horizontal rules without leading newline
    (r"[\u2500-\u257F]{2,}|══+|──+|—+", "DECO"),
    # File paths / URLs / extensions
    (r"(?:https?://\S+|(?:[A-Za-z_][A-Za-z0-9_/\-]*\.(?:dta|csv|xlsx?|xls|db|json|tsv|txt|tex|pdf|py|rs|hay|svg|png|jpg|jpeg|gif|lock)))", "PATH"),
    # Email-ish / version strings
    (r"\b\d+\.\d+\.\d+(?:[-+.]\w+)*\b", "VER"),
]


def protect_text(text: str) -> tuple[str, dict[str, str]]:
    mapping: dict[str, str] = {}
    counter = 0

    def make_token(kind: str) -> str:
        nonlocal counter
        tok = f"{{@PRES{counter:04d}@}}"
        counter += 1
        return tok

    for pattern, kind in TO_PROTECT:

        def repl(m: re.Match) -> str:
            tok = make_token(kind)
            mapping[tok] = m.group(0)
            return tok

        text = re.sub(pattern, repl, text)
    return text, mapping


def restore_placeholders(text: str, mapping: dict[str, str]) -> str:
    # Sort by placeholder number descending so outer (later) placeholders are
    # restored before inner (earlier) ones; then repeat until none remain.
    def token_id(tok: str) -> int:
        m = re.search(r"\d+", tok)
        return int(m.group(0)) if m else -1

    keys = sorted(mapping, key=token_id, reverse=True)
    for _ in range(10):
        if "{@PRES" not in text:
            break
        for tok in keys:
            text = text.replace(tok, mapping[tok])
    return text


def translate_text(text: str) -> str:
    if not text.strip():
        return text
    protected, mapping = protect_text(text)
    try:
        translated = argostranslate.translate.translate(protected, "pt", "en")
    except Exception as e:
        print(f"  argos fail: {protected[:60]!r} -> {e}", file=sys.stderr)
        return text
    if not translated.strip():
        # argos can return an empty string on symbol-heavy input; fall back.
        print(f"  argos empty for: {protected[:60]!r}", file=sys.stderr)
        return text
    return restore_placeholders(translated, mapping)


# Post-translation domain fixes.  Order matters: longer phrases first.
POST_FIXES = [
    ("Default error", "Standard error"),
    ("default error", "standard error"),
    ("Default errors", "Standard errors"),
    ("default errors", "standard errors"),
    ("Returners", "Regressors"),
    ("returners", "regressors"),
    ("Returner", "Regressor"),
    ("returner", "regressor"),
    ("precise T", "requires T"),
    ("precise", "requires"),
    ("first-differented", "first-differenced"),
    ("equation residues", "equation residuals"),
    ("Result equation", "Outcome equation"),
    ("result equation", "outcome equation"),
    ("Equation of result", "Outcome equation"),
    ("equations of result", "outcome equations"),
    ("Selectioning", "Selection"),
    ("selectioning", "selection"),
    ("Covariate variable", "Covariate"),
    ("covariate variable", "covariate"),
    ("Post treatment", "Post-treatment"),
    ("post treatment", "post-treatment"),
    ("Pre treatment", "Pre-treatment"),
    ("pre treatment", "pre-treatment"),
    ("Period not found in the index", "Period not found in index"),
    ("Entity ID not found", "Entity ID not found"),
    ("Control synthetic", "Synthetic control"),
    ("control synthetic", "synthetic control"),
    ("Standardised difference", "Standardized difference"),
    ("standardised difference", "standardized difference"),
    ("Treated unit not found", "Treated unit not found"),
    ("Treatment unit not found", "Treated unit not found"),
    ("not found in", "not found in"),
    ("Only", "Only"),
    ("only", "only"),
    ("No variant return in time found", "No time-varying regressor found"),
    ("no variant return in time found", "no time-varying regressor found"),
    ("No column of augmentation with variance", "No augmented column with variance"),
    ("no column of augmentation with variance", "no augmented column with variance"),
    ("columns of the added model", "columns of the augmented model"),
    ("T very large number of comments", "T too large relative to the number of observations"),
    ("T very large", "T too large"),
    ("number of comments", "number of observations"),
    ("FD waste", "FD residuals"),
    ("waste", "residuals"),
    ("Unique matrix", "Singular matrix"),
    ("unique matrix", "singular matrix"),
    ("All regressors have become zero after differentiation", "All regressors became zero after differencing"),
    ("all regressors have become zero after differentiation", "all regressors became zero after differencing"),
    ("Differentiation", "Differencing"),
    ("differentiation", "differencing"),
    ("Few observations", "Too few observations"),
    ("necessary T", "need T"),
    ("(necessary", "(need"),
    ("Non-restricted", "Unrestricted"),
    ("non-restricted", "unrestricted"),
    ("return", "regressor"),
    ("Return", "Regressor"),
    ("variable in time", "time-varying variable"),
    ("Variant", "Time-varying"),
    ("variant", "time-varying"),
]


def apply_post_fixes(text: str) -> str:
    for bad, good in POST_FIXES:
        text = text.replace(bad, good)
    return text


# Rust string literal extraction ------------------------------------------------


def extract_string_literals(src: str):
    """Yield (start, end, text) for string literal content.

    Handles plain double-quoted strings and line continuations.
    Raw strings r#"..."# are handled in a basic way.
    """
    i = 0
    n = len(src)
    while i < n:
        # raw string r#"..."# and r##"..."##
        m = re.match(r'r(#*)"', src[i:])
        if m:
            start = i
            i += m.end()
            close = '"' + m.group(1) * len(m.group(1))
            end = src.find(close, i)
            if end == -1:
                break
            yield (start + m.end(), end, src[i:end])
            i = end + len(close)
            continue

        if src[i] == '"':
            start = i
            j = i + 1
            content = []
            while j < n:
                ch = src[j]
                if ch == '\\' and j + 1 < n:
                    nxt = src[j + 1]
                    if nxt == '\n':
                        # line continuation: skip backslash, newline and leading whitespace of next line
                        j += 2
                        while j < n and src[j] in ' \t':
                            j += 1
                        continue
                    # decode the escape sequence
                    if nxt == '\\':
                        content.append('\\')
                    elif nxt == '"':
                        content.append('"')
                    elif nxt == 'n':
                        content.append('\n')
                    elif nxt == 'r':
                        content.append('\r')
                    elif nxt == 't':
                        content.append('\t')
                    elif nxt == '0':
                        content.append('\0')
                    elif nxt == "'":
                        content.append("'")
                    else:
                        # unknown escape (e.g. \u{...}, \x..); keep literal
                        content.append('\\')
                        content.append(nxt)
                    j += 2
                elif ch == '"':
                    break
                else:
                    content.append(ch)
                    j += 1
            text = ''.join(content)
            yield (i + 1, j, text)
            i = j + 1
            continue

        i += 1


def in_output_context(src: str, start: int, end: int) -> bool:
    """Heuristic: is the string at this location likely user-facing output or error?"""
    # Look at the source around the opening quote (a few characters before).
    preceding = src[max(0, start - 120):start]
    # Strip leading whitespace/newlines
    last_line = preceding.splitlines()[-1] if preceding else ''
    stripped = last_line.lstrip()
    if stripped.startswith('//') or stripped.startswith('*') or stripped.startswith('/*'):
        return False
    for marker in ('writeln!(', 'write!(', 'format!(', 'println!(', 'print!(', 'panic!(',
                   'bail!(', 'anyhow::bail!(', 'return Err(', '.map_err(', '.ok_or('):
        if marker in preceding:
            return True
    # .into() and .to_string() patterns often are error messages, but not
    # struct field initializers like `model_type: "sem".into()`.
    tail = src[end:min(len(src), end + 120)]
    if re.search(r'"\s*\.\s*(into|to_string)\s*\(', tail):
        if not re.search(r'[\w_]+\s*:\s*"?$', preceding[-60:]):
            return True
    return False


def translate_file(path: Path) -> int:
    src = path.read_text(encoding='utf-8')
    # First, line comments
    lines = src.splitlines(keepends=True)
    changed = 0

    def get_ending(line: str) -> str:
        if line.endswith('\r\n'):
            return '\r\n'
        if line.endswith('\n'):
            return '\n'
        if line.endswith('\r'):
            return '\r'
        return ''

    # Comments are translated only if they appear to be plain prose (no math).
    new_lines = []
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith('//'):
            prefix_len = len(line) - len(stripped)
            prefix = line[:prefix_len + 2]
            text = line[prefix_len + 2:].rstrip('\n\r')
            ending = get_ending(line[prefix_len + 2:])
            if needs_translation(text) and not MATH_RE.search(text):
                new_text = apply_post_fixes(translate_text(text))
                if new_text != text:
                    changed += 1
                    line = prefix + new_text + ending
        else:
            # inline comment
            idx = find_unquoted_double_slash(line)
            if idx != -1:
                code = line[:idx]
                comment = line[idx:]
                text = comment[2:].rstrip('\n\r')
                ending = get_ending(comment[2:])
                if needs_translation(text) and not MATH_RE.search(text):
                    new_text = apply_post_fixes(translate_text(text))
                    if new_text != text:
                        changed += 1
                        line = code + '//' + new_text + ending
        new_lines.append(line)
    src = ''.join(new_lines)

    # Process string literals
    # We need to process from right to left so replacements don't shift indices.
    segments = []
    for start, end, text in extract_string_literals(src):
        if not needs_translation(text):
            continue
        if not in_output_context(src, start, end):
            continue
        new_text = apply_post_fixes(translate_text(text))
        if new_text != text:
            # Escape double quotes, backslashes and physical line breaks for
            # re-insertion into a Rust double-quoted string literal.
            safe = (
                new_text
                .replace('\\', '\\\\')
                .replace('\n', '\\n')
                .replace('\r', '\\r')
                .replace('\t', '\\t')
                .replace('"', '\\"')
            )
            segments.append((start, end, safe))

    if not segments:
        if not changed:
            return 0
        path.write_text(src, encoding='utf-8')
        return changed

    segments.sort(key=lambda x: x[0], reverse=True)
    for start, end, safe in segments:
        src = src[:start] + safe + src[end:]
    path.write_text(src, encoding='utf-8')
    return changed + len(segments)


def find_unquoted_double_slash(line: str) -> int:
    in_str = False
    esc = False
    for i, ch in enumerate(line):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            esc = False
            continue
        if ch == '/' and i + 1 < len(line) and line[i + 1] == '/':
            return i
    return -1


def main():
    root = Path(__file__).resolve().parent / 'src'
    files = sorted(root.rglob('*.rs'))
    total = 0
    for path in files:
        try:
            n = translate_file(path)
        except Exception as e:
            print(f'ERROR {path}: {e}', file=sys.stderr)
            continue
        if n:
            print(f'  {path.name}: {n} changes')
            total += n
    print(f'\nTotal changed segments: {total}')


if __name__ == '__main__':
    main()

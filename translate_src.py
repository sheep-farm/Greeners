#!/usr/bin/env python3
"""Translate Portuguese comments and string literals in Greeners Rust source.

This script is intentionally conservative:
- It only touches lines that contain Portuguese characters or common PT words.
- String literals are translated by argostranslate (fallback to original).
- Comment lines are translated and then passed through a small cleanup pass.
- Placeholders like {foo}, {{, }}, and Rust format specifiers are preserved.
"""

import re
import sys
from pathlib import Path

import argostranslate.translate

PT_WORDS = {
    "não", "nao", "é", "são", "para", "com", "por", "se", "como", "mas", "ou",
    "mais", "menos", "bem", "muito", "pouco", "tudo", "nada", "agora", "antes",
    "depois", "aqui", "ali", "assim", "mesmo", "tal", "quando", "onde", "porque",
    "quem", "qual", "todos", "está", "estao", "estão", "ser", "estar", "tem",
    "tinha", "fazer", "faz", "fez", "vai", "ir", "pode", "deve", "esse", "essa",
    "isso", "aquilo", "nosso", "seu", "sua", "meu", "minha", "teu", "esta", "este",
    "aquele", "aquela", "outro", "outra", "também", "já", "ainda", "só", "nem",
    "então", "cada", "sobre", "entre", "durante", "desde", "até", "após", "sem",
    "contra", "debaixo", "dentro", "fora", "acima", "abaixo", "perto", "longe",
    "direita", "esquerda", "frente", "trás", "lugar", "parte", "coisa", "pessoa",
    "tempo", "vez", "ano", "dia", "hoje", "ontem", "amanhã", "semana", "mês",
    "hora", "minuto", "segundo", "momento", "início", "fim", "começo", "meio",
    "ponto", "lado", "forma", "modo", "ordem", "caso", "exemplo", "razão", "ideia",
    "palavra", "nome", "número", "grupo", "tipo", "classe", "mundo", "país",
    "cidade", "estado", "região", "área", "setor", "mercado", "empresa", "trabalho",
    "emprego", "dinheiro", "custo", "preço", "valor", "quantidade", "total",
    "média", "mínimo", "máximo", "soma", "produto", "resultado", "efeito", "causa",
    "problema", "solução", "sistema", "processo", "método", "modelo", "variável",
    "coeficiente", "constante", "termo", "equação", "função", "dado", "conjunto",
    "base", "tabela", "gráfico", "figura", "imagem", "relatório", "texto", "linha",
    "coluna", "campo", "registro", "variaveis", "coeficientes", "funcao", "equacao",
    "grafico", "tabela", "relatorio", "imagem", "numero", "observacoes", "variavel",
    "equacao", "resultado", "tratamento", "selecao", "seleção", "covariavel",
    "sintetico", "sintético", "controle", "periodo", "período", "doador", "tratada",
    "censurada", "verossimilhanca", "assintotica", "assintótica", "padrao", "padrão",
    "depois", "antes", "media", "média", "mediana", "desvio", "minimo", "mínimo",
    "maximo", "máximo", "primeiro", "segundo", "terceiro", "quarto", "quinta",
    "sexta", "septimo", "oitavo", "nono", "decimo", "ultimo", "último",
}

PT_PATTERN = re.compile(
    r"(?:^|\b)(" + "|".join(map(re.escape, PT_WORDS)) + r")(?=\b|$)",
    re.IGNORECASE,
)
DIACRITIC_RE = re.compile(r"[áàãâéêíóôõúçÁÀÃÂÉÊÍÓÔÕÚÇ]")


def contains_portuguese(text: str) -> bool:
    return DIACRITIC_RE.search(text) is not None or PT_PATTERN.search(text) is not None


# Preserve format placeholders and similar braces while translating.
PH_RE = re.compile(r"(\{\{)|(\}\})|(\{[^\}\n]+?\})")


def translate_one(text: str) -> str:
    if not text.strip():
        return text
    # Protect braces / placeholders
    protected: dict[str, str] = {}
    counter = [0]

    def repl(m: re.Match) -> str:
        ph = f"__B{counter[0]:04d}__"
        counter[0] += 1
        protected[ph] = m.group(0)
        return ph

    protected_text = PH_RE.sub(repl, text)
    try:
        translated = argostranslate.translate.translate(protected_text, "pt", "en")
    except Exception as e:
        print(f"  argos fail: {protected_text[:60]!r} -> {e}", file=sys.stderr)
        translated = protected_text

    # Restore placeholders
    for ph in sorted(protected, key=len, reverse=True):
        translated = translated.replace(ph, protected[ph])

    # Post-fix a few awkward argos choices for our domain
    fixes = {
        "Variable coefficient": "Variable",  # argos may translate header "Variável" weirdly
        "Result equation": "Outcome equation",
        "Selection equation": "Selection equation",  # keep if already good
        "Synthetic Control": "Synthetic Control",
        "treated unit not found": "treated unit not found",
        "pre treatment period": "pre-treatment period",
        "post treatment period": "post-treatment period",
        "pre-treatment (s)": "pre-treatment period(s)",
        "post-treatment (s)": "post-treatment period(s)",
        "Covariate variable": "Covariate",
        "Standardized difference": "standardized difference",
    }
    for bad, good in fixes.items():
        if bad in translated and bad != good:
            translated = translated.replace(bad, good)

    return translated


def extract_string_literals(content: str):
    """Yields (start, end, text, quote) for double-quoted string literals.

    Handles raw strings r"...", r#"..."#, and r##"..."##.
    This is not a full Rust parser but is sufficient for the literal output
    strings used in this codebase (plain or r#"..."#).
    """
    i = 0
    n = len(content)
    while i < n:
        # Look for a string start
        # Plain " or r#"..., r##"... etc.
        m = re.match(r'r(#*)"', content[i:])
        if m:
            hash_count = len(m.group(1))
            start = i
            i += m.end()
            close = '"' + ("#" * hash_count)
            end = content.find(close, i)
            if end == -1:
                break
            text = content[i:end]
            yield (start + m.end(), end, text, 'r')
            i = end + len(close)
        elif content[i] == '"':
            start = i
            i += 1
            # Read until unescaped "
            j = i
            while j < n:
                ch = content[j]
                if ch == '\\' and j + 1 < n:
                    j += 2
                elif ch == '"':
                    break
                else:
                    j += 1
            text = content[i:j]
            yield (i, j, text, '"')
            i = j + 1
        else:
            i += 1


def is_output_string_context(content: str, start: int) -> bool:
    """Heuristic: is the string at `start` likely a user-facing or error string?"""
    preceding = content[:start]
    # Skip doc comments / normal comments that happen to contain quotes
    # Find the most recent line start
    last_newline = preceding.rfind("\n")
    line_start = preceding[last_newline + 1 :]
    stripped = line_start.lstrip()
    if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
        return False
    # Check if the call site is one of the common output macros
    for macro in ("writeln!(", "write!(", "format!(", "println!(", "print!(", "panic!(", "bail!(", "anyhow::bail!(", "return Err("):
        if macro in preceding[-120:]:
            return True
    # Also .into() pattern: "...".into() is often an error message
    after_end = content[start + len(content[start:].split('"', 1)[0]):start + 120]
    # crude: if a string is followed by .into() or .to_string(), it's likely an error
    if re.search(r'"\s*\.\s*(into|to_string)\s*\(', after_end):
        return True
    return False


def translate_file(path: Path) -> int:
    content = path.read_text(encoding="utf-8")
    # First pass: comments (// and /// and /* */)
    # We'll use a line-oriented approach for // comments, plus block-comment aware.
    lines = content.splitlines(keepends=True)
    changed = 0

    def translate_line_comment(line: str, marker: str) -> str:
        nonlocal changed
        prefix = line[: len(line) - len(line.lstrip())] + marker
        text = line[len(prefix):].rstrip("\n\r")
        if not text.strip():
            return line
        if not contains_portuguese(text):
            return line
        new_text = translate_one(text)
        if new_text != text:
            changed += 1
            return prefix + new_text + get_ending(line)
        return line

    def get_ending(line: str) -> str:
        if line.endswith("\r\n"):
            return "\r\n"
        if line.endswith("\n"):
            return "\n"
        if line.endswith("\r"):
            return "\r"
        return ""

    new_lines = []
    in_block = False
    block_buffer = []
    block_start_marker = ""
    for line in lines:
        if in_block:
            # Look for */ to end block
            end_idx = line.find("*/")
            if end_idx != -1:
                block_buffer.append(line[: end_idx + 2])
                # translate the whole block comment
                full = "".join(block_buffer)
                # preserve leading whitespace and /* */ markers roughly
                # We only translate the inner text: the part between /* and */
                m = re.match(r"^(\s*)(/\*)(.*)(\*/)(.*)$", full, re.DOTALL)
                if m:
                    ws, open_m, inner, close_m, rest = m.groups()
                    if contains_portuguese(inner):
                        new_inner = translate_one(inner)
                        if new_inner != inner:
                            changed += 1
                            full = ws + open_m + new_inner + close_m + rest
                new_lines.append(full)
                new_lines.append(line[end_idx + 2:])
                in_block = False
                block_buffer = []
                continue
            else:
                block_buffer.append(line)
                continue

        # start of block comment
        m = re.match(r"^(\s*/\*)", line)
        if m:
            # Does it end on same line?
            end_idx = line.find("*/")
            if end_idx != -1:
                new_lines.append(line)
                continue
            in_block = True
            block_buffer = [line]
            continue

        # line comments
        stripped = line.lstrip()
        if stripped.startswith("//"):
            new_lines.append(translate_line_comment(line, "//"))
        else:
            # inline // after code
            idx = find_unquoted(line, "//")
            if idx != -1:
                code = line[:idx]
                comment = line[idx:]
                new_comment = translate_line_comment(comment, "//")
                if new_comment != comment:
                    changed += 1
                new_lines.append(code + new_comment)
            else:
                new_lines.append(line)

    content = "".join(new_lines)

    # Second pass: string literals
    # We only translate string literals that contain Portuguese and are in an
    # output or error context.
    replacements = []
    for start, end, text, quote_type in extract_string_literals(content):
        if not contains_portuguese(text):
            continue
        if not is_output_string_context(content, start):
            continue
        new_text = translate_one(text)
        if new_text != text:
            replacements.append((start, end, text, new_text, quote_type))

    # Apply replacements from right to left
    if replacements:
        replacements.sort(key=lambda x: x[0], reverse=True)
        for start, end, old, new, quote_type in replacements:
            before = content[:start]
            after = content[end:]
            if quote_type == "r":
                # raw string: re-encode unchanged
                content = before + new + after
            else:
                # Plain "...": must escape any new double-quotes introduced.
                # Also handle backslashes? Argos shouldn't introduce them.
                safe_new = new.replace("\\", "\\\\").replace('"', '\\"')
                content = before + safe_new + after
        changed += len(replacements)

    if changed:
        path.write_text(content, encoding="utf-8")
    return changed


def find_unquoted(s: str, target: str) -> int:
    """Find the first occurrence of `target` outside of double-quoted strings."""
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
                continue
            if ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            esc = False
            continue
        if s.startswith(target, i):
            return i
    return -1


def main():
    root = Path(__file__).resolve().parent / "src"
    files = sorted(root.rglob("*.rs"))
    total = 0
    for path in files:
        try:
            n = translate_file(path)
        except Exception as e:
            print(f"ERROR {path}: {e}", file=sys.stderr)
            continue
        if n:
            print(f"  {path.name}: {n} changes")
        total += n
    print(f"\nTotal changed strings/comments: {total}")


if __name__ == "__main__":
    main()

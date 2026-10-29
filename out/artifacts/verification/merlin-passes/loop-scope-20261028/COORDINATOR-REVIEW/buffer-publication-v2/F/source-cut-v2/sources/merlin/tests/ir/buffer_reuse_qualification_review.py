"""Structural comparison for the controlled generated Lean command format.

This host regression guard does NOT prove Lean syntax/semantics or authorize a
check. It preserves all encoded definitions, old theorem headers and command
context while permitting proof-body repairs. Actual new proofs still run Lean.
Comments nest; strings/quoted identifiers cannot manufacture commands.
"""


def tokens(source):
    result = []
    i, column, first_code = 0, 0, True

    def advance(end):
        nonlocal i, column, first_code
        text = source[i:end]
        column = len(text.rsplit("\n", 1)[-1]) if "\n" in text else column + len(text)
        if "\n" in text:
            first_code = True
        i = end

    while i < len(source):
        if source[i].isspace():
            advance(i + 1)
            continue
        if source.startswith("--", i):
            end = source.find("\n", i)
            advance(len(source) if end < 0 else end)
            continue
        if source.startswith("/-", i):
            end, depth = i + 2, 1
            while end < len(source) and depth:
                if source.startswith("/-", end):
                    depth += 1
                    end += 2
                elif source.startswith("-/", end):
                    depth -= 1
                    end += 2
                else:
                    end += 1
            if depth:
                raise ValueError("unterminated nested Lean comment")
            advance(end)
            continue
        start, col = i, column
        if source[i] == '"':
            end = i + 1
            while end < len(source):
                if source[end] == "\\":
                    end += 2
                elif source[end] == '"':
                    end += 1
                    break
                else:
                    end += 1
            else:
                raise ValueError("unterminated Lean string")
        elif source[i] == "«":
            end = source.find("»", i + 1)
            if end < 0:
                raise ValueError("unterminated quoted identifier")
            end += 1
        elif source[i].isalnum() or source[i] == "_":
            end = i + 1
            while end < len(source) and (source[end].isalnum() or source[end] in "_.'"):
                end += 1
        else:
            end = i + (2 if source.startswith(":=", i) else 1)
        result.append((source[start:end], col, first_code))
        advance(end)
        first_code = False
    return result


def inventory(source):
    words = tokens(source)
    commands = {"import", "open", "set_option", "namespace", "section", "end", "def", "theorem"}
    forbidden = {
        "axiom",
        "unsafe",
        "opaque",
        "abbrev",
        "instance",
        "structure",
        "inductive",
        "notation",
        "macro",
        "syntax",
        "attribute",
        "example",
        "noncomputable",
        "private",
        "protected",
        "partial",
        "initialize",
    }
    if any(first and column != 0 and word in commands | forbidden for word, column, first in words):
        raise ValueError("all generated commands must begin at column zero, including after comments")
    starts, depth = [], 0
    for index, (word, column, _) in enumerate(words):
        if column == 0 and depth == 0:
            starts.append(index)
        if word in ("(", "[", "{", "⟨"):
            depth += 1
        elif word in (")", "]", "}", "⟩"):
            depth -= 1
        if depth < 0:
            raise ValueError("unbalanced generated source delimiters")
    if depth:
        raise ValueError("unclosed generated source delimiters")
    if words and (not starts or starts[0] != 0):
        raise ValueError("unaccounted source before first controlled command")
    if any(words[i][0] not in commands for i in starts):
        raise ValueError("unrecognized column-zero command in controlled format")
    starts.append(len(words))
    scopes, definitions, theorems, context, layout = [], {}, {}, [], []
    for begin, end in zip(starts, starts[1:]):
        command = [word for word, _, _ in words[begin:end]]
        tag = command[0]
        if tag not in ("def", "theorem"):
            layout.append(("context", command))
        if tag in ("namespace", "section"):
            if len(command) != 2:
                raise ValueError("named exact scope required")
            scopes.append((tag, command[1]))
            context.append(command)
        elif tag == "end":
            if len(command) != 2 or not scopes or command[1] != scopes[-1][1]:
                raise ValueError("exact scope closure required")
            scopes.pop()
            context.append(command)
        elif tag in ("def", "theorem"):
            if len(command) < 4:
                raise ValueError("incomplete declaration")
            name = ".".join([name for kind, name in scopes if kind == "namespace"] + [command[1]])
            layout.append((tag, name))
            target = definitions if tag == "def" else theorems
            if name in definitions or name in theorems:
                raise ValueError("duplicate declaration")
            if tag == "def":
                target[name] = command
            else:
                depth, assignment = 0, None
                for index, token in enumerate(command):
                    if token in ("(", "[", "{", "⟨"):
                        depth += 1
                    elif token in (")", "]", "}", "⟩"):
                        depth -= 1
                    elif token == ":=" and depth == 0:
                        assignment = index
                        break
                    if depth < 0:
                        raise ValueError("unbalanced declaration header")
                if assignment is None:
                    raise ValueError("closed generated theorem body required")
                target[name] = command[:assignment]
        else:
            context.append(command)
    if scopes:
        raise ValueError("unclosed generated scope")
    return {"definitions": definitions, "theorem_headers": theorems, "command_context": context, "layout": layout}


def compare(old, new):
    before, after = inventory(old), inventory(new)
    if before["definitions"] != after["definitions"]:
        raise ValueError("encoded semantic definition changed")
    if before["command_context"] != after["command_context"]:
        raise ValueError("import/open/options/scope context changed")
    if any(after["theorem_headers"].get(name) != header for name, header in before["theorem_headers"].items()):
        raise ValueError("original theorem statement changed or dropped")
    added = set(after["theorem_headers"]) - set(before["theorem_headers"])
    if before["layout"] != [row for row in after["layout"] if not (row[0] == "theorem" and row[1] in added)]:
        raise ValueError("original declaration/context order changed")
    return {
        "definitions": len(before["definitions"]),
        "original_theorem_headers": len(before["theorem_headers"]),
        "new_helper_theorems": sorted(added),
        "proofs_executed": False,
        "authority": False,
    }

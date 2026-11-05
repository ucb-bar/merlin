#!/usr/bin/env python3
"""Expand FIRRTL's indentation shorthand for parsers lacking `else when`.

Only indentation and the explicit nested when spelling change. No expressions,
constants, memories or timing behavior are rewritten. A fresh FIRRTL parse and
independent simulator comparison remain necessary after this compatibility step.
"""
import argparse
from pathlib import Path


def expand(text):
    scopes, result = [], []
    label_index = 0
    for line in text.splitlines():
        stripped = line.lstrip(' ')
        if not stripped or stripped.startswith(';'):
            result.append(line)
            continue
        if '\t' in line[:len(line)-len(line.lstrip())]:
            raise ValueError('tab indentation is unsupported')
        indent = len(line)-len(stripped)
        if stripped.startswith(('assert(', 'assume(', 'cover(', 'printf(', 'stop(')):
            # Verification labels are diagnostic names, not values.
            statement, marker, suffix = stripped.rpartition(') : ')
            if marker:
                label, space, info = suffix.partition(' ')
                if not label.isidentifier():
                    raise ValueError('invalid verification label')
                stripped = statement + marker + label + '_boundary_' + str(label_index) + space + info
                label_index += 1
        is_else = stripped.startswith('else ')
        while scopes and (indent < scopes[-1] or indent == scopes[-1] and not is_else):
            scopes.pop()
        offset = 2*len(scopes)
        if stripped.startswith('else when '):
            result.append(' '*(indent+offset)+'else :')
            result.append(' '*(indent+offset+2)+stripped[len('else '):])
            scopes.append(indent)
        else:
            result.append(' '*(indent+offset)+stripped)
    return '\n'.join(result)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input');parser.add_argument('output')
    args=parser.parse_args()
    Path(args.output).write_text(expand(Path(args.input).read_text()))


if __name__ == '__main__': main()

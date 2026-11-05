"""Read explicit LLVM memory effects of plain floating scalar declarations.

No callee name, numerical law or saved report supplies this premise. The normal
compiler/assembler owns validity and implementation correspondence of the input
LLVM. Unknown declarations, pointer arguments and indirect calls stay opaque.
"""

from .late_quant_rne import _tokens


def memory_none_float_callees(source):
    """Return only unique plain scalar prototypes with literal memory(none)."""
    groups, declarations, repeated = {}, {}, set()
    for line in source.splitlines():
        stripped = line.lstrip()
        if not stripped.startswith(('attributes ', 'declare ', 'define ')):
            continue
        words = tuple(token.text for token in _tokens(line))
        if words[:1] == ('attributes',) and len(words) >= 6 and words[2:4] == ('=', '{') and words[-1] == '}':
            name = words[1]
            if name in groups:
                groups[name] = None
            else:
                groups[name] = words[4:-1]
        elif words[:1] in (('declare',), ('define',)):
            names = [word for word in words if word.startswith('@')]
            if not names:
                continue
            name = names[0]
            if name in declarations:
                repeated.add(name)
            declarations[name] = words
    accepted = set()
    scalar = {'half', 'float', 'double', 'fp128', 'bfloat'}
    for name, words in declarations.items():
        if (name in repeated or len(words) < 5 or words[0] != 'declare'
                or words[1] not in scalar or words[2:4] != (name, '(') or ')' not in words[4:]):
            continue
        close = words.index(')', 4)
        parameters = words[4:close]
        if parameters and (len(parameters) % 2 != 1
                or any(word not in scalar for word in parameters[::2])
                or any(word != ',' for word in parameters[1::2])):
            continue
        attrs, supported = [], True
        for word in words[close + 1:]:
            if word.startswith('#'):
                if word not in groups or groups[word] is None:
                    supported = False
                    break
                attrs.extend(groups[word])
            else:
                attrs.append(word)
        if supported and attrs.count('memory') == 1:
            at = attrs.index('memory')
            if attrs[at:at + 4] == ['memory', '(', 'none', ')']:
                accepted.add(name)
    return frozenset(accepted)

"""Embed policy-authenticated unary binary32 supplier sources in C units."""


def _word(character):
    return character.isalnum() or character == '_'


def _typedef_names(source):
    """Names declared by single-line file-scope typedefs: `typedef ... NAME;` at the start of a line."""
    names = set()
    for line in source.split('\n'):
        body = line.rstrip(' \t')
        if not body.startswith('typedef') or not body.endswith(';') or (len(body) > 7 and _word(body[7])):
            continue
        head = body[:-1]
        start = len(head)
        while start > 7 and _word(head[start - 1]):
            start -= 1
        if start < len(head) and (start == 7 or not _word(head[start - 1])):
            names.add(head[start:])
    return sorted(names)


def embedded_host_math(host_math, functions):
    """Policy-pinned CORE-MATH sources for one translation unit, with canonical-NaN wrappers.

    Equivalent to the archive build: the support header precedes each unmodified
    source, whose entry gets internal linkage. Its macros are undefined after each
    source, so later code in the unit keeps the real fenv and builtin names.
    """
    policy = host_math.load_policy(required=functions)
    support = (host_math.ROOT / 'supplier_support.h').read_text()
    directives = '\n'.join(line for line in support.splitlines() if line.startswith('#'))
    undefine = '\n'.join('#undef ' + line.split()[1]
                         for line in support.splitlines() if line.startswith('#define '))
    parts = ['#ifndef MERLIN_EMBEDDED_HOST_MATH_SUPPORT\n#define MERLIN_EMBEDDED_HOST_MATH_SUPPORT\n'
             + support + undefine + '\n#endif']
    for function in functions:
        spec = policy['functions'][function]
        if function in host_math.FUNCTIONS64 or host_math.arity(function) != 1 or function == 'sqrtf':
            raise ValueError('embedded host math supports unary binary32 CORE-MATH functions only')
        source = (host_math.ROOT / 'third_party/core-math' / spec['source']).read_text()
        guard = 'MERLIN_EMBEDDED_' + spec['symbol'].upper()
        # Upstream sources repeat file-scope typedef names; one unit may embed several.
        types = _typedef_names(source)
        rename = '\n'.join(f'#define {name} {guard}_{name}' for name in types)
        restore = '\n'.join(f'#undef {name}' for name in types)
        parts.append(f'''#ifndef {guard}
#define {guard}
{directives}
{rename}
static float {spec['implementation']}(float);
{source}
{restore}
{undefine}
static float {spec['symbol']}(float x) {{
    float r={spec['implementation']}(x); uint32_t u; memcpy(&u,&r,4);
    if ((u&0x7fffffffU)>0x7f800000U) {{ u=0x7fc00000U; memcpy(&r,&u,4); }}
    return r;
}}
#endif''')
    return '\n'.join(parts), policy

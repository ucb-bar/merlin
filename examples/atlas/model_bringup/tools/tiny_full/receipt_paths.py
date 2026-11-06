"""Portable, checksum-preserving receipt path encoding with explicit caller roots."""
from pathlib import Path
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked


def rewrite(value, translate):
    if isinstance(value, dict):
        return {translate(key): rewrite(item, translate) for key, item in value.items()}
    if isinstance(value, list):
        return [rewrite(item, translate) for item in value]
    return translate(value) if isinstance(value, str) else value


def encode(value, roots):
    ordered = sorted(((name, checked(root)) for name, root in roots.items()), key=lambda row: -len(str(row[1])))
    def translate(text):
        if text.startswith('/'):
            path = checked(text)
            for name, root in ordered:
                if path.is_relative_to(root):
                    return '@' + name + '/' + path.relative_to(root).as_posix()
        return text
    return rewrite(value, translate)


def decode(value, roots, relocations=()):
    roots = {name: checked(root) for name, root in roots.items() if root is not None}
    def translate(text):
        if text.startswith('@'):
            name, separator, member = text[1:].partition('/')
            if not separator or name not in roots or Path(member).is_absolute() or '..' in Path(member).parts:
                raise ValueError('invalid receipt root or member: ' + text)
            return str(checked(roots[name] / member))
        if text.startswith('/'):
            path = checked(text)
            for old, new in relocations:
                if path.is_relative_to(old):
                    return str(checked(new / path.relative_to(old)))
        return text
    return rewrite(value, translate)

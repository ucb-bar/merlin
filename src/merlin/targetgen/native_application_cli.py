"""Lightweight CLI registration for explicitly selected native applications."""

def add_arguments(parser, *, compiling, require_application=None):
    """Register only the implemented staged-application route."""
    parser.add_argument("--application", action="store_true",
                        required=not compiling if require_application is None else require_application,
                        help="compile/replay all selected regions of a staged application")
    parser.add_argument("--application-state", help="source-bound state input/update interface")
    parser.add_argument("--application-parameters", choices=("embedded", "external"), default="embedded")
    if compiling:
        parser.add_argument("--application-name", default="application")
        parser.add_argument("--application-memory-bytes", type=int, help="explicit tensor arena and staging budget")


def verify_command(args):
    from .cli import _write_status
    from .native_application import verify_application_command

    return verify_application_command(args, _write_status)

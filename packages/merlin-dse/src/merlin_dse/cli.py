"""Distribution-owned lazy entrypoints preserving the existing command contracts."""


def search_main():
    from merlin.dse.cli import main

    return main()


def pressure_main():
    from merlin.design_pressure.cli import main

    return main()


def guidance_main():
    from merlin.dse_guidance.cli import main

    return main()

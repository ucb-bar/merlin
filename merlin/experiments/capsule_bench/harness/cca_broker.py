"""Native CLI compatibility launcher for the package-owned CCA broker."""

if __name__ == "__main__":
    from merlin_experiments.phase1.brokers.cca import main

    raise SystemExit(main())

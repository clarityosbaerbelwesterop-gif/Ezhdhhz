"""Donor-model tooling for transplants: provenance, safetensors inventories by byte range, vocabulary
contracts, donor forwards, bridges and expert adaptation. Everything here separates
    direct copy   (bit-exact, SHA-256 must match)
    projection    (deterministic arithmetic with stated tolerance)
    training/heal (not performed by this package; only specified)."""

"""Fixed thresholds for the C2 reproduction claim."""

# MNIST and Fashion-MNIST use (minimum accuracy, maximum BP gap).
# CIFAR10 uses (minimum accuracy, maximum accuracy); c2_summary.py also
# requires its mean accuracy to remain below BP.
C2_CRITERIA = {
    "MNIST": (97.0, 1.0),
    "Fashion-MNIST": (87.0, 1.5),
    "CIFAR10": (35.0, 41.0),
}

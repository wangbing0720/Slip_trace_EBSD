import numpy as np

# ============================================================
# SCHMID FACTOR CALCULATOR FOR TiAl (L1₀ TETRAGONAL STRUCTURE)
# ============================================================
# Lattice parameters for TiAl (L1₀ structure)
a = 4.01867
c = 4.06542


def calculate_schmid_factor(load_dir, slip_plane, slip_dir):
    """Calculate Schmid factor for given crystallographic vectors."""
    # Convert directions to Cartesian
    load_cart = np.array([load_dir[0] * a, load_dir[1] * a, load_dir[2] * c])
    slip_cart = np.array([slip_dir[0] * a, slip_dir[1] * a, slip_dir[2] * c])

    # Plane normal using reciprocal lattice (correct method)
    # Normal direction is proportional to [h/a², k/b², l/c²]
    normal_cart = np.array(
        [
            slip_plane[0] / (a**2),
            slip_plane[1] / (a**2),  # a = b for tetragonal
            slip_plane[2] / (c**2),
        ]
    )

    # Normalize all vectors
    load_norm = load_cart / np.linalg.norm(load_cart)
    normal_norm = normal_cart / np.linalg.norm(normal_cart)
    slip_norm = slip_cart / np.linalg.norm(slip_cart)

    # Check slip direction is in slip plane
    orthogonality = abs(np.dot(normal_norm, slip_norm))
    if orthogonality > 1e-6:
        print(f"Warning: n·b = {orthogonality:.2e}")

    # Calculate angles and Schmid factor
    cos_phi = abs(np.dot(load_norm, normal_norm))
    cos_lambda = abs(np.dot(load_norm, slip_norm))
    m = cos_phi * cos_lambda

    return {
        "schmid_factor": m,
        "cos_phi": cos_phi,
        "cos_lambda": cos_lambda,
        "phi_deg": np.degrees(np.arccos(cos_phi)),
        "lambda_deg": np.degrees(np.arccos(cos_lambda)),
    }


def format_3index(indices, is_plane=False):
    """Format 3-index Miller indices."""

    def fmt(num):
        return f"{-num}̅" if num < 0 else str(num)

    if is_plane:
        return f"({fmt(indices[0])}{fmt(indices[1])}{fmt(indices[2])})"
    return f"[{fmt(indices[0])}{fmt(indices[1])}{fmt(indices[2])}]"


def main():
    print("=" * 50)
    print("TiAl (L1₀) Schmid Factor Calculator")
    print(f"Lattice: a = b = {a:.5f} Å, c = {c:.5f} Å, c/a = {c / a:.4f}")
    print("Index notation: 3-index Miller (h k l) / [u v w]")
    print("=" * 50)

    while True:
        try:
            # Get input
            load_dir = [int(x) for x in input("Load direction [u v w]: ").split()]
            slip_plane = [int(x) for x in input("Slip plane (h k l): ").split()]
            slip_dir = [int(x) for x in input("Slip direction [u v w]: ").split()]

            # Calculate
            results = calculate_schmid_factor(load_dir, slip_plane, slip_dir)

            # Display results
            print("\n" + "-" * 40)
            print("RESULTS:")
            print(f"  Load:   {format_3index(load_dir)}")
            print(f"  Plane:  {format_3index(slip_plane, is_plane=True)}")
            print(f"  Slip:   {format_3index(slip_dir)}")
            print("-" * 40)
            print(
                "  φ = "
                f"{results['phi_deg']:.2f}° (cos φ = {results['cos_phi']:.4f})"
            )
            print(
                "  λ = "
                f"{results['lambda_deg']:.2f}° (cos λ = {results['cos_lambda']:.4f})"
            )
            print(f"  Schmid factor: m = {results['schmid_factor']:.6f}")
            print("-" * 40 + "\n")

        except ValueError:
            print("Error: Please enter three integers separated by spaces.\n")
            continue

        if input("Calculate again? (y/n): ").strip().lower() != "y":
            break

    print("Calculation terminated.")


if __name__ == "__main__":
    main()

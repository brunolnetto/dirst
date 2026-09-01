# dirst

Spatial distance metrics, geodesic curves, and travel-time calculations.

## Features

- Distance metrics: Euclidean, Manhattan, Chebyshev, cosine, Canberra, Bray-Curtis, and p-norm.
- Sphere and geographical distances.
- Geodesic curve solver for parametric surfaces.
- Optional parallelization via rayon.

## Parallel toggle

Default build enables parallel mode.

```bash
cargo test
cargo bench --bench geodesic_bench
```

Disable parallel mode for serial benchmarking:

```bash
cargo test --no-default-features
cargo bench --bench geodesic_bench --no-default-features
```

Enable explicitly:

```bash
cargo test --features parallel
cargo bench --bench geodesic_bench --features parallel
```

## Crates publishing

Tag-based publish is automated by GitHub Actions:

- Create `CARGO_REGISTRY_TOKEN` in repository secrets.
- Push tag `vX.Y.Z`.
- Workflow runs package + dry-run + publish.

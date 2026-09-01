//! Geodesic curves on parametric surfaces.
//!
//! Adaptation of `lab/main.py`: SymPy symbolic differentiation is replaced by
//! numerical central-difference differentiation of the parametric map, and
//! `scipy.integrate.solve_ivp` / `simps` are replaced by RK4 and composite
//! Simpson's rule.  Full generality is preserved – any smooth parametric
//! surface works without a CAS dependency.

#[cfg(feature = "parallel")]
use rayon::prelude::*;

// Step sizes chosen to balance truncation and roundoff for 64-bit floats.
const H_SURF: f64 = 1e-6; // for ∂fₖ/∂xⁱ  (optimal ≈ ε^{1/3} ≈ 6e-6)
const H_METR: f64 = 1e-4; // for ∂g_{ab}/∂xᶜ  (optimal ≈ ε_g^{1/3})
// scipy solve_ivp (RK45 adaptive) uses internal steps far smaller than the
// t_eval spacing.  We match that behaviour with this fixed maximum RK4 step.
const DT_MAX: f64 = 1e-3;

pub const PARALLEL_ENABLED: bool = cfg!(feature = "parallel");

// ── Surface ───────────────────────────────────────────────────────────────────

/// A parametric surface f : R^{`n_params`} → R^{`n_embed`} given by a closure.
pub struct Surface {
    pub n_params: usize,
    pub n_embed: usize,
    f: Box<dyn Fn(&[f64]) -> Vec<f64> + Send + Sync>,
}

impl Surface {
    pub fn new(
        n_params: usize,
        n_embed: usize,
        f: impl Fn(&[f64]) -> Vec<f64> + Send + Sync + 'static,
    ) -> Self {
        Surface { n_params, n_embed, f: Box::new(f) }
    }

    /// Paraboloid: f(u, v) = (u, v, u² + v²)
    pub fn paraboloid() -> Self {
        Surface::new(2, 3, |c| {
            let (u, v) = (c[0], c[1]);
            vec![u, v, u * u + v * v]
        })
    }

    /// Sphere of radius `r`: f(u, v) = r·(cos u · sin v, sin u · sin v, cos v)
    pub fn sphere(r: f64) -> Self {
        Surface::new(2, 3, move |c| {
            let (u, v) = (c[0], c[1]);
            vec![r * u.cos() * v.sin(), r * u.sin() * v.sin(), r * v.cos()]
        })
    }

    /// Ellipsoid with semi-axes a, b, c:
    /// f(u, v) = (a·cos u · sin v,  b·sin u · sin v,  c·cos v)
    pub fn ellipsoid(a: f64, b: f64, c: f64) -> Self {
        Surface::new(2, 3, move |coord| {
            let (u, v) = (coord[0], coord[1]);
            vec![a * u.cos() * v.sin(), b * u.sin() * v.sin(), c * v.cos()]
        })
    }

    #[inline]
    pub fn eval(&self, coords: &[f64]) -> Vec<f64> {
        (self.f)(coords)
    }
}

// ── Internal math helpers ─────────────────────────────────────────────────────

fn surface_diff(s: &Surface, coords: &[f64], idx: usize) -> Vec<f64> {
    let mut cp = coords.to_vec();
    let mut cm = coords.to_vec();
    cp[idx] += H_SURF;
    cm[idx] -= H_SURF;
    let fp = s.eval(&cp);
    let fm = s.eval(&cm);
    fp.iter().zip(&fm).map(|(a, b)| (a - b) / (2.0 * H_SURF)).collect()
}

/// g_{ij} = Σₖ (∂fₖ/∂xⁱ)(∂fₖ/∂xʲ)
fn metric_ij(s: &Surface, coords: &[f64], i: usize, j: usize) -> f64 {
    let di = surface_diff(s, coords, i);
    let dj = surface_diff(s, coords, j);
    di.iter().zip(&dj).map(|(a, b)| a * b).sum()
}

fn metric_tensor(s: &Surface, coords: &[f64]) -> Vec<f64> {
    let n = s.n_params;
    let mut g = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in i..n {
            let v = metric_ij(s, coords, i, j);
            g[i * n + j] = v;
            g[j * n + i] = v;
        }
    }
    g
}

/// Gauss-Jordan inversion; panics on singular metric tensor.
fn mat_inv(a: &[f64], n: usize) -> Vec<f64> {
    let w = 2 * n;
    let mut aug = vec![0.0_f64; n * w];
    for i in 0..n {
        for j in 0..n { aug[i * w + j] = a[i * n + j]; }
        aug[i * w + n + i] = 1.0;
    }
    for col in 0..n {
        let pr = (col..n)
            .max_by(|&r1, &r2| {
                aug[r1 * w + col].abs().partial_cmp(&aug[r2 * w + col].abs()).unwrap()
            })
            .unwrap();
        for j in 0..w { aug.swap(col * w + j, pr * w + j); }
        let pv = aug[col * w + col];
        assert!(pv.abs() > 1e-14, "singular metric tensor at this point");
        let ip = 1.0 / pv;
        for j in 0..w { aug[col * w + j] *= ip; }
        for row in 0..n {
            if row != col {
                let f = aug[row * w + col];
                for j in 0..w {
                    let delta = f * aug[col * w + j];
                    aug[row * w + j] -= delta;
                }
            }
        }
    }
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n { inv[i * n + j] = aug[i * w + n + j]; }
    }
    inv
}

fn metric_deriv(s: &Surface, coords: &[f64], a: usize, b: usize, c: usize) -> f64 {
    let n = s.n_params;
    let mut cp = coords.to_vec();
    let mut cm = coords.to_vec();
    cp[c] += H_METR;
    cm[c] -= H_METR;
    let gp = metric_tensor(s, &cp);
    let gm = metric_tensor(s, &cm);
    (gp[a * n + b] - gm[a * n + b]) / (2.0 * H_METR)
}

/// Γⁱ_{jk} = ½ Σₗ gⁱˡ (∂ₖg_{lj} + ∂ⱼg_{lk} − ∂ₗg_{jk}), stored as [i·n² + j·n + k].
fn christoffel_at(s: &Surface, coords: &[f64]) -> Vec<f64> {
    let n = s.n_params;
    let g = metric_tensor(s, coords);
    let gi = mat_inv(&g, n);

    // Compute all unique ∂g_{ab}/∂xᶜ, exploiting g_{ab} = g_{ba}.
    let dg_index: Vec<(usize, usize, usize)> = (0..n)
        .flat_map(|a| (a..n).flat_map(move |b| (0..n).map(move |c| (a, b, c))))
        .collect();
    #[cfg(feature = "parallel")]
    let dg_entries: Vec<(usize, usize, usize, f64)> = dg_index
        .into_par_iter()
        .map(|(a, b, c)| (a, b, c, metric_deriv(s, coords, a, b, c)))
        .collect();
    #[cfg(not(feature = "parallel"))]
    let dg_entries: Vec<(usize, usize, usize, f64)> = dg_index
        .into_iter()
        .map(|(a, b, c)| (a, b, c, metric_deriv(s, coords, a, b, c)))
        .collect();
    let mut dg = vec![0.0_f64; n * n * n];
    for (a, b, c, v) in dg_entries {
        dg[a * n * n + b * n + c] = v;
        dg[b * n * n + a * n + c] = v;
    }

    // Compute all Γⁱ_{jk} once dg is ready.
    #[cfg(feature = "parallel")]
    let gamma: Vec<f64> = (0..n * n * n)
        .into_par_iter()
        .map(|idx| {
            let i = idx / (n * n);
            let j = (idx / n) % n;
            let k = idx % n;
            0.5 * (0..n)
                .map(|l| {
                    gi[i * n + l]
                        * (dg[l * n * n + j * n + k]
                            + dg[l * n * n + k * n + j]
                            - dg[j * n * n + k * n + l])
                })
                .sum::<f64>()
        })
        .collect();
    #[cfg(not(feature = "parallel"))]
    let gamma: Vec<f64> = (0..n * n * n)
        .map(|idx| {
            let i = idx / (n * n);
            let j = (idx / n) % n;
            let k = idx % n;
            0.5 * (0..n)
                .map(|l| {
                    gi[i * n + l]
                        * (dg[l * n * n + j * n + k]
                            + dg[l * n * n + k * n + j]
                            - dg[j * n * n + k * n + l])
                })
                .sum::<f64>()
        })
        .collect();
    gamma
}

/// 4th-order Runge-Kutta step.
fn rk4<F: Fn(f64, &[f64]) -> Vec<f64>>(f: F, t: f64, y: &[f64], dt: f64) -> Vec<f64> {
    let k1 = f(t, y);
    let y2: Vec<f64> = y.iter().zip(&k1).map(|(yi, ki)| yi + 0.5 * dt * ki).collect();
    let k2 = f(t + 0.5 * dt, &y2);
    let y3: Vec<f64> = y.iter().zip(&k2).map(|(yi, ki)| yi + 0.5 * dt * ki).collect();
    let k3 = f(t + 0.5 * dt, &y3);
    let y4: Vec<f64> = y.iter().zip(&k3).map(|(yi, ki)| yi + dt * ki).collect();
    let k4 = f(t + dt, &y4);
    y.iter()
        .enumerate()
        .map(|(i, yi)| yi + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

/// Composite Simpson's 1/3 rule; trapezoidal fallback for the last interval when n is even.
fn simpsons(t: &[f64], f: &[f64]) -> f64 {
    let n = t.len();
    if n < 2 { return 0.0; }
    let mut acc = 0.0;
    let mut i = 0;
    while i + 2 < n {
        let h = t[i + 1] - t[i];
        acc += h / 3.0 * (f[i] + 4.0 * f[i + 1] + f[i + 2]);
        i += 2;
    }
    // i == n-2  → one interval [n-2, n-1] was not covered; use trapezoidal.
    // i == n-1  → all intervals covered by Simpson's pairs; nothing to add.
    if i + 2 == n {
        acc += (t[n - 1] - t[n - 2]) * (f[n - 2] + f[n - 1]) / 2.0;
    }
    acc
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Solution of a geodesic initial-value problem.
pub struct Solution {
    /// Time evaluation grid.
    pub t: Vec<f64>,
    /// `y[var][time_idx]`; indices `0..n` are coordinates, `n..2n` are velocities.
    pub y: Vec<Vec<f64>>,
}

pub struct GeodesicCurve {
    pub surface: Surface,
}

impl GeodesicCurve {
    pub fn new(surface: Surface) -> Self {
        GeodesicCurve { surface }
    }

    /// Metric tensor g at `coords` (row-major, n×n).
    pub fn metric_tensor_at(&self, coords: &[f64]) -> Vec<f64> {
        metric_tensor(&self.surface, coords)
    }

    /// Christoffel symbols Γⁱ_{jk} at `coords`, stored as `[i·n² + j·n + k]`.
    pub fn christoffel_symbols_at(&self, coords: &[f64]) -> Vec<f64> {
        christoffel_at(&self.surface, coords)
    }

    /// Riemannian inner product ⟨v, v⟩_g = Σ_{ij} g_{ij} vⁱ vʲ at `coords`.
    pub fn inner_product(&self, coords: &[f64], v: &[f64]) -> f64 {
        let n = self.surface.n_params;
        let g = metric_tensor(&self.surface, coords);
        (0..n)
            .flat_map(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| g[i * n + j] * v[i] * v[j])
            .sum()
    }

    /// d/dt [q, q̇] = [q̇, −Γⁱ_{jk} q̇ʲ q̇ᵏ]
    fn geodesic_rhs(&self, _t: f64, y: &[f64]) -> Vec<f64> {
        let n = self.surface.n_params;
        let (coords, vel) = (&y[..n], &y[n..]);
        let gamma = christoffel_at(&self.surface, coords);
        #[cfg(feature = "parallel")]
        let accel: Vec<f64> = (0..n)
            .into_par_iter()
            .map(|i| {
                -(0..n)
                    .flat_map(|j| (0..n).map(move |k| (j, k)))
                    .map(|(j, k)| gamma[i * n * n + j * n + k] * vel[j] * vel[k])
                    .sum::<f64>()
            })
            .collect();
        #[cfg(not(feature = "parallel"))]
        let accel: Vec<f64> = (0..n)
            .map(|i| {
                -(0..n)
                    .flat_map(|j| (0..n).map(move |k| (j, k)))
                    .map(|(j, k)| gamma[i * n * n + j * n + k] * vel[j] * vel[k])
                    .sum::<f64>()
            })
            .collect();
        vel.iter().chain(&accel).copied().collect()
    }

    /// Solve the geodesic IVP with RK4; state is recorded at every point in `t_eval`.
    ///
    /// Internally sub-steps each interval to at most `DT_MAX` to match the
    /// accuracy of scipy's adaptive `solve_ivp`.
    pub fn solve_ivp(&self, t_eval: &[f64], y0: &[f64]) -> Solution {
        let n_y = y0.len();
        let n_t = t_eval.len();
        let mut y = vec![vec![0.0; n_t]; n_y];
        let mut yc = y0.to_vec();
        let mut t_curr = t_eval[0];
        for i in 0..n_y { y[i][0] = yc[i]; }
        for ti in 1..n_t {
            let t_target = t_eval[ti];
            let span = t_target - t_curr;
            let n_sub = (span / DT_MAX).ceil() as usize;
            let dt_sub = span / n_sub as f64;
            for s in 0..n_sub {
                let t_step = t_curr + s as f64 * dt_sub;
                yc = rk4(|t, yy| self.geodesic_rhs(t, yy), t_step, &yc, dt_sub);
            }
            t_curr = t_target;
            for i in 0..n_y { y[i][ti] = yc[i]; }
        }
        Solution { t: t_eval.to_vec(), y }
    }

    /// Arc length of the curve via composite Simpson's rule on the Riemannian speed.
    pub fn curve_length(&self, sol: &Solution) -> f64 {
        let n = self.surface.n_params;
        #[cfg(feature = "parallel")]
        let speeds: Vec<f64> = (0..sol.t.len())
            .into_par_iter()
            .map(|ti| {
                let coords: Vec<f64> = (0..n).map(|i| sol.y[i][ti]).collect();
                let vel: Vec<f64> = (0..n).map(|i| sol.y[n + i][ti]).collect();
                self.inner_product(&coords, &vel).sqrt()
            })
            .collect();
        #[cfg(not(feature = "parallel"))]
        let speeds: Vec<f64> = (0..sol.t.len())
            .map(|ti| {
                let coords: Vec<f64> = (0..n).map(|i| sol.y[i][ti]).collect();
                let vel: Vec<f64> = (0..n).map(|i| sol.y[n + i][ti]).collect();
                self.inner_product(&coords, &vel).sqrt()
            })
            .collect();
        simpsons(&sol.t, &speeds)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() <= eps, "left={a}, right={b}, eps={eps}");
    }

    #[test]
    fn sphere_metric_and_christoffel_match_known_values() {
        let gc = GeodesicCurve::new(Surface::sphere(1.0));
        let pt = [std::f64::consts::PI / 4.0, std::f64::consts::PI / 3.0];
        let g = gc.metric_tensor_at(&pt);
        approx(g[0], 0.75, 1e-6);
        approx(g[1], 0.0, 1e-6);
        approx(g[2], 0.0, 1e-6);
        approx(g[3], 1.0, 1e-6);

        let gamma = gc.christoffel_symbols_at(&pt);
        approx(gamma[1], 1.0 / 3.0_f64.sqrt(), 1e-4); // Γ^u_uv
        approx(gamma[2], 1.0 / 3.0_f64.sqrt(), 1e-4); // Γ^u_vu
        approx(gamma[4], -0.4330127018922193, 1e-4); // Γ^v_uu
    }

    #[test]
    fn solve_ivp_and_curve_length_on_plane() {
        let plane = Surface::new(2, 2, |c| vec![c[0], c[1]]);
        let gc = GeodesicCurve::new(plane);
        let t_eval: Vec<f64> = (0..11).map(|i| i as f64 / 10.0).collect();
        let y0 = [0.0, 0.0, 1.0, 2.0];

        let sol = gc.solve_ivp(&t_eval, &y0);
        approx(sol.y[0][10], 1.0, 1e-8);
        approx(sol.y[1][10], 2.0, 1e-8);
        approx(sol.y[2][10], 1.0, 1e-8);
        approx(sol.y[3][10], 2.0, 1e-8);

        let len = gc.curve_length(&sol);
        approx(len, 5.0_f64.sqrt(), 1e-7);
    }

    #[test]
    fn simpsons_odd_and_even_samples() {
        let t_odd = [0.0, 0.5, 1.0];
        let f_odd = [0.0, 0.25, 1.0]; // x^2
        approx(simpsons(&t_odd, &f_odd), 1.0 / 3.0, 1e-12);

        let t_even = [0.0, 0.5, 1.0, 1.5];
        let f_even = [1.0, 1.0, 1.0, 1.0];
        approx(simpsons(&t_even, &f_even), 1.5, 1e-12);
    }

    #[test]
    #[should_panic(expected = "singular metric tensor")]
    fn mat_inv_panics_on_singular_matrix() {
        let singular = [1.0, 2.0, 2.0, 4.0];
        let _ = mat_inv(&singular, 2);
    }

    #[test]
    fn inner_product_is_positive_for_nonzero_velocity() {
        let gc = GeodesicCurve::new(Surface::paraboloid());
        let val = gc.inner_product(&[0.2, -0.3], &[1.0, -0.5]);
        assert!(val > 0.0);
    }

    #[test]
    fn parallel_flag_matches_feature() {
        #[cfg(feature = "parallel")]
        assert!(PARALLEL_ENABLED);
        #[cfg(not(feature = "parallel"))]
        assert!(!PARALLEL_ENABLED);
    }
}

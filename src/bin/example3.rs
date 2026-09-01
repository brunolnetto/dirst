//! Mirrors lab/main.py: metric tensors, Christoffel symbols, and geodesic IVP.
//!
//! Adaptation notes vs Python:
//!   - SymPy symbolic diff   → numerical central differences (H_SURF, H_METR)
//!   - scipy solve_ivp       → RK4
//!   - scipy simps           → composite Simpson's rule
use std::f64::consts::PI;
use dirst::geodesic::{GeodesicCurve, Surface};

fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
    let d = (end - start) / (n - 1) as f64;
    (0..n).map(|i| start + i as f64 * d).collect()
}

fn print_christoffels(gc: &GeodesicCurve, pt: &[f64], labels: &[&str]) {
    let n = gc.surface.n_params;
    let gamma = gc.christoffel_symbols_at(pt);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                println!(
                    "  Γ^{}_{{{}{}}}: {:>12.6}",
                    labels[i], labels[j], labels[k],
                    gamma[i * n * n + j * n + k]
                );
            }
        }
    }
}

fn main() {
    // ── Sphere ────────────────────────────────────────────────────────────────
    {
        let gc = GeodesicCurve::new(Surface::sphere(1.0));
        let pt = [PI / 4.0, PI / 3.0];

        let g = gc.metric_tensor_at(&pt);
        println!("Sphere metric tensor at (u=π/4, v=π/3):");
        println!("  [[{:.6}, {:.6}],", g[0], g[1]);
        println!("   [{:.6}, {:.6}]]", g[2], g[3]);
        println!("  (analytical: g_uu = sin²(π/3) = 3/4,  g_vv = 1)");
        println!();

        println!("Sphere Christoffel symbols at (u=π/4, v=π/3):");
        println!("  (analytical: Γ^u_uv = Γ^u_vu = cot(π/3) ≈  0.577350)");
        println!("  (analytical: Γ^v_uu = -sin(π/3)cos(π/3) ≈ -0.433013)");
        print_christoffels(&gc, &pt, &["u", "v"]);
        println!();
    }

    // ── Paraboloid ────────────────────────────────────────────────────────────
    {
        let gc = GeodesicCurve::new(Surface::paraboloid());
        let pt = [0.5_f64, 0.5];

        let g = gc.metric_tensor_at(&pt);
        println!("Paraboloid metric tensor at (u=0.5, v=0.5):");
        println!("  [[{:.6}, {:.6}],", g[0], g[1]);
        println!("   [{:.6}, {:.6}]]", g[2], g[3]);
        println!("  (analytical: g_uu = g_vv = 2,  g_uv = 1)");
        println!();

        println!("Paraboloid Christoffel symbols at (u=0.5, v=0.5):");
        println!("  (analytical: Γ^u_uu = Γ^u_vv = Γ^v_uu = Γ^v_vv = 4u/D ≈ 0.666667,  rest 0)");
        print_christoffels(&gc, &pt, &["u", "v"]);
        println!();
    }

    // ── Ellipsoid Christoffel symbols ─────────────────────────────────────────
    {
        let gc = GeodesicCurve::new(Surface::ellipsoid(3.0, 2.0, 1.0));
        let pt = [PI / 4.0, PI / 4.0];
        println!("Ellipsoid (a=3,b=2,c=1) Christoffel symbols at (u=π/4, v=π/4):");
        print_christoffels(&gc, &pt, &["u", "v"]);
        println!();
    }

    // ── Paraboloid geodesic IVP ───────────────────────────────────────────────
    {
        let gc = GeodesicCurve::new(Surface::paraboloid());

        // Exact arc length of the path (t/√2, t/√2) on the paraboloid, t ∈ [0,1]:
        //   ∫₀¹ √(1 + 4t²) dt  =  (2√5 + arcsinh(2)) / 4
        let vel = (2.0 * 5.0_f64.sqrt() + 2.0_f64.asinh()) / 4.0;
        let s2 = 2.0_f64.sqrt();
        let y0 = [0.0, 0.0, vel / s2, vel / s2];

        let t_eval = linspace(0.0, 1.0, 100);
        let sol = gc.solve_ivp(&t_eval, &y0);
        let length = gc.curve_length(&sol);

        println!("Paraboloid geodesic IVP – p₀=(0,0), direction [1,1]/√2:");
        println!("  Initial speed (exact) : {:.10}", vel);
        println!("  Curve length  (RK4)   : {:.10}", length);
        println!("  Error                 : {:.2e}", (length - vel).abs());
    }
}

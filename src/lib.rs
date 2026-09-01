pub mod geodesic;
pub mod utils;

pub use utils::geo_to_spher;

use utils::{hav, is_geographical, is_spherical, spher_to_cart};

/// Distance method, carrying its configuration inline.
#[derive(Debug, Clone, PartialEq)]
pub enum Method {
    /// p-norm with given exponent (use `f64::INFINITY` for Chebyshev)
    PNorm(f64),
    Manhattan,
    CityBlock,
    Cosine,
    Canberra,
    BrayCurtis,
    Euclidean,
    SqEuclidean,
    Max,
    Chebyshev,
    /// Great-circle distance on a sphere of given radius
    Sphere(f64),
    /// Great-circle distance from geographical (lat/lng°) coordinates
    Geographical(f64),
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Method::PNorm(_) => write!(f, "pnorm"),
            Method::Manhattan => write!(f, "manhattan"),
            Method::CityBlock => write!(f, "cityblock"),
            Method::Cosine => write!(f, "cosine"),
            Method::Canberra => write!(f, "canberra"),
            Method::BrayCurtis => write!(f, "braycurtis"),
            Method::Euclidean => write!(f, "euclidean"),
            Method::SqEuclidean => write!(f, "sqeuclidean"),
            Method::Max => write!(f, "max"),
            Method::Chebyshev => write!(f, "chebyshev"),
            Method::Sphere(_) => write!(f, "sphere"),
            Method::Geographical(_) => write!(f, "geographical"),
        }
    }
}

// ─── Internal helpers ────────────────────────────────────────────────────────

fn p_norm(arr: &[f64], p: f64) -> f64 {
    arr.iter().map(|&x| x.abs().powf(p)).sum::<f64>().powf(1.0 / p)
}

fn p_norm_distance(c1: &[f64], c2: &[f64], p: f64) -> Result<f64, String> {
    if p < 1.0 {
        return Err("The exponent n must be a number greater or equal to 1!".to_string());
    }
    let diff: Vec<f64> = c1.iter().zip(c2).map(|(a, b)| (a - b).abs()).collect();
    if p == f64::INFINITY {
        Ok(diff.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
    } else {
        Ok(p_norm(&diff, p))
    }
}

fn dot(u: &[f64], v: &[f64]) -> f64 {
    u.iter().zip(v).map(|(a, b)| a * b).sum()
}

fn cos_nuv(u: &[f64], v: &[f64], n: f64) -> Result<f64, String> {
    let nu = p_norm(u, n);
    let nv = p_norm(v, n);
    if nu == 0.0 || nv == 0.0 {
        Err("Method 'cosine' does not support a null vector.".to_string())
    } else {
        Ok(dot(u, v) / (nu * nv))
    }
}

fn arg_uv(u: &[f64], v: &[f64], n: f64) -> Result<f64, String> {
    Ok(cos_nuv(u, v, n)?.acos())
}

fn sphere_central_angle(c1: &[f64], c2: &[f64]) -> f64 {
    let (lon1, lon2) = (c1[0], c2[0]);
    let (lat1, lat2) = (c1[1], c2[1]);
    let hav_theta =
        hav(lat2 - lat1) + hav(lon2 - lon1) * (1.0 - hav(lat1 - lat2) - hav(lat1 + lat2));
    2.0 * hav_theta.sqrt().asin()
}

fn central_angle(v1: &[f64], v2: &[f64], r: f64) -> Result<f64, String> {
    let cart1 = spher_to_cart(v1, r)?;
    let cart2 = spher_to_cart(v2, r)?;
    arg_uv(&cart1, &cart2, 2.0)
}

fn n_sphere_distance(c1: &[f64], c2: &[f64], r: f64) -> Result<f64, String> {
    Ok(r * central_angle(c1, c2, r)?)
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Haversine great-circle distance between two spherical points on a sphere of radius `r`.
pub fn great_circle_distance(c1: &[f64], c2: &[f64], r: f64) -> f64 {
    r * sphere_central_angle(c1, c2)
}

pub fn distance(c1: &[f64], c2: &[f64], method: &Method) -> Result<f64, String> {
    match method {
        Method::PNorm(p) => p_norm_distance(c1, c2, *p),
        Method::Manhattan | Method::CityBlock => p_norm_distance(c1, c2, 1.0),
        Method::Cosine => Ok(1.0 - cos_nuv(c1, c2, 2.0)?),
        Method::Canberra => Ok(c1
            .iter()
            .zip(c2)
            .map(|(a, b)| (a - b).abs() / (a.abs() + b.abs()))
            .sum()),
        Method::BrayCurtis => {
            let num: f64 = c1.iter().zip(c2).map(|(a, b)| (a - b).abs()).sum();
            let den: f64 = c1.iter().zip(c2).map(|(a, b)| (a + b).abs()).sum();
            Ok(num / den)
        }
        Method::Euclidean => p_norm_distance(c1, c2, 2.0),
        Method::SqEuclidean => Ok(p_norm_distance(c1, c2, 2.0)?.powi(2)),
        Method::Max | Method::Chebyshev => p_norm_distance(c1, c2, f64::INFINITY),
        Method::Sphere(r) => {
            if !is_spherical(c1) || !is_spherical(c2) {
                Err("Provided coordinates are not spherical!".to_string())
            } else {
                n_sphere_distance(c1, c2, *r)
            }
        }
        Method::Geographical(r) => {
            let g1 = is_geographical(c1);
            let g2 = is_geographical(c2);
            if !g1 || !g2 {
                let subject = match (g1, g2) {
                    (false, false) => "Both provided coordinates",
                    (false, true) => "Provided coordinate 1",
                    _ => "Provided coordinate 2",
                };
                let verb = if !g1 && !g2 { "are" } else { "is" };
                Err(format!("{} {} not geographical!", subject, verb))
            } else {
                let s1 = geo_to_spher(c1[0], c1[1]);
                let s2 = geo_to_spher(c2[0], c2[1]);
                n_sphere_distance(&s1, &s2, *r)
            }
        }
    }
}

pub fn travel_time(
    average_speed: f64,
    c1: &[f64],
    c2: &[f64],
    method: &Method,
) -> Result<f64, String> {
    Ok(distance(c1, c2, method)? / average_speed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() <= eps, "left={a}, right={b}, eps={eps}");
    }

    #[test]
    fn method_display_covers_all_variants() {
        let labels = [
            Method::PNorm(2.0).to_string(),
            Method::Manhattan.to_string(),
            Method::CityBlock.to_string(),
            Method::Cosine.to_string(),
            Method::Canberra.to_string(),
            Method::BrayCurtis.to_string(),
            Method::Euclidean.to_string(),
            Method::SqEuclidean.to_string(),
            Method::Max.to_string(),
            Method::Chebyshev.to_string(),
            Method::Sphere(1.0).to_string(),
            Method::Geographical(1.0).to_string(),
        ];
        assert_eq!(
            labels,
            [
                "pnorm",
                "manhattan",
                "cityblock",
                "cosine",
                "canberra",
                "braycurtis",
                "euclidean",
                "sqeuclidean",
                "max",
                "chebyshev",
                "sphere",
                "geographical",
            ]
        );
    }

    #[test]
    fn pnorm_family_and_basic_metrics() {
        let a = [1.0, 1.0];
        let b = [2.0, 2.0];
        approx(distance(&a, &b, &Method::PNorm(2.0)).unwrap(), 2.0_f64.sqrt(), 1e-12);
        approx(distance(&a, &b, &Method::PNorm(3.0)).unwrap(), 2.0_f64.powf(1.0 / 3.0), 1e-12);
        approx(distance(&a, &b, &Method::PNorm(f64::INFINITY)).unwrap(), 1.0, 1e-12);
        assert!(distance(&a, &b, &Method::PNorm(0.5)).is_err());

        approx(distance(&a, &b, &Method::Manhattan).unwrap(), 2.0, 1e-12);
        approx(distance(&a, &b, &Method::CityBlock).unwrap(), 2.0, 1e-12);
        approx(distance(&a, &b, &Method::Euclidean).unwrap(), 2.0_f64.sqrt(), 1e-12);
        approx(distance(&a, &b, &Method::SqEuclidean).unwrap(), 2.0, 1e-12);
        approx(distance(&a, &b, &Method::Max).unwrap(), 1.0, 1e-12);
        approx(distance(&a, &b, &Method::Chebyshev).unwrap(), 1.0, 1e-12);
    }

    #[test]
    fn cosine_canberra_braycurtis_and_errors() {
        let u = [1.0, 0.0];
        let v = [0.0, 1.0];
        approx(distance(&u, &v, &Method::Cosine).unwrap(), 1.0, 1e-12);
        assert!(distance(&[0.0, 0.0], &v, &Method::Cosine).is_err());

        let c1 = [1.0, 2.0];
        let c2 = [3.0, 4.0];
        approx(distance(&c1, &c2, &Method::Canberra).unwrap(), 2.0 / 4.0 + 2.0 / 6.0, 1e-12);
        approx(distance(&c1, &c2, &Method::BrayCurtis).unwrap(), 0.4, 1e-12);
    }

    #[test]
    fn sphere_and_geographical_methods() {
        let s1 = [0.0, 0.0];
        let s2 = [PI / 2.0, 0.0];
        approx(distance(&s1, &s2, &Method::Sphere(1.0)).unwrap(), PI / 2.0, 1e-12);
        approx(great_circle_distance(&s1, &s2, 1.0), PI / 2.0, 1e-12);
        assert!(distance(&[-1.0, 0.0], &s2, &Method::Sphere(1.0)).is_err());

        let g1 = [0.0, 0.0];
        let g2 = [0.0, 90.0];
        approx(distance(&g1, &g2, &Method::Geographical(1.0)).unwrap(), PI / 2.0, 1e-12);

        let both_bad = distance(&[200.0, 300.0], &[200.0, 300.0], &Method::Geographical(1.0))
            .unwrap_err();
        assert_eq!(both_bad, "Both provided coordinates are not geographical!");

        let bad_1 = distance(&[200.0, 0.0], &[0.0, 0.0], &Method::Geographical(1.0)).unwrap_err();
        assert_eq!(bad_1, "Provided coordinate 1 is not geographical!");

        let bad_2 = distance(&[0.0, 0.0], &[0.0, 200.0], &Method::Geographical(1.0)).unwrap_err();
        assert_eq!(bad_2, "Provided coordinate 2 is not geographical!");
    }

    #[test]
    fn travel_time_works() {
        let t = travel_time(2.0, &[0.0, 0.0], &[2.0, 0.0], &Method::Euclidean).unwrap();
        approx(t, 1.0, 1e-12);
    }
}

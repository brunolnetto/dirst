use std::f64::consts::PI;

/// Product of sines of all angles; empty slice returns 1.
fn prod_sin(angles: &[f64]) -> f64 {
    angles.iter().map(|&a| a.sin()).product()
}

pub fn radian_to_degree(angle_radian: f64) -> f64 {
    (180.0 * angle_radian) / PI
}

pub fn degree_to_radian(angle_degree: f64) -> f64 {
    (PI * angle_degree) / 180.0
}

/// Haversine: sin²(θ/2)
pub fn hav(theta_radian: f64) -> f64 {
    (theta_radian / 2.0).sin().powi(2)
}

/// Geographical (lat, lng in degrees) → spherical coordinates
pub fn geo_to_spher(lat_degree: f64, lng_degree: f64) -> [f64; 2] {
    [
        PI / 2.0 + degree_to_radian(lat_degree),
        PI + degree_to_radian(lng_degree),
    ]
}

/// Spherical coordinates → geographical (lat, lng in degrees)
pub fn spher_to_geo(coords: &[f64]) -> (f64, f64) {
    (
        radian_to_degree(coords[0] - PI / 2.0),
        radian_to_degree(coords[1] - PI),
    )
}

/// Geographical: len == 2, lat ∈ [-90, 90], lng ∈ [-180, 180]
pub fn is_geographical(u: &[f64]) -> bool {
    u.len() == 2 && u[0] >= -90.0 && u[0] <= 90.0 && u[1] >= -180.0 && u[1] <= 180.0
}

/// Spherical: len >= 2, u[0..n-2] ∈ [0, π], u[n-1] ∈ [0, 2π]
pub fn is_spherical(u: &[f64]) -> bool {
    let n = u.len();
    if n < 2 {
        return false;
    }
    u[..n - 1].iter().all(|&x| x >= 0.0 && x <= PI)
        && u[n - 1] >= 0.0
        && u[n - 1] <= 2.0 * PI
}

/// Convert n-dimensional spherical coordinates to (n+1)-dimensional Cartesian.
/// Output order mirrors the Python implementation: (z, x, y) for 3-D.
pub fn spher_to_cart(coords: &[f64], r: f64) -> Result<Vec<f64>, String> {
    if !is_spherical(coords) {
        return Err(format!(
            "These are criteria for input to be spherical: \n\
             1. It is an array with more than two elements;\n \
             2. Elements between indexes 0 to {} must be between -pi and pi;\n \
             3. Last element must be between 0 and 2*pi.",
            coords.len() - 1
        ));
    }
    let n = coords.len();
    // x_i = R · ∏sin(coords[0..i]) · cos(coords[i])
    let mut result: Vec<f64> = (0..n)
        .map(|i| r * prod_sin(&coords[..i]) * coords[i].cos())
        .collect();
    // last component: R · ∏sin(coords[0..n-1]) · sin(coords[n-1])
    result.push(r * prod_sin(&coords[..n - 1]) * coords[n - 1].sin());
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, eps: f64) {
        assert!((a - b).abs() <= eps, "left={a}, right={b}, eps={eps}");
    }

    #[test]
    fn angle_conversion_roundtrip() {
        approx(radian_to_degree(PI), 180.0, 1e-12);
        approx(degree_to_radian(180.0), PI, 1e-12);
    }

    #[test]
    fn haversine_basics() {
        approx(hav(0.0), 0.0, 1e-12);
        approx(hav(PI), 1.0, 1e-12);
    }

    #[test]
    fn geo_and_spherical_roundtrip() {
        let geo = [27.700769, 85.30014];
        let sph = geo_to_spher(geo[0], geo[1]);
        let back = spher_to_geo(&sph);
        approx(back.0, geo[0], 1e-12);
        approx(back.1, geo[1], 1e-12);
    }

    #[test]
    fn coordinate_validators() {
        assert!(is_geographical(&[0.0, 0.0]));
        assert!(!is_geographical(&[91.0, 0.0]));
        assert!(!is_geographical(&[0.0, 181.0]));
        assert!(!is_geographical(&[0.0]));

        assert!(is_spherical(&[PI / 2.0, PI]));
        assert!(!is_spherical(&[PI / 2.0]));
        assert!(!is_spherical(&[-1.0, PI]));
        assert!(!is_spherical(&[PI / 2.0, 3.0 * PI]));
    }

    #[test]
    fn spherical_to_cartesian_success_and_error() {
        let cart = spher_to_cart(&[PI / 2.0, PI], 1.0).unwrap();
        approx(cart[0], 0.0, 1e-12);
        approx(cart[1], -1.0, 1e-12);
        approx(cart[2], 0.0, 1e-12);

        let err = spher_to_cart(&[-1.0, 0.0], 1.0).unwrap_err();
        assert!(err.contains("criteria"));
    }
}

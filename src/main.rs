/// Mirrors examples/example.py from the spycio Python package.
use std::f64::consts::PI;
use dirst::{distance, travel_time, Method};

fn format_distance(a: &[f64], b: &[f64], speed: f64, method: &Method) -> String {
    let d = distance(a, b, method).unwrap();
    let eta = travel_time(speed, a, b, method).unwrap();
    format!(
        "A:{a:?}, B:{b:?}, speed:{speed}, method:{method}, distance:{d}, eta:{eta}"
    )
}

fn format_distance_with_config(a: &[f64], b: &[f64], speed: f64, method: &Method) -> String {
    let d = distance(a, b, method).unwrap();
    let eta = travel_time(speed, a, b, method).unwrap();
    let config = match method {
        Method::PNorm(exp) => format!("{{\"exponent\": {exp}}}"),
        Method::Sphere(r) | Method::Geographical(r) => format!("{{\"radius\": {r}}}"),
        _ => "{}".to_string(),
    };
    format!(
        "A:{a:?}, B:{b:?}, speed:{speed}, method:{method}, config:{config}, distance:{d}, eta:{eta}"
    )
}

fn main() {
    let a = &[0.0_f64, 0.0];
    let b = &[1.0_f64, 1.0];
    let c = &[2.0_f64, 2.0];
    let d = &[PI / 2.0, 0.0];
    let speed = 1.0_f64;

    println!("Euclidean distance: {}", distance(a, b, &Method::Euclidean).unwrap());
    println!();

    let no_config = [
        (b as &[f64], c as &[f64], speed, Method::Manhattan),
        (b, c, speed, Method::Euclidean),
        (b, c, speed, Method::Max),
    ];
    println!("Format distance without configuration:");
    for (p, q, s, m) in &no_config {
        println!("{}", format_distance(p, q, *s, m));
    }

    println!();

    let with_config = [
        (b as &[f64], c as &[f64], speed, Method::PNorm(2.0)),
        (b, c, speed, Method::PNorm(3.0)),
        (b, c, speed, Method::PNorm(4.0)),
        (b, c, speed, Method::PNorm(f64::INFINITY)),
        (a, d as &[f64], speed, Method::Sphere(1.0)),
    ];
    println!("Format distance with configuration:");
    for (p, q, s, m) in &with_config {
        println!("{}", format_distance_with_config(p, q, *s, m));
    }
}

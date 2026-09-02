/// Mirrors the second Python example: pairwise great-circle distances between world cities.
use dirst::{distance, geo_to_spher, travel_time, Method};

struct City {
    name: &'static str,
    spher: [f64; 2],
    geo: [f64; 2],
}

fn city(name: &'static str, lat: f64, lng: f64) -> City {
    City {
        name,
        spher: geo_to_spher(lat, lng),
        geo: [lat, lng],
    }
}

fn format_distance(a: &[f64], b: &[f64], speed: f64, method: &Method) -> String {
    let d = distance(a, b, method).unwrap();
    let eta = travel_time(speed, a, b, method).unwrap();
    let config = match method {
        Method::Sphere(r) | Method::Geographical(r) => format!("{{\"radius\": {r}}}"),
        Method::PNorm(e) => format!("{{\"exponent\": {e}}}"),
        _ => "{}".to_string(),
    };
    format!("A:{a:?}, B:{b:?}, speed:{speed}, method:{method}, config:{config}, distance:{d}, eta:{eta}")
}

fn main() {
    let cities = [
        city("kathmandu", 27.700769, 85.300140),
        city("brasilia", -15.793889, -47.882778),
        city("sao paulo", -23.435500, -46.473100),
        city("curitiba", -25.428400, -49.273300),
        city("goiania", -16.666667, -49.266667),
        city("buenos aires", -34.603700, -58.381600),
        city("new york", 40.730610, -73.935242),
        city("sydney", -33.865143, 151.209900),
        city("berlin", 13.381777, 52.531677),
        city("tokyo", 35.689500, 139.691700),
    ];

    let earth_radius = 6371.0_f64;
    let speed = 900.0_f64;

    for c in &cities {
        println!("{:?}", c.spher);
    }
    println!();
    for c in &cities {
        println!("{:?}", c.geo);
    }
    println!();

    let n = cities.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let (a, b) = (&cities[i], &cities[j]);
            println!("Origin: {}, Target: {}", a.name, b.name);
            println!(
                "{}",
                format_distance(&a.spher, &b.spher, speed, &Method::Sphere(earth_radius))
            );
            println!(
                "{}",
                format_distance(&a.geo, &b.geo, speed, &Method::Geographical(earth_radius))
            );
            println!();
        }
    }
}

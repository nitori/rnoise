use itertools::Itertools;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::distr::{Distribution, Uniform};
use rand::prelude::*;

// fn get_cpu_count() -> PyResult<usize> {
//     Python::with_gil(|py| -> PyResult<usize> {
//         let os = py.import("os")?;
//         let fun = os.getattr("cpu_count")?;
//         let result: usize = fun.call0()?.extract()?;
//         Ok(result)
//     })
// }

enum NoiseError {
    ValueError(String),
}

type Result<T> = std::result::Result<T, NoiseError>;

#[derive(Debug)]
struct RandVec {
    coords: Vec<f64>,
    vec: Vec<f64>,
}

impl RandVec {
    fn new(coords: Vec<f64>, seed: u64) -> Result<Self> {
        let len = coords.len();
        let vec = sample_vector(len, seed)?;
        Ok(Self { coords, vec })
    }

    fn weight_to(&self, coords: &Vec<f64>) -> Result<f64> {
        let dt = self.dists_to(coords)?;
        let weighted_dists: Result<Vec<_>> = dt.iter().map(|v| fade(1f64 - v.abs())).collect();
        let result = weighted_dists?
            .iter()
            .copied()
            .reduce(|a, b| a * b)
            .unwrap_or(0.0f64);

        Ok(result)
    }

    fn dists_to(&self, coords: &Vec<f64>) -> Result<Vec<f64>> {
        let result: Vec<_> = coords
            .iter()
            .zip(&self.coords)
            .map(|(coor1, coor2)| coor1 - coor2)
            .collect();
        Ok(result)
    }

    fn get_weighted_val(&self, coords: &Vec<f64>) -> Result<f64> {
        let dt = self.dists_to(&coords)?;
        Ok(self.weight_to(coords)? * dot(&self.vec, &dt)?)
    }
}

fn fade(given_value: f64) -> Result<f64> {
    if given_value < -0.1 || given_value > 1.1 {
        return Err(NoiseError::ValueError(
            "expected to have value in [-0.1, 1.1]".into(),
        ));
    }

    let result =
        6.0 * given_value.powf(5.0) - 15.0 * given_value.powf(4.0) + 10.0 * given_value.powf(3.0);
    Ok(result)
}

fn sample_vector(len: usize, seed: u64) -> Result<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);

    let range = match Uniform::try_from(-1f64..1f64) {
        Ok(r) => Ok(r),
        Err(e) => Err(NoiseError::ValueError(format!(
            "Failed to create uniform thingy: {:?}",
            e
        ))),
    }?;

    let v: Vec<_> = (0..len).map(|_| range.sample(&mut rng)).collect();
    Ok(v)
}

fn dot(v1: &Vec<f64>, v2: &Vec<f64>) -> Result<f64> {
    if v1.len() != v2.len() {
        return Err(NoiseError::ValueError(
            "Vectors need to be the same length for dot product.".into(),
        ));
    }
    let muls: Vec<_> = v1.iter().zip(v2).map(|(a, b)| a * b).collect();
    Ok(muls.iter().sum())
}

fn hasher(coords: &Vec<f64>) -> Result<u64> {
    let foo = coords
        .iter()
        .enumerate()
        .map(|t| 10i64.pow(t.0 as u32) as f64)
        .collect::<Vec<f64>>();

    let val = (dot(&foo, &coords)? + 1.0).abs() as u64;
    Ok(val.max(1))
}

fn product(ranges: &Vec<(f64, f64)>) -> Vec<Vec<f64>> {
    let all_vecs: Vec<Vec<f64>> = ranges
        .iter()
        .map(|(start, end)| {
            let start_int = *start as i64;
            let end_int = *end as i64;
            (start_int..=end_int).map(|v| v as f64).collect()
        })
        .collect();

    let result: Vec<_> = all_vecs.into_iter().multi_cartesian_product().collect();
    result
}

#[pyclass]
struct Perlin {
    seed: u64,
}

#[derive(Debug)]
struct NoiseSettings {
    coords: Vec<f64>,
    seed: u64,
    freq: f64,
    octaves: u32,
    persistence: f64,
    lacunarity: f64,
}

fn raw_noise(coords: Vec<f64>, seed: u64) -> Result<f64> {
    let coor_bounding_box = coords
        .iter()
        .map(|val| (val.floor(), (val + 1.0).floor()))
        .collect::<Vec<(f64, f64)>>();

    let mut total: f64 = 0.0;
    for coors in product(&coor_bounding_box) {
        let rv = RandVec::new(coors.clone(), seed * hasher(&coors)?)?;
        total += rv.get_weighted_val(&coords)?;
    }

    Ok(total)
}

fn noise(settings: NoiseSettings) -> Result<f64> {
    if settings.coords.len() == 0 {
        return Err(NoiseError::ValueError(
            "Cannot create noise with empty coordinates.".into(),
        ));
    }

    let mut total = 0f64;
    let mut amplitude = 1f64;
    let mut total_amplitude = 1f64;
    let mut _freq = settings.freq;

    for _ in 0..settings.octaves {
        total += raw_noise(
            settings.coords.iter().map(|v| v * _freq).collect(),
            settings.seed,
        )? * amplitude;
        total_amplitude += amplitude;
        amplitude *= settings.persistence;
        _freq *= settings.lacunarity;
    }

    Ok(total / total_amplitude)
}

fn noise_img(width: usize, height: usize, settings: NoiseSettings) -> Result<Vec<Vec<f64>>> {
    let mut result = vec![vec![0f64; width]; height];

    for y in 0..height {
        for x in 0..width {
            let mut c = vec![x as f64, y as f64];
            c.extend(settings.coords.iter());
            let val = noise(NoiseSettings {
                coords: c,
                ..settings
            })?;
            result[y][x] = val;
        }
    }

    Ok(result)
}

#[pymethods]
impl Perlin {
    #[new]
    #[pyo3(signature = (seed))]
    fn new(seed: u64) -> Self {
        Self { seed }
    }

    #[pyo3(signature=(coords, freq=1.0, octaves=1, persistence=1.0, lacunarity=1.0))]
    fn noise(
        &self,
        coords: Vec<f64>,
        freq: f64,
        octaves: u32,
        persistence: f64,
        lacunarity: f64,
    ) -> PyResult<f64> {
        if coords.len() == 0 {
            return Err(PyValueError::new_err("coords must not be empty."));
        }
        if octaves == 0 {
            return Err(PyValueError::new_err("octaves must not be zero."));
        }

        let settings = NoiseSettings {
            coords,
            seed: self.seed,
            freq,
            octaves,
            persistence,
            lacunarity,
        };
        match noise(settings) {
            Ok(v) => Ok(v),
            Err(e) => match e {
                NoiseError::ValueError(s) => Err(PyValueError::new_err(s)),
            },
        }
    }

    #[pyo3(signature=(width, height, coords, freq=1.0, octaves=1, persistence=1.0, lacunarity=1.0))]
    fn noise_img(
        &self,
        width: usize,
        height: usize,
        coords: Vec<f64>,
        freq: f64,
        octaves: u32,
        persistence: f64,
        lacunarity: f64,
    ) -> PyResult<Vec<Vec<f64>>> {
        if width == 0 || height == 0 {
            return Err(PyValueError::new_err("width and height must not be zero."));
        }
        if octaves == 0 {
            return Err(PyValueError::new_err("octaves must not be zero."));
        }

        let settings = NoiseSettings {
            coords,
            seed: self.seed,
            freq,
            octaves,
            persistence,
            lacunarity,
        };
        match noise_img(width, height, settings) {
            Ok(v) => Ok(v),
            Err(e) => match e {
                NoiseError::ValueError(s) => Err(PyValueError::new_err(s)),
            },
        }
    }
}

#[pymodule]
fn rnoise(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Perlin>()?;
    Ok(())
}

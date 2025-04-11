use itertools::Itertools;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
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

#[derive(Debug)]
struct RandVec {
    coords: Vec<f64>,
    vec: Vec<f64>,
}

impl RandVec {
    fn new(coords: Vec<f64>, seed: u64) -> PyResult<Self> {
        let len = coords.len();
        let vec = sample_vector(len, seed)?;
        Ok(Self { coords, vec })
    }

    fn weight_to(&self, coords: &Vec<f64>) -> PyResult<f64> {
        let dt = self.dists_to(coords)?;
        let weighted_dists: PyResult<Vec<_>> = dt.iter().map(|v| fade(1f64 - v.abs())).collect();
        let result = weighted_dists?
            .iter()
            .copied()
            .reduce(|a, b| a * b)
            .unwrap_or(0.0f64);

        Ok(result)
    }

    fn dists_to(&self, coords: &Vec<f64>) -> PyResult<Vec<f64>> {
        let result: Vec<_> = coords
            .iter()
            .zip(&self.coords)
            .map(|(coor1, coor2)| coor1 - coor2)
            .collect();
        Ok(result)
    }

    fn get_weighted_val(&self, coords: &Vec<f64>) -> PyResult<f64> {
        let dt = self.dists_to(&coords)?;
        Ok(self.weight_to(coords)? * dot(&self.vec, &dt)?)
    }
}

fn fade(given_value: f64) -> PyResult<f64> {
    if given_value < -0.1 || given_value > 1.1 {
        return Err(PyValueError::new_err(
            "expected to have value in [-0.1, 1.1]",
        ));
    }

    let result =
        6.0 * given_value.powf(5.0) - 15.0 * given_value.powf(4.0) + 10.0 * given_value.powf(3.0);
    Ok(result)
}

fn sample_vector(len: usize, seed: u64) -> PyResult<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);

    let range = match Uniform::try_from(-1f64..1f64) {
        Ok(r) => Ok(r),
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "Failed to create uniform thingy: {:?}",
            e
        ))),
    }?;

    let v: Vec<_> = (0..len).map(|_| range.sample(&mut rng)).collect();
    Ok(v)
}

fn dot(v1: &Vec<f64>, v2: &Vec<f64>) -> PyResult<f64> {
    if v1.len() != v2.len() {
        return Err(PyValueError::new_err(
            "Vectors need to be the same length for dot product.",
        ));
    }
    let muls: Vec<_> = v1.iter().zip(v2).map(|(a, b)| a * b).collect();
    Ok(muls.iter().sum())
}

fn hasher(coords: &Vec<f64>) -> PyResult<u64> {
    let foo = coords
        .iter()
        .enumerate()
        .map(|t| 10i64.pow(t.0 as u32) as f64)
        .collect::<Vec<f64>>();

    let val = (dot(&foo, &coords)? + 1.0).abs() as u64;
    Ok(val.max(1))
}

#[pyclass]
struct Perlin {
    seed: u64,
    octaves: u32,
}

#[pymethods]
impl Perlin {
    #[new]
    #[pyo3(signature = (seed, octaves=1))]
    fn new(seed: u64, octaves: u32) -> Self {
        Self { seed, octaves }
    }

    fn noise(&self, coords: Vec<f64>) -> PyResult<f64> {
        if coords.len() == 0 {
            return Err(PyTypeError::new_err(
                "Cannot create noise with empty coordinates.",
            ));
        }

        let coords = coords
            .iter()
            .map(|val| val * self.octaves as f64)
            .collect::<Vec<f64>>();

        let coor_bounding_box = coords
            .iter()
            .map(|val| (val.floor(), (val + 1.0).floor()))
            .collect::<Vec<(f64, f64)>>();

        let mut total: f64 = 0.0;
        for coors in product(&coor_bounding_box) {
            let rv = RandVec::new(coors.clone(), self.seed * hasher(&coors)?)?;
            total += rv.get_weighted_val(&coords)?;
        }

        Ok(total)
    }

    fn __repr__(&self) -> String {
        format!("<Perlin seed={}>", self.seed).into()
    }
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

#[pymodule]
fn rnoise(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Perlin>()?;
    Ok(())
}

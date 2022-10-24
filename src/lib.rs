use numpy::PyArray4;
use pyo3::prelude::*;
use pyo3::{pymodule, types::PyModule, PyResult, Python};

#[pyclass]
struct Sim {
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    m: Py<PyArray4<f64>>,
    #[pyo3(get)]
    t: f64,
}

#[pymethods]
impl Sim {
    #[new]
    fn new(py: Python<'_>, name: String) -> Self {
        let m = PyArray4::zeros(py, [1, 1, 1, 3], true).into_py(py);

        let t = 0.;
        Sim { name, m, t }
    }

    fn run(&self, py: Python<'_>, t: f64) {
        let mut m = self.m.as_ref(py).readwrite();
        let mut m = m.as_array_mut();
        m *= t
    }

    fn double_array(&self, py: Python<'_>) {
        let mut m = self.m.as_ref(py).readwrite();
        let mut m = m.as_array_mut();
        m *= 2.0;
    }
}

#[pymodule]
fn gwarell(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Sim>()?;
    Ok(())
}

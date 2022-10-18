use std::ops::Add;
use std::sync::Arc;

use numpy::ndarray::{Array1, ArrayD, ArrayView1, ArrayViewD, ArrayViewMutD, Zip};
use numpy::{
    datetime::{units, Timedelta},
    Complex64, IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn,
    PyReadwriteArray1, PyReadwriteArrayDyn,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{
    exceptions::PyIndexError, pymodule, types::PyModule, FromPyObject, PyAny, PyObject, PyResult,
    Python,
};

#[pyclass]
struct Sim {
    #[pyo3(get, set)]
    num: i32,
    #[pyo3(get, set)]
    name: String,
    #[pyo3(get, set)]
    m: Arc<Option<PyArray1<f64>>>,
}

#[pymethods]
impl Sim {
    #[new]
    #[args(num = "-1")]
    fn new(num: i32, name: String) -> Self {
        Sim {
            num,
            name,
            m: todo!(),
        }
    }

    fn __call__(&self) -> String {
        format!("{:?}", self.num)
    }

    #[args(num = "10", py_args = "*", name = "\"Hello\"", py_kwargs = "**")]
    fn method(
        &mut self,
        num: i32,
        name: &str,
        py_args: &PyTuple,
        py_kwargs: Option<&PyDict>,
    ) -> String {
        let num_before = self.num;
        self.num = num;
        format!(
            "py_args={:?}, py_kwargs={:?}, name={}, num={} num_before={}",
            py_args, py_kwargs, name, self.num, num_before,
        )
    }
    // #[args(m)]
}

#[pymodule]
fn gwarell(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Sim>()?;
    // example using generic PyObject
    fn head(x: ArrayViewD<'_, PyObject>) -> PyResult<PyObject> {
        x.get(0)
            .cloned()
            .ok_or_else(|| PyIndexError::new_err("array index out of range"))
    }

    // example using immutable borrows producing a new array
    fn axpy(a: f64, x: ArrayViewD<'_, f64>, y: ArrayViewD<'_, f64>) -> ArrayD<f64> {
        a * &x + &y
    }

    // example using a mutable borrow to modify an array in-place
    fn mult(a: f64, mut x: ArrayViewMutD<'_, f64>) {
        x *= a;
    }

    // example using complex numbers
    fn conj(x: ArrayViewD<'_, Complex64>) -> ArrayD<Complex64> {
        x.map(|c| c.conj())
    }

    // example using generics
    fn generic_add<T: Copy + Add<Output = T>>(x: ArrayView1<T>, y: ArrayView1<T>) -> Array1<T> {
        &x + &y
    }

    // wrapper of `head`
    #[pyfn(m)]
    #[pyo3(name = "head")]
    fn head_py(_py: Python<'_>, x: PyReadonlyArrayDyn<'_, PyObject>) -> PyResult<PyObject> {
        head(x.as_array())
    }

    // wrapper of `axpy`
    #[pyfn(m)]
    #[pyo3(name = "axpy")]
    fn axpy_py<'py>(
        py: Python<'py>,
        a: f64,
        x: PyReadonlyArrayDyn<'_, f64>,
        y: PyReadonlyArrayDyn<'_, f64>,
    ) -> &'py PyArrayDyn<f64> {
        let x = x.as_array();
        let y = y.as_array();
        let z = axpy(a, x, y);
        z.into_pyarray(py)
    }

    // wrapper of `mult`
    #[pyfn(m)]
    #[pyo3(name = "mult")]
    fn mult_py(a: f64, mut x: PyReadwriteArrayDyn<f64>) {
        let x = x.as_array_mut();
        mult(a, x);
    }

    // wrapper of `conj`
    #[pyfn(m)]
    #[pyo3(name = "conj")]
    #[pyo3(text_signature = "(py,x,/)")]
    fn conj_py<'py>(
        py: Python<'py>,
        x: PyReadonlyArrayDyn<'_, Complex64>,
    ) -> &'py PyArrayDyn<Complex64> {
        conj(x.as_array()).into_pyarray(py)
    }

    // example of how to extract an array from a dictionary
    #[pyfn(m)]
    fn extract(d: &PyDict) -> f64 {
        let x = d
            .get_item("x")
            .unwrap()
            .downcast::<PyArray1<f64>>()
            .unwrap();

        x.readonly().as_array().sum()
    }

    // example using timedelta64 array
    #[pyfn(m)]
    fn add_minutes_to_seconds(
        mut x: PyReadwriteArray1<Timedelta<units::Seconds>>,
        y: PyReadonlyArray1<Timedelta<units::Minutes>>,
    ) {
        #[allow(deprecated)]
        Zip::from(x.as_array_mut())
            .and(y.as_array())
            .apply(|x, y| *x = (i64::from(*x) + 60 * i64::from(*y)).into());
    }

    // This crate follows a strongly-typed approach to wrapping NumPy arrays
    // while Python API are often expected to work with multiple element types.
    //
    // That kind of limited polymorphis can be recovered by accepting an enumerated type
    // covering the supported element types and dispatching into a generic implementation.
    #[derive(FromPyObject)]
    enum SupportedArray<'py> {
        F64(&'py PyArray1<f64>),
        I64(&'py PyArray1<i64>),
    }

    #[pyfn(m)]
    fn polymorphic_add<'py>(
        x: SupportedArray<'py>,
        y: SupportedArray<'py>,
    ) -> PyResult<&'py PyAny> {
        match (x, y) {
            (SupportedArray::F64(x), SupportedArray::F64(y)) => Ok(generic_add(
                x.readonly().as_array(),
                y.readonly().as_array(),
            )
            .into_pyarray(x.py())
            .into()),
            (SupportedArray::I64(x), SupportedArray::I64(y)) => Ok(generic_add(
                x.readonly().as_array(),
                y.readonly().as_array(),
            )
            .into_pyarray(x.py())
            .into()),
            (SupportedArray::F64(x), SupportedArray::I64(y))
            | (SupportedArray::I64(y), SupportedArray::F64(x)) => {
                let y = y.cast::<f64>(false)?;

                Ok(
                    generic_add(x.readonly().as_array(), y.readonly().as_array())
                        .into_pyarray(x.py())
                        .into(),
                )
            }
        }
    }

    Ok(())
}

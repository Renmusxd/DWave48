pub mod graph;
use pyo3::prelude::*;
use graph::*;
use rayon::prelude::*;


#[pyfunction]
fn run_monte_carlo(beta: f64, timesteps: usize, num_experiments: u64, edges: Vec<((usize, usize), f64)>, biases: Vec<f64>) -> Vec<(f64, Vec<bool>)>{
    (0 .. num_experiments).into_par_iter().map(|_| {
        let mut gs = GraphState::new(&edges, &biases);
        for _ in 0 .. timesteps {
            gs.do_time_step(beta).unwrap()
        }
        let e = gs.get_energy();
        (e, gs.get_state())
    }).collect()
}

#[pymodule]
fn monte_carlo(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(pyo3::wrap_pyfunction!(run_monte_carlo))?;
    Ok(())
}

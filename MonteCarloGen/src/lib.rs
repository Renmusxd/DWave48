pub mod graph;
use graph::*;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn run_monte_carlo(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    only_basic_moves: Option<bool>
) -> Vec<(f64, Vec<bool>)> {
    let only_basic_moves = only_basic_moves.unwrap_or(false);
    (0..num_experiments)
        .into_par_iter()
        .map(|_| {
            let mut gs = GraphState::new(&edges, &biases);
            for _ in 0..timesteps {
                gs.do_time_step(beta, only_basic_moves).unwrap()
            }
            let e = gs.get_energy();
            (e, gs.get_state())
        })
        .collect()
}

#[pyfunction]
fn run_monte_carlo_annealing(
    mut betas: Vec<(usize, f64)>,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    only_basic_moves: Option<bool>
) -> Vec<(f64, Vec<bool>)> {
    let only_basic_moves = only_basic_moves.unwrap_or(false);
    betas.sort_by_key(|(i, _)| *i);
    if betas.is_empty() {
        betas.push((0, 1.0));
        betas.push((timesteps, 1.0));
    }
    // Make first stop correspond to 0 timestep
    let (i, v) = betas[0];
    if i > 0 {
        betas.insert(0, (0, v));
    }
    // Make last stop correspond to max timestep
    let (i, v) = betas[betas.len() - 1];
    if i < timesteps {
        betas.push((timesteps, v));
    }

    (0..num_experiments)
        .into_par_iter()
        .map(|_| {
            let mut gs = GraphState::new(&edges, &biases);
            let mut beta_index = 0;
            for i in 0..timesteps {
                while i > betas[beta_index + 1].0 {
                    beta_index += 1;
                }
                let (ia, va) = betas[beta_index];
                let (ib, vb) = betas[beta_index + 1];
                let beta = (vb - va) * ((i - ia) as f64 / (ib - ia) as f64) + va;
                gs.do_time_step(beta, only_basic_moves).unwrap()
            }
            let e = gs.get_energy();
            (e, gs.get_state())
        })
        .collect()
}

#[pymodule]
fn monte_carlo(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(pyo3::wrap_pyfunction!(run_monte_carlo))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(run_monte_carlo_annealing))?;
    Ok(())
}

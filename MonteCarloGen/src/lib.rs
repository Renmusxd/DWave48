pub mod graph;
pub mod live_ops;
pub mod qmc;
pub mod qmc_traits;
pub mod qmc_types;
pub mod simple_ops;
use graph::*;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::max;

#[pyfunction]
fn run_monte_carlo(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    only_basic_moves: Option<bool>,
) -> Vec<(f64, Vec<bool>)> {
    let only_basic_moves = only_basic_moves.unwrap_or(false);
    (0..num_experiments)
        .into_par_iter()
        .map(|_| {
            let mut gs = GraphState::new(&edges, &biases);
            (0..timesteps).for_each(|_| gs.do_time_step(beta, only_basic_moves).unwrap());
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
    only_basic_moves: Option<bool>,
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
            (0..timesteps).for_each(|_| {
                while i > betas[beta_index + 1].0 {
                    beta_index += 1;
                }
                let (ia, va) = betas[beta_index];
                let (ib, vb) = betas[beta_index + 1];
                let beta = (vb - va) * ((i - ia) as f64 / (ib - ia) as f64) + va;
                gs.do_time_step(beta, only_basic_moves).unwrap()
            });
            let e = gs.get_energy();
            (e, gs.get_state())
        })
        .collect()
}

#[pyfunction]
fn run_monte_carlo_annealing_and_get_energies(
    mut betas: Vec<(usize, f64)>,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    only_basic_moves: Option<bool>,
) -> Vec<(Vec<f64>, Vec<bool>)> {
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

            let v = (0..timesteps)
                .map(|_| {
                    while i > betas[beta_index + 1].0 {
                        beta_index += 1;
                    }
                    let (ia, va) = betas[beta_index];
                    let (ib, vb) = betas[beta_index + 1];
                    let beta = (vb - va) * ((i - ia) as f64 / (ib - ia) as f64) + va;
                    gs.do_time_step(beta, only_basic_moves).unwrap();
                    gs.get_energy()
                })
                .collect();
            (v, gs.get_state())
        })
        .collect()
}

#[pyfunction]
fn run_quantum_monte_carlo(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    energy_offset: Option<f64>,
) -> Vec<Vec<bool>> {
    let offset = energy_offset.unwrap_or_else(|| {
        let offset_edge =
            edges.iter().map(|(_, j)| *j).fold(
                0.0,
                |acc, j| {
                    if j.abs() > acc {
                        j.abs()
                    } else {
                        acc
                    }
                },
            );
        let offset_bias = biases
            .iter()
            .cloned()
            .fold(0.0, |acc, h| if h < acc { h } else { acc })
            .abs();
        offset_edge + offset_bias
    });
    (0..num_experiments)
        .into_par_iter()
        .map(|_| {
            let gs = GraphState::new(&edges, &biases);
            let cutoff = biases.len() * max(beta.round() as usize, 1);
            let mut qmc_graph = qmc::new_qmc(gs, cutoff, offset);
            qmc_graph.timesteps(timesteps as u64, beta);
            qmc_graph.into_vec()
        })
        .collect()
}

#[pymodule]
fn monte_carlo(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(pyo3::wrap_pyfunction!(run_monte_carlo))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(run_monte_carlo_annealing))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(
        run_monte_carlo_annealing_and_get_energies
    ))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(run_quantum_monte_carlo))?;
    Ok(())
}

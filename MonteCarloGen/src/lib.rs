pub mod graph;
pub mod sse;
use crate::sse::qmc_transverse_graph::new_transverse_qmc;
use graph::*;
use pyo3::prelude::*;
use rayon::prelude::*;
use sse::qmc_graph;
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

pub fn get_offset(edges: &[(Edge, f64)], biases: &[f64]) -> f64 {
    let offset_edge =
        edges
            .iter()
            .map(|(_, j)| *j)
            .fold(0.0, |acc, j| if j.abs() > acc { j.abs() } else { acc });
    let offset_bias = biases
        .iter()
        .cloned()
        .fold(0.0, |acc, h| if h < acc { h } else { acc })
        .abs();
    offset_edge + offset_bias
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
    let offset = energy_offset.unwrap_or_else(|| get_offset(&edges, &biases));
    let cutoff = biases.len();
    (0..num_experiments)
        .into_par_iter()
        .map(|_| {
            let gs = GraphState::new(&edges, &biases);
            let mut qmc_graph = qmc_graph::new_qmc(gs, cutoff, offset);
            qmc_graph.timesteps(timesteps as u64, beta);
            qmc_graph.into_vec()
        })
        .collect()
}

fn run_and_measure_helper_postprocess<F, I, T>(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    energy_offset: Option<f64>,
    sampling_freq: Option<u64>,
    init: I,
    fold: F,
) -> Vec<(T, f64)>
where
    I: Copy + Send + Sync + Fn() -> T,
    F: Copy + Send + Sync + Fn(T, &[bool], f64) -> T,
    T: Send + Sync,
{
    let offset = energy_offset.unwrap_or_else(|| get_offset(&edges, &biases));
    let cutoff = biases.len();
    (0..num_experiments)
        .into_par_iter()
        .map(|_| {
            let gs = GraphState::new(&edges, &biases);
            let mut qmc_graph = qmc_graph::new_qmc(gs, cutoff, offset);
            let (t, e) = qmc_graph.timesteps_measure(timesteps as u64, beta, init(), fold, sampling_freq);
            (t, e + offset)
        }).collect::<Vec<_>>()
}

fn run_and_measure_helper<F>(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    energy_offset: Option<f64>,
    sampling_freq: Option<u64>,
    f: F,
) -> impl Iterator<Item=(f64, f64)>
where
    F: Copy + Send + Sync + Fn(&[bool]) -> f64,
{
    run_and_measure_helper_postprocess(
        beta,
        timesteps,
        num_experiments,
        edges,
        biases,
        energy_offset,
        sampling_freq,
        || (0.0, 0u64),
        |(acc, step), state, _w| (acc + f(state), step+1)
    ).into_iter().map(|((m, s), e)| (m / s as f64, e))
}

fn run_and_measure_variance_helper<F>(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    energy_offset: Option<f64>,
    sampling_freq: Option<u64>,
    f: F,
) -> impl Iterator<Item=(f64, f64, f64)>
where
    F: Copy + Send + Sync + Fn(&[bool]) -> f64,
{
    run_and_measure_helper_postprocess(
        beta,
        timesteps,
        num_experiments,
        edges,
        biases,
        energy_offset,
        sampling_freq,
        || (vec![], 0u64),
        |(mut acc, s), state, _w| {
            let m = f(state);
            acc.push(m);
            (acc, s + 1)
        },
    ).into_iter().map(|((acc, length), energy)| {
        let len = length as f64;
        let prod = acc.iter().map(|m| m).sum::<f64>();
        let mean = prod / len;
        let variance = acc.into_iter().map(|m| (m - mean).powi(2)).sum::<f64>();
        let variance = variance / len;
        (mean, variance, energy)
    })
}

#[pyfunction]
fn run_quantum_monte_carlo_and_measure_spins(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    spin_measurement: Option<(f64, f64)>,
    energy_offset: Option<f64>,
    sampling_freq: Option<u64>,
) -> Vec<(f64, f64)> {
    let (down_m, up_m) = spin_measurement.unwrap_or((-1.0, 1.0));
    run_and_measure_helper(
        beta,
        timesteps,
        num_experiments,
        edges,
        biases,
        energy_offset,
        sampling_freq,
        |state| {
            state
                .iter()
                .fold(0.0, |acc, b| if *b { acc + up_m } else { acc + down_m })
        },
    ).collect::<Vec<_>>()
}

#[pyfunction]
fn run_quantum_monte_carlo_and_measure_spins_and_variance(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    spin_measurement: Option<(f64, f64)>,
    energy_offset: Option<f64>,
    sampling_freq: Option<u64>,
) -> Vec<(f64, f64, f64)> {
    let (down_m, up_m) = spin_measurement.unwrap_or((-1.0, 1.0));
    run_and_measure_variance_helper(
        beta,
        timesteps,
        num_experiments,
        edges,
        biases,
        energy_offset,
        sampling_freq,
        |state| {
            state
                .iter()
                .fold(0.0, |acc, b| if *b { acc + up_m } else { acc + down_m })
        },
    ).collect::<Vec<_>>()
}

#[pyfunction]
fn run_quantum_monte_carlo_and_measure_edges(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    edge_measurement: Option<(f64, f64, f64, f64)>,
    energy_offset: Option<f64>,
    sampling_freq: Option<u64>,
) -> Vec<(f64, f64)> {
    let edge_measurement = edge_measurement.unwrap_or((1.0, -1.0, -1.0, 1.0));
    run_and_measure_helper(
        beta,
        timesteps,
        num_experiments,
        edges.clone(),
        biases,
        energy_offset,
        sampling_freq,
        |state| {
            edges.iter().fold(0.0, |acc, ((vara, varb), j)| {
                acc + *j
                    * match (state[*vara], state[*varb]) {
                        (false, false) => edge_measurement.0,
                        (false, true) => edge_measurement.1,
                        (true, false) => edge_measurement.2,
                        (true, true) => edge_measurement.3,
                    }
            })
        },
    ).collect::<Vec<_>>()
}

#[pyfunction]
fn run_quantum_monte_carlo_and_measure_edges_and_variance(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    edge_measurement: Option<(f64, f64, f64, f64)>,
    energy_offset: Option<f64>,
    sampling_freq: Option<u64>,
) -> Vec<(f64, f64, f64)> {
    let edge_measurement = edge_measurement.unwrap_or((1.0, -1.0, -1.0, 1.0));
    run_and_measure_variance_helper(
        beta,
        timesteps,
        num_experiments,
        edges.clone(),
        biases,
        energy_offset,
        sampling_freq,
        |state| {
            edges.iter().fold(0.0, |acc, ((vara, varb), j)| {
                acc + *j
                    * match (state[*vara], state[*varb]) {
                        (false, false) => edge_measurement.0,
                        (false, true) => edge_measurement.1,
                        (true, false) => edge_measurement.2,
                        (true, true) => edge_measurement.3,
                    }
            })
        },
    ).collect::<Vec<_>>()
}

#[pyfunction]
fn run_transverse_quantum_monte_carlo(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    nvars: usize,
    transverse: f64,
    energy_offset: Option<f64>,
    use_loop_update: Option<bool>,
) -> Vec<(f64, Vec<bool>)> {
    let biases = vec![0.0; nvars];
    let offset = energy_offset.unwrap_or_else(|| get_offset(&edges, &biases));
    let use_loop_update = use_loop_update.unwrap_or(false);
    (0..num_experiments)
        .into_par_iter()
        .map(|_| {
            let gs = GraphState::new(&edges, &biases);
            let cutoff = biases.len() * max(beta.round() as usize, 1);
            let mut qmc_graph = new_transverse_qmc(gs, transverse, cutoff, offset, use_loop_update);
            let average_energy = qmc_graph.timesteps(timesteps as u64, beta);
            (average_energy + offset, qmc_graph.into_vec())
        })
        .collect()
}

#[pyfunction]
fn run_transverse_quantum_monte_carlo_and_measure_spins(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    nvars: usize,
    transverse: f64,
    spin_measurement: Option<(f64, f64)>,
    energy_offset: Option<f64>,
    use_loop_update: Option<bool>,
) -> Vec<(f64, f64)> {
    let biases = vec![0.0; nvars];
    let offset = energy_offset.unwrap_or_else(|| get_offset(&edges, &biases));
    let use_loop_update = use_loop_update.unwrap_or(false);
    let cutoff = biases.len();
    let (down_m, up_m) = spin_measurement.unwrap_or((-1.0, 1.0));
    (0..num_experiments)
        .into_par_iter()
        .map(|_| {
            let gs = GraphState::new(&edges, &biases);
            let mut qmc_graph = new_transverse_qmc(gs, transverse, cutoff, offset, use_loop_update);
            let ((measure, steps), average_energy) = qmc_graph.timesteps_measure(
                timesteps as u64,
                beta,
                (0.0, 0),
                |(acc, step), state, _| {
                    let acc = state
                        .iter()
                        .fold(0.0, |acc, b| if *b { acc + up_m } else { acc + down_m })
                        + acc;
                    (acc, step+1)
                },
                None,
            );
            (measure / steps as f64, average_energy + offset)
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
    m.add_wrapped(pyo3::wrap_pyfunction!(
        run_quantum_monte_carlo_and_measure_spins
    ))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(
        run_quantum_monte_carlo_and_measure_edges
    ))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(
        run_quantum_monte_carlo_and_measure_spins_and_variance
    ))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(
        run_quantum_monte_carlo_and_measure_edges_and_variance
    ))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(run_transverse_quantum_monte_carlo))?;
    m.add_wrapped(pyo3::wrap_pyfunction!(
        run_transverse_quantum_monte_carlo_and_measure_spins
    ))?;
    Ok(())
}

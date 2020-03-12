extern crate monte_carlo;
use monte_carlo::graph::*;
use monte_carlo::sse::qmc_graph::new_qmc;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::min;

#[pyclass]
struct Lattice {
    nvars: usize,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    transverse: Option<f64>,
}

#[pymethods]
impl Lattice {
    #[new]
    fn new_lattice(obj: &PyRawObject, nvars: usize, edges: Vec<((usize, usize), f64)>) {
        let biases = vec![0.0; nvars];
        obj.init({
            Lattice {
                nvars,
                edges,
                biases,
                transverse: None,
            }
        });
    }

    fn set_bias(&mut self, var: usize, bias: f64) -> PyResult<()> {
        if var < self.nvars {
            self.biases[var] = bias;
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::ValueError, String>(format!(
                "Index out of bounds: variable {} out of {}",
                var, self.nvars
            )))
        }
    }

    fn set_transverse_field(&mut self, transverse: f64) {
        self.transverse = Some(transverse)
    }

    fn run_monte_carlo(
        &self,
        beta: f64,
        timesteps: usize,
        num_experiments: usize,
        only_basic_moves: Option<bool>,
    ) -> PyResult<Vec<(f64, Vec<bool>)>> {
        if self.transverse.is_none() {
            let only_basic_moves = only_basic_moves.unwrap_or(false);
            let res = (0..num_experiments)
                .into_par_iter()
                .map(|_| {
                    let mut gs = GraphState::new(&self.edges, &self.biases);
                    (0..timesteps).for_each(|_| gs.do_time_step(beta, only_basic_moves).unwrap());
                    let e = gs.get_energy();
                    (e, gs.get_state())
                })
                .collect();
            Ok(res)
        } else {
            Err(PyErr::new::<pyo3::exceptions::ValueError, String>(
                "Cannot run classic monte carlo with transverse field".to_string(),
            ))
        }
    }

    fn run_monte_carlo_annealing(
        &self,
        mut betas: Vec<(usize, f64)>,
        timesteps: usize,
        num_experiments: usize,
        only_basic_moves: Option<bool>,
    ) -> PyResult<Vec<(f64, Vec<bool>)>> {
        if self.transverse.is_none() {
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

            let res = (0..num_experiments)
                .into_par_iter()
                .map(|_| {
                    let mut gs = GraphState::new(&self.edges, &self.biases);
                    let mut beta_index = 0;
                    (0..timesteps)
                        .try_for_each(|_| {
                            while i > betas[beta_index + 1].0 {
                                beta_index += 1;
                            }
                            let (ia, va) = betas[beta_index];
                            let (ib, vb) = betas[beta_index + 1];
                            let beta = (vb - va) * ((i - ia) as f64 / (ib - ia) as f64) + va;
                            gs.do_time_step(beta, only_basic_moves)
                        })
                        .unwrap();
                    let e = gs.get_energy();
                    (e, gs.get_state())
                })
                .collect();
            Ok(res)
        } else {
            Err(PyErr::new::<pyo3::exceptions::ValueError, String>(
                "Cannot run classic monte carlo with transverse field".to_string(),
            ))
        }
    }

    fn run_monte_carlo_annealing_and_get_energies(
        &self,
        mut betas: Vec<(usize, f64)>,
        timesteps: usize,
        num_experiments: usize,
        only_basic_moves: Option<bool>,
    ) -> PyResult<Vec<(Vec<f64>, Vec<bool>)>> {
        if self.transverse.is_none() {
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

            let res = (0..num_experiments)
                .into_par_iter()
                .map(|_| {
                    let mut gs = GraphState::new(&self.edges, &self.biases);
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
                .collect();
            Ok(res)
        } else {
            Err(PyErr::new::<pyo3::exceptions::ValueError, String>(
                "Cannot run classic monte carlo with transverse field".to_string(),
            ))
        }
    }

    fn run_quantum_monte_carlo(
        &self,
        beta: f64,
        timesteps: u64,
        num_experiments: usize,
        use_loop_update: Option<bool>,
        use_heatbath_diagonal_update: Option<bool>,
    ) -> PyResult<Vec<(Vec<bool>, f64)>> {
        if self.biases.iter().any(|b| *b != 0.0) {
            Err(PyErr::new::<pyo3::exceptions::ValueError, String>(
                "Cannot run quantum monte carlo with spin biases".to_string(),
            ))
        } else {
            match self.transverse {
                None => Err(PyErr::new::<pyo3::exceptions::ValueError, String>(
                    "Cannot run quantum monte carlo without transverse field.".to_string(),
                )),
                Some(transverse) => {
                    let use_loop_update = use_loop_update.unwrap_or(false);
                    let use_heatbath_diagonal_update =
                        use_heatbath_diagonal_update.unwrap_or(false);
                    let res = (0..num_experiments)
                        .into_par_iter()
                        .map(|_| {
                            let gs = GraphState::new(&self.edges, &self.biases);
                            let cutoff = self.nvars;
                            let mut qmc_graph = new_qmc(
                                gs,
                                transverse,
                                cutoff,
                                use_loop_update,
                                use_heatbath_diagonal_update,
                            );
                            let average_energy = qmc_graph.timesteps(timesteps, beta);
                            (qmc_graph.into_vec(), average_energy)
                        })
                        .collect();
                    Ok(res)
                }
            }
        }
    }

    fn run_quantum_monte_carlo_sampling(
        &self,
        beta: f64,
        timesteps: u64,
        num_experiments: usize,
        sampling_wait_buffer: Option<u64>,
        sampling_freq: Option<u64>,
        use_loop_update: Option<bool>,
        use_heatbath_diagonal_update: Option<bool>,
    ) -> PyResult<Vec<(Vec<Vec<bool>>, f64)>> {
        if self.biases.iter().any(|b| *b != 0.0) {
            Err(PyErr::new::<pyo3::exceptions::ValueError, String>(
                "Cannot run quantum monte carlo with spin biases".to_string(),
            ))
        } else {
            match self.transverse {
                None => Err(PyErr::new::<pyo3::exceptions::ValueError, String>(
                    "Cannot run quantum monte carlo without transverse field.".to_string(),
                )),
                Some(transverse) => {
                    let use_loop_update = use_loop_update.unwrap_or(false);
                    let use_heatbath_diagonal_update =
                        use_heatbath_diagonal_update.unwrap_or(false);
                    let sampling_wait_buffer =
                        sampling_wait_buffer.map(|wait| min(wait, timesteps));
                    let res = (0..num_experiments)
                        .into_par_iter()
                        .map(|_| {
                            let gs = GraphState::new(&self.edges, &self.biases);
                            let cutoff = self.nvars;
                            let mut qmc_graph = new_qmc(
                                gs,
                                transverse,
                                cutoff,
                                use_loop_update,
                                use_heatbath_diagonal_update,
                            );
                            let wait = if let Some(wait) = sampling_wait_buffer {
                                qmc_graph.timesteps(wait, beta);
                                wait
                            } else {
                                0
                            };

                            qmc_graph.timesteps_sample(timesteps - wait, beta, sampling_freq)
                        })
                        .collect::<Vec<_>>();
                    Ok(res)
                }
            }
        }
    }

    fn run_quantum_monte_carlo_and_measure_spins(
        &self,
        beta: f64,
        timesteps: usize,
        num_experiments: usize,
        spin_measurement: Option<(f64, f64)>,
        use_loop_update: Option<bool>,
        use_heatbath_diagonal_update: Option<bool>,
        exponent: Option<i32>,
        sampling_freq: Option<u64>,
    ) -> PyResult<Vec<(f64, f64)>> {
        if self.biases.iter().any(|b| *b != 0.0) {
            Err(PyErr::new::<pyo3::exceptions::ValueError, String>(
                "Cannot run quantum monte carlo with spin biases".to_string(),
            ))
        } else {
            match self.transverse {
                None => Err(PyErr::new::<pyo3::exceptions::ValueError, String>(
                    "Cannot run quantum monte carlo without transverse field.".to_string(),
                )),
                Some(transverse) => {
                    let use_loop_update = use_loop_update.unwrap_or(false);
                    let use_heatbath_diagonal_update =
                        use_heatbath_diagonal_update.unwrap_or(false);
                    let cutoff = self.nvars;
                    let (down_m, up_m) = spin_measurement.unwrap_or((-1.0, 1.0));
                    let exponent = exponent.unwrap_or(1);
                    let res = (0..num_experiments)
                        .into_par_iter()
                        .map(|_| {
                            let gs = GraphState::new(&self.edges, &self.biases);
                            let mut qmc_graph = new_qmc(
                                gs,
                                transverse,
                                cutoff,
                                use_loop_update,
                                use_heatbath_diagonal_update,
                            );
                            let ((measure, steps), average_energy) = qmc_graph.timesteps_measure(
                                timesteps as u64,
                                beta,
                                (0.0, 0),
                                |(acc, step), state, _| {
                                    let acc = state
                                        .iter()
                                        .fold(
                                            0.0,
                                            |acc, b| if *b { acc + up_m } else { acc + down_m },
                                        )
                                        .powi(exponent)
                                        + acc;
                                    (acc, step + 1)
                                },
                                sampling_freq,
                            );
                            (measure / steps as f64, average_energy)
                        })
                        .collect();
                    Ok(res)
                }
            }
        }
    }
}

#[pymodule]
fn py_monte_carlo(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Lattice>()?;
    Ok(())
}

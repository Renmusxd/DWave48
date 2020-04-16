extern crate monte_carlo;
use monte_carlo::graph::*;
use monte_carlo::sse::qmc_graph::new_qmc;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::{max, min};

#[pyclass]
struct Lattice {
    nvars: usize,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    transverse: Option<f64>,
    initial_state: Option<Vec<bool>>,
}

/// A lattice for running monte carlo simulations. Takes a list of edges: ((a, b), j), ...
#[pymethods]
impl Lattice {
    /// Make a new lattice with `edges`, positive implies antiferromagnetic bonds, negative is
    /// ferromagnetic.
    #[new]
    fn new_lattice(edges: Vec<((usize, usize), f64)>) -> PyResult<Self> {
        let nvars = edges
            .iter()
            .map(|((a, b), _)| max(*a, *b))
            .max()
            .map(|x| x + 1);
        if let Some(nvars) = nvars {
            Ok(Lattice {
                nvars,
                edges,
                biases: vec![0.0; nvars],
                transverse: None,
                initial_state: None,
            })
        } else {
            Err(PyErr::new::<pyo3::exceptions::ValueError, String>(
                "Must supply some edges for graph".to_string(),
            ))
        }
    }

    /// Set the bias of the variable `var` to `bias`.
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

    /// Set the transverse field on the system to `transverse`
    fn set_transverse_field(&mut self, transverse: f64) -> PyResult<()> {
        if transverse > 0.0 {
            self.transverse = Some(transverse);
            Ok(())
        } else if transverse == 0.0 {
            self.transverse = None;
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::ValueError, String>(
                "Transverse field must be positive".to_string(),
            ))
        }
    }

    /// Set the initial state for monte carlo, if it's empty then choose a random state each time.
    fn set_initial_state(&mut self, initial_state: Vec<bool>) -> PyResult<()> {
        if initial_state.len() == self.biases.len() {
            self.initial_state = Some(initial_state);
            Ok(())
        } else if initial_state.is_empty() {
            self.initial_state = None;
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::ValueError, String>(
                "Initial state must be of the same size as biases, or 0.".to_string(),
            ))
        }
    }

    /// Run a classical monte carlo simulation.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `num_experiments`: number of simultaneous experiments to run.
    /// * `only_basic_moves`: disallow things other than simple spin flips.
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
                    if let Some(s) = &self.initial_state {
                        gs.set_state(s.clone())
                    };
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

    /// Run a classical monte carlo simulation with an annealing schedule.
    ///
    /// # Arguments:
    /// * `betas`: (t,E/kt) array to use for the simulation, interpolates between times linearly.
    /// * `timesteps`: number of timesteps to run.
    /// * `num_experiments`: number of simultaneous experiments to run.
    /// * `only_basic_moves`: disallow things other than simple spin flips.
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
                    if let Some(s) = &self.initial_state {
                        gs.set_state(s.clone())
                    };
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

    /// Run a classical monte carlo simulation with an annealing schedule, returns energies too.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `num_experiments`: number of simultaneous experiments to run.
    /// * `only_basic_moves`: disallow things other than simple spin flips.
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
                    if let Some(s) = &self.initial_state {
                        gs.set_state(s.clone())
                    };
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

    /// Run a quantum monte carlo simulation, return the final state and average energy.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `num_experiments`: number of simultaneous experiments to run.
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
                    let cutoff = self.nvars;
                    let res = (0..num_experiments)
                        .into_par_iter()
                        .map(|_| {
                            let mut qmc_graph = new_qmc(
                                self.edges.clone(),
                                transverse,
                                cutoff,
                                use_loop_update,
                                use_heatbath_diagonal_update,
                                self.initial_state.clone(),
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

    /// Run a quantum monte carlo simulation, sample the state at each `sampling_freq`, returns that
    /// array plus the average energy.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `num_experiments`: number of simultaneous experiments to run.
    /// * `sampling_wait_buffer`: timesteps to wait before sampling begins.
    /// * `sampling_freq`: frequency of sampling in number of timesteps.
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
                    let cutoff = self.nvars;
                    let res = (0..num_experiments)
                        .into_par_iter()
                        .map(|_| {
                            let mut qmc_graph = new_qmc(
                                self.edges.clone(),
                                transverse,
                                cutoff,
                                use_loop_update,
                                use_heatbath_diagonal_update,
                                self.initial_state.clone(),
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

    /// Run a quantum monte carlo simulation, get the variable's autocorrelation across time for
    /// each experiment.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `num_experiments`: number of simultaneous experiments to run.
    /// * `sampling_wait_buffer`: timesteps to wait before sampling begins.
    /// * `sampling_freq`: frequency of sampling in number of timesteps.
    fn run_quantum_monte_carlo_and_measure_variable_autocorrelation(
        &self,
        beta: f64,
        timesteps: u64,
        num_experiments: usize,
        sampling_wait_buffer: Option<u64>,
        sampling_freq: Option<u64>,
        use_loop_update: Option<bool>,
        use_heatbath_diagonal_update: Option<bool>,
        use_fft: Option<bool>,
    ) -> PyResult<Vec<Vec<f64>>> {
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
                    let sampling_wait_buffer = sampling_wait_buffer.unwrap_or(0);
                    let cutoff = self.nvars;
                    let res = (0..num_experiments)
                        .into_par_iter()
                        .map(|_| {
                            let mut qmc_graph = new_qmc(
                                self.edges.clone(),
                                transverse,
                                cutoff,
                                use_loop_update,
                                use_heatbath_diagonal_update,
                                self.initial_state.clone(),
                            );

                            if sampling_wait_buffer > 0 {
                                qmc_graph.timesteps(sampling_wait_buffer, beta);
                            }

                            qmc_graph.calculate_variable_autocorrelation(
                                timesteps,
                                beta,
                                sampling_freq,
                                use_fft,
                            )
                        })
                        .collect::<Vec<_>>();
                    Ok(res)
                }
            }
        }
    }

    /// Run a quantum monte carlo simulation, get the bond's autocorrelation across time for each
    /// experiment.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `num_experiments`: number of simultaneous experiments to run.
    /// * `sampling_wait_buffer`: timesteps to wait before sampling begins.
    /// * `sampling_freq`: frequency of sampling in number of timesteps.
    fn run_quantum_monte_carlo_and_measure_bond_autocorrelation(
        &self,
        beta: f64,
        timesteps: u64,
        num_experiments: usize,
        sampling_wait_buffer: Option<u64>,
        sampling_freq: Option<u64>,
        use_loop_update: Option<bool>,
        use_heatbath_diagonal_update: Option<bool>,
        use_fft: Option<bool>,
    ) -> PyResult<Vec<Vec<f64>>> {
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
                    let sampling_wait_buffer = sampling_wait_buffer.unwrap_or(0);
                    let cutoff = self.nvars;
                    let res = (0..num_experiments)
                        .into_par_iter()
                        .map(|_| {
                            let mut qmc_graph = new_qmc(
                                self.edges.clone(),
                                transverse,
                                cutoff,
                                use_loop_update,
                                use_heatbath_diagonal_update,
                                self.initial_state.clone(),
                            );

                            if sampling_wait_buffer > 0 {
                                qmc_graph.timesteps(sampling_wait_buffer, beta);
                            }

                            qmc_graph.calculate_bond_autocorrelation(
                                timesteps,
                                beta,
                                sampling_freq,
                                use_fft,
                            )
                        })
                        .collect::<Vec<_>>();
                    Ok(res)
                }
            }
        }
    }

    /// Run a quantum monte carlo simulation, measure the spins using a given 2x2 matrix, then sum
    /// and raise to the given exponent.
    ///
    /// # Arguments:
    /// * `beta`: E/kt to use for the simulation.
    /// * `timesteps`: number of timesteps to run.
    /// * `num_experiments`: number of simultaneous experiments to run.
    /// * `sampling_freq`: frequency of sampling in number of timesteps.
    /// * `exponent`: defaults to 1.
    fn run_quantum_monte_carlo_and_measure_spins(
        &self,
        beta: f64,
        timesteps: usize,
        num_experiments: usize,
        sampling_freq: Option<u64>,
        sampling_wait_buffer: Option<u64>,
        spin_measurement: Option<(f64, f64)>,
        use_loop_update: Option<bool>,
        use_heatbath_diagonal_update: Option<bool>,
        exponent: Option<i32>,
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
                            let mut qmc_graph = new_qmc(
                                self.edges.clone(),
                                transverse,
                                cutoff,
                                use_loop_update,
                                use_heatbath_diagonal_update,
                                self.initial_state.clone(),
                            );

                            let wait = if let Some(wait) = sampling_wait_buffer {
                                qmc_graph.timesteps(wait, beta);
                                wait
                            } else {
                                0
                            };
                            let ((measure, steps), average_energy) = qmc_graph.timesteps_measure(
                                timesteps as u64 - wait,
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

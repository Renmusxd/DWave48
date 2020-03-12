use crate::graph::GraphState;
use crate::sse::qmc_traits::*;
use crate::sse::simple_ops::*;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::cmp::max;

type VecEdge = Vec<usize>;
pub struct QMCGraph<R: Rng> {
    edges: Vec<(VecEdge, f64)>,
    transverse: f64,
    state: Option<Vec<bool>>,
    cutoff: usize,
    op_manager: Option<SimpleOpDiagonal>,
    twosite_energy_offset: f64,
    singlesite_energy_offset: f64,
    use_loop_update: bool,
    use_heatbath_diagonal_update: bool,
    rng: R,
}

pub fn new_qmc(
    graph: GraphState,
    transverse: f64,
    cutoff: usize,
    use_loop_update: bool,
    use_heatbath_diagonal_update: bool,
) -> QMCGraph<ThreadRng> {
    let rng = rand::thread_rng();
    QMCGraph::<ThreadRng>::new_with_rng(
        graph,
        transverse,
        cutoff,
        use_loop_update,
        use_heatbath_diagonal_update,
        rng,
    )
}

impl<R: Rng> QMCGraph<R> {
    pub fn new_with_rng<Rg: Rng>(
        graph: GraphState,
        transverse: f64,
        cutoff: usize,
        use_loop_update: bool,
        use_heatbath_diagonal_update: bool,
        rng: Rg,
    ) -> QMCGraph<Rg> {
        assert!(graph.biases.into_iter().all(|v| v == 0.0));

        let edges = graph
            .edges
            .into_iter()
            .map(|((a, b), j)| (vec![a, b], j))
            .collect::<Vec<_>>();
        let twosite_energy_offset = edges
            .iter()
            .fold(None, |acc, (_, j)| match acc {
                None => Some(*j),
                Some(acc) => Some(if *j < acc { *j } else { acc }),
            })
            .unwrap_or(0.0)
            .abs();
        let singlesite_energy_offset = transverse;

        let state = graph.state;
        let mut ops = SimpleOpDiagonal::new(state.as_ref().map_or(0, |s| s.len()));
        ops.set_min_size(cutoff);
        QMCGraph::<Rg> {
            edges,
            transverse,
            state,
            op_manager: Some(ops),
            cutoff,
            twosite_energy_offset,
            singlesite_energy_offset,
            use_loop_update,
            use_heatbath_diagonal_update,
            rng,
        }
    }

    pub fn timesteps(&mut self, t: u64, beta: f64) -> f64 {
        let (_, average_energy) =
            self.timesteps_measure(t, beta, (), |_acc, _state, _weight| (), None);
        average_energy
    }

    pub fn timesteps_sample(
        &mut self,
        t: u64,
        beta: f64,
        sampling_freq: Option<u64>,
    ) -> (Vec<Vec<bool>>, f64) {
        let acc = Vec::with_capacity((t / sampling_freq.unwrap_or(1) + 1) as usize);
        self.timesteps_measure(
            t,
            beta,
            acc,
            |mut acc, state, _weight| {
                acc.push(state.to_vec());
                acc
            },
            sampling_freq,
        )
    }

    pub fn timesteps_measure<F, T>(
        &mut self,
        timesteps: u64,
        beta: f64,
        init_t: T,
        state_fold: F,
        sampling_freq: Option<u64>,
    ) -> (T, f64)
    where
        F: Fn(T, &[bool], f64) -> T,
    {
        let mut state = self.state.take().unwrap();
        let nvars = state.len();
        let edges = &self.edges;
        let transverse = self.transverse;
        let twosite_energy_offset = self.twosite_energy_offset;
        let singlesite_energy_offset = self.singlesite_energy_offset;
        let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            if vars.len() == 2 {
                two_site_hamiltonian(
                    (input_state[0], input_state[1]),
                    (output_state[0], output_state[1]),
                    edges[bond].1,
                    transverse,
                    twosite_energy_offset,
                )
            } else if vars.len() == 1 {
                single_site_hamiltonian(
                    input_state[0],
                    output_state[0],
                    transverse,
                    singlesite_energy_offset,
                )
            } else {
                unreachable!()
            }
        };

        let mut acc = init_t;
        let mut steps_measured = 0;
        let mut total_n = 0;
        let sampling_freq = sampling_freq.unwrap_or(1);
        let vars = (0..nvars).collect::<Vec<_>>();

        let num_bonds = edges.len() + nvars;
        let bonds_fn = |b: usize| -> &[usize] {
            if b < edges.len() {
                &edges[b].0
            } else {
                let b = b - edges.len();
                &vars[b..b + 1]
            }
        };

        for t in 0..timesteps {
            // Start by editing the ops list
            let mut manager = self.op_manager.take().unwrap();
            let rng = &mut self.rng;

            let bond_weights = if self.use_heatbath_diagonal_update {
                Some(SimpleOpDiagonal::make_bond_weights(
                    &state, h, num_bonds, bonds_fn,
                ))
            } else {
                None
            };
            manager.make_diagonal_update_with_rng(
                self.cutoff,
                beta,
                &state,
                h,
                (num_bonds, bonds_fn, bond_weights),
                rng,
            );
            self.cutoff = max(self.cutoff, manager.get_n() + manager.get_n() / 2);

            let mut manager = manager.convert_to_looper();
            // Now we can do loop updates easily.
            if self.use_loop_update {
                let state_updates = manager.make_loop_update_with_rng(None, h, rng);
                state_updates.into_iter().for_each(|(i, v)| {
                    state[i] = v;
                });
            }

            let state_updates = manager.flip_each_cluster_rng(0.5, rng);
            state_updates.into_iter().for_each(|(i, v)| {
                state[i] = v;
            });

            let manager = manager;
            state.iter_mut().enumerate().for_each(|(var, state)| {
                if !manager.does_var_have_ops(var) && rng.gen_bool(0.5) {
                    *state = !*state;
                }
            });

            // Ignore first one.
            if (t + 1) % sampling_freq == 0 {
                let weight = manager.weight(h);
                acc = state_fold(acc, &state, weight);
                steps_measured += 1;
                total_n += manager.get_n();
            }

            self.op_manager = Some(manager.convert_to_diagonal());
        }
        self.state = Some(state);
        let average_energy = -(total_n as f64 / (steps_measured as f64 * beta));
        // Get total energy offset (num_vars * singlesite + num_edges * twosite)
        let offset =
            twosite_energy_offset * edges.len() as f64 + singlesite_energy_offset * nvars as f64;
        (acc, average_energy + offset)
    }

    pub fn clone_state(&self) -> Vec<bool> {
        self.state.as_ref().unwrap().clone()
    }

    pub fn into_vec(self) -> Vec<bool> {
        self.state.unwrap()
    }

    pub fn debug_print(&self) {
        let twosite_energy_offset = self.twosite_energy_offset;
        let singlesite_energy_offset = self.singlesite_energy_offset;
        let transverse = self.transverse;
        let edges = &self.edges;
        let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            if vars.len() == 2 {
                two_site_hamiltonian(
                    (input_state[0], input_state[1]),
                    (output_state[0], output_state[1]),
                    edges[bond].1,
                    transverse,
                    twosite_energy_offset,
                )
            } else if vars.len() == 1 {
                single_site_hamiltonian(
                    input_state[0],
                    output_state[0],
                    transverse,
                    singlesite_energy_offset,
                )
            } else {
                unreachable!()
            }
        };
        if let Some(opm) = self.op_manager.as_ref() {
            opm.debug_print(h)
        }
    }
}

fn two_site_hamiltonian(
    inputs: (bool, bool),
    outputs: (bool, bool),
    bond: f64,
    transverse: f64,
    energy_offset: f64,
) -> f64 {
    let matentry = if inputs == outputs {
        energy_offset
            + match inputs {
                (false, false) => -bond,
                (false, true) => bond,
                (true, false) => bond,
                (true, true) => -bond,
            }
    } else {
        let diff_state = (inputs.0 == outputs.0, inputs.1 == outputs.1);
        match diff_state {
            (false, false) => 0.0,
            (true, false) | (false, true) => transverse,
            (true, true) => unreachable!(),
        }
    };
    assert!(matentry >= 0.0);
    matentry
}

fn single_site_hamiltonian(
    input_state: bool,
    output_state: bool,
    transverse: f64,
    energy_offset: f64,
) -> f64 {
    match (input_state, output_state) {
        (false, false) | (true, true) => energy_offset,
        (false, true) | (true, false) => transverse,
    }
}

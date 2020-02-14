use crate::graph::{Edge, GraphState};
use crate::qmc_traits::*;
use crate::simple_ops::*;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::cmp::{min, max};
use std::io::Write;

pub struct QMCGraph<R: Rng> {
    edges: Vec<(Edge, f64)>,
    biases: Vec<f64>,
    transverse_fields: Vec<f64>,
    state: Option<Vec<bool>>,
    cutoff: usize,
    op_manager: Option<SimpleOpDiagonal>,
    energy_offset: f64,
    rng: R,
}

pub fn new_qmc(graph: GraphState, transverse_fields: Vec<f64>, cutoff: usize, energy_offset: f64) -> QMCGraph<ThreadRng> {
    let rng = rand::thread_rng();
    QMCGraph::<ThreadRng>::new_with_rng(graph, transverse_fields, cutoff, energy_offset, rng)
}

impl<R: Rng> QMCGraph<R> {
    pub fn new_with_rng<Rg: Rng>(
        graph: GraphState,
        transverse_fields: Vec<f64>,
        cutoff: usize,
        energy_offset: f64,
        rng: Rg,
    ) -> QMCGraph<Rg> {
        let edges = graph.edges;
        let biases = graph.biases;
        let state = graph.state;
        let mut ops = SimpleOpDiagonal::new(state.as_ref().map_or(0, |s| s.len()));
        ops.set_min_size(cutoff);
        QMCGraph::<Rg> {
            edges,
            biases,
            transverse_fields,
            state,
            op_manager: Some(ops),
            cutoff,
            energy_offset,
            rng,
        }
    }

    pub fn timesteps(&mut self, t: u64, beta: f64) {
        self.timesteps_measure(t, beta, (), |_acc, _state, _weight| (), None);
    }

    pub fn timesteps_measure<F, T>(
        &mut self,
        timesteps: u64,
        beta: f64,
        init_t: T,
        state_fold: F,
        sampling_freq: Option<u64>
    ) -> (T, f64, usize)
    where
        F: Fn(T, &[bool], f64) -> T,
    {
        let mut state = self.state.take().unwrap();
        let edges = &self.edges;
        let biases = &self.biases;
        let transverse_fields = &self.transverse_fields;
        let energy_offset = self.energy_offset;
        let h = |vara: usize,
                 varb: usize,
                 bond: usize,
                 input_state: (bool, bool),
                 output_state: (bool, bool)| {
            hamiltonian(
                (vara, varb),
                bond,
                input_state,
                output_state,
                edges,
                biases,
                transverse_fields,
                energy_offset,
            )
        };

        let mut acc = init_t;
        let mut total_weight = 0.0;
        let mut steps_measured = 0;
        let sampling_freq = sampling_freq.unwrap_or(1);
        for t in 0..timesteps {
            // Start by editing the ops list
            let mut manager = self.op_manager.take().unwrap();
            manager.make_diagonal_update_with_rng(
                self.cutoff,
                beta,
                &state,
                h,
                (edges.len(), |i| edges[i].0),
                &mut self.rng,
            );
            self.cutoff = max(self.cutoff, manager.get_n() + manager.get_n()/2);

            let mut manager = manager.convert_to_looper();
            // Now we can do loop updates easily.
            let state_updates = manager.make_loop_update_with_rng(None, h, &mut self.rng);
            state_updates.into_iter().for_each(|(i, v)| {
                state[i] = v;
            });
            let rng = &mut self.rng;
            let manager = manager;
            state.iter_mut().enumerate().for_each(|(var, state)| {
                if !manager.does_var_have_ops(var) && rng.gen_bool(0.5) {
                    *state = !*state;
                }
            });

            // Ignore first one.
            if (t+1) % sampling_freq == 0 {
                let weight = manager.weight(h);
                acc = state_fold(acc, &state, weight);
                total_weight += weight;
                steps_measured += 1;
            }

            self.op_manager = Some(manager.convert_to_diagonal());
        }
        self.state = Some(state);
        (acc, total_weight, steps_measured)
    }

    pub fn clone_state(&self) -> Vec<bool> {
        self.state.as_ref().unwrap().clone()
    }

    pub fn into_vec(self) -> Vec<bool> {
        self.state.unwrap()
    }
    //
    //    pub fn debug_print(&self) {
    //        let edges = &self.edges;
    //        let biases = &self.biases;
    //        let offset = self.energy_offset;
    //        let h = |vara: usize,
    //                 varb: usize,
    //                 bond: usize,
    //                 input_state: (bool, bool),
    //                 output_state: (bool, bool)| {
    //            hamiltonian(
    //                vara,
    //                varb,
    //                bond,
    //                input_state,
    //                output_state,
    //                edges,
    //                biases,
    //                offset,
    //            )
    //        };
    //        self.op_manager.debug_print(h)
    //    }
    //
    //    pub fn mat_element(&self) -> f64 {
    //        let edges = &self.edges;
    //        let biases = &self.biases;
    //        let offset = self.energy_offset;
    //        let h = |vara: usize,
    //                 varb: usize,
    //                 bond: usize,
    //                 input_state: (bool, bool),
    //                 output_state: (bool, bool)| {
    //            hamiltonian(
    //                vara,
    //                varb,
    //                bond,
    //                input_state,
    //                output_state,
    //                edges,
    //                biases,
    //                offset,
    //            )
    //        };
    //        self.op_manager.total_matrix_weight(h)
    //    }
}

fn hamiltonian(
    vars: (usize, usize),
    bond: usize,
    input_state: (bool, bool),
    output_state: (bool, bool),
    edges: &[(Edge, f64)],
    biases: &[f64],
    transverse_fields: &[f64],
    offset: f64,
) -> f64 {
    let (vara, varb) = vars;
    let matentry = if input_state == output_state {
        offset + match input_state {
            (false, false) => -edges[bond].1,
            (false, true) => edges[bond].1 + biases[varb],
            (true, false) => edges[bond].1 + biases[vara],
            (true, true) => -edges[bond].1 + biases[vara] + biases[varb],
        }
    } else {
        let a_shift = input_state.0 ^ output_state.0;
        let b_shift = input_state.1 ^ output_state.1;
        match (a_shift, b_shift) {
            (false, false) => unreachable!(),  // THis was handled above
            (true, false) => transverse_fields[vara],
            (false, true) => transverse_fields[varb],
            (true, true) => 0.0,
        }
    };
    assert!(matentry >= 0.0);
    matentry
}

#[cfg(test)]
mod qmc_tests {
    use super::*;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn single_timestep() {
        let graph = GraphState::new(&[((0, 1), 1.0)], &[0.0, 0.0]);
        let rng: ChaCha20Rng = rand::SeedableRng::seed_from_u64(12345678);
        let mut qmc = QMCGraph::<ChaCha20Rng>::new_with_rng(graph, vec![0.0, 0.0], 10, 3.0, rng);
        qmc.timesteps(1, 1.0);
    }

    #[test]
    fn many_timestep() {
        let graph = GraphState::new(
            &[((0, 1), 1.0), ((1, 2), 1.0), ((2, 3), 1.0), ((3, 4), 1.0)],
            &[0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let rng: ChaCha20Rng = rand::SeedableRng::seed_from_u64(12345678);
        let mut qmc = QMCGraph::<ChaCha20Rng>::new_with_rng(graph, vec![0.0, 0.0], 10, 3.0, rng);
        qmc.timesteps(1000, 1.0);
        println!("{:?}", qmc.into_vec())
    }
}

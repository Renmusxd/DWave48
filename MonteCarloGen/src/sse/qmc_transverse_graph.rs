use crate::graph::GraphState;
use crate::sse::qmc_traits::*;
use crate::sse::simple_ops::*;
use rand::rngs::ThreadRng;
use rand::Rng;
use std::cmp::max;

type VecEdge = Vec<usize>;
pub struct QMCTransverseGraph<R: Rng> {
    edges: Vec<(VecEdge, f64)>,
    transverse: f64,
    state: Option<Vec<bool>>,
    cutoff: usize,
    op_manager: Option<SimpleOpDiagonal>,
    energy_offset: f64,
    rng: R,
}

pub fn new_transverse_qmc(
    graph: GraphState,
    transverse: f64,
    cutoff: usize,
    energy_offset: f64,
) -> QMCTransverseGraph<ThreadRng> {
    let rng = rand::thread_rng();
    QMCTransverseGraph::<ThreadRng>::new_with_rng(graph, transverse, cutoff, energy_offset, rng)
}

impl<R: Rng> QMCTransverseGraph<R> {
    pub fn new_with_rng<Rg: Rng>(
        graph: GraphState,
        transverse: f64,
        cutoff: usize,
        energy_offset: f64,
        rng: Rg,
    ) -> QMCTransverseGraph<Rg> {
        let edges = graph
            .edges
            .into_iter()
            .map(|((a, b), j)| (vec![a, b], j))
            .collect::<Vec<_>>();
        let state = graph.state;
        let mut ops = SimpleOpDiagonal::new(state.as_ref().map_or(0, |s| s.len()));
        ops.set_min_size(cutoff);
        QMCTransverseGraph::<Rg> {
            edges,
            transverse,
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
        sampling_freq: Option<u64>,
    ) -> (T, f64, usize)
    where
        F: Fn(T, &[bool], f64) -> T,
    {
        let mut state = self.state.take().unwrap();
        let edges = &self.edges;
        let transverse = self.transverse;
        let energy_offset = self.energy_offset;
        let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            hamiltonian(
                (vars[0], vars[1]),
                (input_state[0], input_state[1]),
                (output_state[0], output_state[1]),
                edges[bond].1,
                transverse,
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
                (edges.len(), |i| &edges[i].0),
                &mut self.rng,
            );
            self.cutoff = max(self.cutoff, manager.get_n() + manager.get_n() / 2);

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
            if (t + 1) % sampling_freq == 0 {
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

    pub fn debug_print(&self) {
        let offset = self.energy_offset;
        let transverse = self.transverse;
        let edges = &self.edges;
        let h = |vars: &[usize], bond: usize, input_state: &[bool], output_state: &[bool]| {
            hamiltonian(
                (vars[0], vars[1]),
                (input_state[0], input_state[1]),
                (output_state[0], output_state[1]),
                edges[bond].1,
                transverse,
                offset,
            )
        };
        self.op_manager.as_ref().map(|opm| opm.debug_print(h));
    }
}

fn hamiltonian(
    _vars: (usize, usize),
    input_state: (bool, bool),
    output_state: (bool, bool),
    binding: f64,
    transverse: f64,
    offset: f64,
) -> f64 {
    let matentry = if input_state == output_state {
        offset
            + match input_state {
                (false, false) => -2.0 * transverse,
                (false, true) => 0.0,
                (true, false) => 0.0,
                (true, true) => 2.0 * transverse,
            }
    } else {
        let diff = (
            input_state.0 == output_state.0,
            input_state.1 == output_state.1,
        );
        match diff {
            (false, false) => binding,
            (false, true) => 0.0,
            (true, false) => 0.0,
            (true, true) => unreachable!(),
        }
    };
    assert!(matentry >= 0.0);
    matentry
}

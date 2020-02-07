use crate::graph::{Edge, GraphState};
use crate::qmc_utils::*;
use rand::Rng;

struct QMCGraph {
    edges: Vec<(Edge, f64)>,
    binding_mat: Vec<Vec<(usize, f64)>>,
    biases: Vec<f64>,
    state: Option<Vec<bool>>,
    cutoff: usize,
    ops: FastOps,
}

impl QMCGraph {
    fn new(graph: GraphState, cutoff: usize) -> Self {
        let edges = graph.edges;
        let binding_mat = graph.binding_mat;
        let biases = graph.biases;
        let state = graph.state;
        let mut ops = FastOps::new(state.as_ref().map_or(0, |s| s.len()));
        ops.set_min_size(cutoff);
        Self {
            edges,
            binding_mat,
            biases,
            state,
            ops,
            cutoff,
        }
    }

    fn diagonal_op_updates(
        ops: &mut FastOps,
        state: &[bool],
        edges: &[(Edge, f64)],
        cutoff: usize,
        beta: f64,
    ) {
        unimplemented!()
        //        let mut rng = rand::thread_rng();
        //        // Start by editing the ops list
        //        for p in 0..cutoff {
        //            let op = ops.get_pth(p);
        //            let b = match op {
        //                None => rng.gen_range(0, edges.len()),
        //                Some(Op::Diagonal(_, _, b)) => b,
        //                Some(Op::OffDiagonal(_, _, _)) => continue,
        //            };
        //            let ((vara, varb), j) = edges[b];
        //            let same = state[vara] == state[varb];
        //            let mat_element = if same { j } else { -j };
        //            let numerator = beta * edges.len() as f64 * mat_element;
        //            let denominator = (cutoff - ops.get_size()) as f64;
        //            match op {
        //                None => {
        //                    let prob = numerator / denominator;
        //                    if rng.gen::<f64>() < prob {
        //                        let op = Op::Diagonal(vara, varb, b);
        //                        ops.set_pth(p, Some(op));
        //                    }
        //                }
        //                Some(Op::Diagonal(_, _, b)) => {
        //                    let prob = (denominator + 1.0) / numerator;
        //                    if rng.gen::<f64>() < prob {
        //                        ops.set_pth(p, None);
        //                    }
        //                }
        //                Some(Op::OffDiagonal(_, _, _)) => (),
        //            };
        //        }
    }

    fn timesteps(&mut self, t: u64, beta: f64) {
        self.ops.set_min_size(self.edges.len());
        let mut state = self.state.take().unwrap();

        // Start by editing the ops list
        Self::diagonal_op_updates(&mut self.ops, &state, &self.edges, self.cutoff, beta);

        // Now we can do loop updates easily.
    }
}

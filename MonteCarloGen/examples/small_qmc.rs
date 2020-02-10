extern crate monte_carlo;
use monte_carlo::graph::GraphState;
use monte_carlo::qmc::new_qmc;
use std::cmp::max;

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
        .map(|_| {
            let gs = GraphState::new(&edges, &biases);
            let cutoff = biases.len() * max(beta.round() as usize, 1);
            let mut qmc_graph = new_qmc(gs, cutoff, offset);
            qmc_graph.timesteps(timesteps as u64, beta);
            qmc_graph.into_vec()
        })
        .collect()
}

fn main() {
    run_quantum_monte_carlo(
        100.0,
        1000,
        1000,
        vec![
            ((0, 1), -1.0),
            ((1, 2), -1.0),
            ((2, 3), -1.0),
            ((3, 4), -1.0),
        ],
        vec![1.0, 0.0, 0.0, 0.0, 0.0],
        None,
    );
}

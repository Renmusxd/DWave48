extern crate monte_carlo;
use monte_carlo::get_offset;
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
    let offset = energy_offset.unwrap_or_else(|| get_offset(&edges, &biases));
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

fn run_quantum_monte_carlo_and_measure_spins(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    biases: Vec<f64>,
    spin_measurement: (f64, f64),
    energy_offset: Option<f64>,
) -> Vec<f64> {
    let offset = energy_offset.unwrap_or_else(|| get_offset(&edges, &biases));
    let (down_m, up_m) = spin_measurement;
    let cutoff = biases.len(); // * max(beta.round() as usize, 1);
    (0..num_experiments)
        .map(|_| {
            let gs = GraphState::new(&edges, &biases);
            let mut qmc_graph = new_qmc(gs, cutoff, offset);
            let (measure, weight) =
                qmc_graph.timesteps_measure(timesteps as u64, beta, 0.0, |acc, state, weight| {
                    state
                        .iter()
                        .fold(0.0, |acc, b| if *b { acc + up_m } else { acc + down_m })
                        * weight
                        + acc
                });
            measure / weight
        })
        .collect()
}

fn main() {
    let result = run_quantum_monte_carlo_and_measure_spins(
        10.0,
        10000,
        10,
        vec![
            ((0, 1), -1.0),
            ((1, 2), -1.0),
            ((2, 3), -1.0),
            ((3, 4), -1.0),
        ],
        vec![1.0, 0.0, 0.0, 0.0, 0.0],
        (-1.0, 1.0),
        None,
    );
    println!("{:?}", result)
}

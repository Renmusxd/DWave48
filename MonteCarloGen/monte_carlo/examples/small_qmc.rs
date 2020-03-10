extern crate monte_carlo;
use monte_carlo::graph::GraphState;
use monte_carlo::sse::qmc_graph::new_qmc;
use std::cmp::max;

fn run_transverse_quantum_monte_carlo(
    beta: f64,
    timesteps: usize,
    num_experiments: u64,
    edges: Vec<((usize, usize), f64)>,
    nvars: usize,
    transverse: f64,
) -> Vec<(Vec<bool>, f64)> {
    let biases = vec![0.0; nvars];
    (0..num_experiments)
        .map(|_| {
            let gs = GraphState::new(&edges, &biases);
            let cutoff = biases.len() * max(beta.round() as usize, 1);
            let mut qmc_graph = new_qmc(gs, transverse, cutoff, false, false);
            let e = qmc_graph.timesteps(timesteps as u64, beta);
            qmc_graph.debug_print();
            (qmc_graph.into_vec(), e)
        })
        .collect()
}
//
// fn run_transverse_quantum_monte_carlo_and_measure_spins(
//     beta: f64,
//     timesteps: usize,
//     num_experiments: u64,
//     edges: Vec<((usize, usize), f64)>,
//     nvars: usize,
//     transverse: f64,
//     spin_measurement: Option<(f64, f64)>,
//     energy_offset: Option<f64>,
// ) -> Vec<f64> {
//     let biases = vec![0.0; nvars];
//     let cutoff = biases.len();
//     let (down_m, up_m) = spin_measurement.unwrap_or((-1.0, 1.0));
//     (0..num_experiments)
//         .map(|_| {
//             let gs = GraphState::new(&edges, &biases);
//             let mut qmc_graph = new_qmc(gs, transverse, cutoff, false);
//             let (measure, steps_measured) = qmc_graph.timesteps_measure(
//                 timesteps as u64,
//                 beta,
//                 0.0,
//                 |acc, state, weight| {
//                     state
//                         .iter()
//                         .fold(0.0, |acc, b| if *b { acc + up_m } else { acc + down_m })
//                         + acc
//                 },
//                 None,
//             );
//             measure / steps_measured as f64
//         })
//         .collect()
// }

fn main() {
    let result = run_transverse_quantum_monte_carlo(
        1.0,
        10000,
        1,
        vec![((0, 1), 1.0), ((1, 2), 1.0), ((2, 3), 1.0), ((3, 4), 1.0)],
        5,
        1.0,
    );
    println!("{:?}", result)
}

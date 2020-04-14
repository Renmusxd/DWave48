use monte_carlo;
use monte_carlo::graph::GraphState;
use monte_carlo::sse::qmc_graph::new_qmc;

fn main() {
    let side_len = 10;
    let nvars = side_len * side_len;
    let beta = 1.0;
    let transverse = 1.0;

    let timesteps = 1000;

    let indices: Vec<(usize, usize)> = (0usize..side_len)
        .map(|i| {
            (0usize..side_len)
                .map(|j| (i, j))
                .collect::<Vec<(usize, usize)>>()
        })
        .flatten()
        .collect();
    let f = |i, j| j * side_len + i;

    let right_connects = indices
        .iter()
        .cloned()
        .map(|(i, j)| (f(i, j), f((i + 1) % side_len, j)));
    let down_connects = indices
        .iter()
        .cloned()
        .map(|(i, j)| (f(i, j), f(i, (j + 1) % side_len)));
    let edges = right_connects
        .chain(down_connects)
        .map(|(i, j)| ((i, j), 1.0))
        .collect::<Vec<_>>();
    let biases = vec![0.0; nvars];

    let gs = GraphState::new(&edges, &biases);
    let cutoff = nvars;
    let mut qmc_graph = new_qmc(gs, transverse, cutoff, false, false);

    let plot = qmc_graph.timesteps_sample(timesteps, beta, None);
}
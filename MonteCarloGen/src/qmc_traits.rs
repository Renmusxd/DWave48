
trait DiagonalUpdater {
    fn make_diagonal_update<H, E>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &[bool],
        hamiltonian: H,
        num_edges: usize,
        edges_fn: E,
    ) where
        H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64,
        E: Fn(usize) -> (usize, usize);
}

trait LoopUpdater {
    fn make_loop_update<H>(
        &mut self,
        initial_n: Option<usize>,
        hamiltonian: H,
    ) -> Vec<(usize, bool)>
        where
            H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64
}
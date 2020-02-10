use crate::qmc_types::*;
use rand::Rng;
use std::cmp::min;

pub trait OpNode {
    fn get_op(&self) -> Op;
    fn get_op_ref(&self) -> &Op;
    fn get_op_mut(&mut self) -> &mut Op;
}

pub trait OpContainer {
    fn get_n(&self) -> usize;
    fn get_nvars(&self) -> usize;
    fn get_pth(&self, p: usize) -> Option<&Op>;
}

pub trait DiagonalUpdater: OpContainer {
    fn set_pth(&mut self, p: usize, op: Option<Op>) -> Option<Op>;

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
        E: Fn(usize) -> (usize, usize),
    {
        self.make_diagonal_update_with_rng(
            cutoff,
            beta,
            state,
            hamiltonian,
            (num_edges, edges_fn),
            &mut rand::thread_rng(),
        )
    }

    fn make_diagonal_update_with_rng<H, E, R: Rng>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &[bool],
        hamiltonian: H,
        edges: (usize, E),
        rng: &mut R,
    ) where
        H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64,
        E: Fn(usize) -> (usize, usize),
    {
        let mut state = state.to_vec();
        let (num_edges, edges_fn) = edges;
        // Start by editing the ops list
        for p in 0..cutoff {
            let op = self.get_pth(p);

            let b = match op {
                None => rng.gen_range(0, num_edges),
                Some(op) if op.is_diagonal() => op.bond,
                Some(Op {
                    vara,
                    varb,
                    outputs,
                    ..
                }) => {
                    state[*vara] = outputs.0;
                    state[*varb] = outputs.1;
                    continue;
                }
            };
            let (vara, varb) = edges_fn(b);

            let substate = (state[vara], state[varb]);
            let mat_element = hamiltonian(vara, varb, b, substate, substate);
            let numerator = beta * num_edges as f64 * mat_element;
            let denominator = (cutoff - self.get_n()) as f64;

            match op {
                None => {
                    if numerator > denominator || rng.gen::<f64>() < (numerator / denominator) {
                        let op = Op::diagonal(vara, varb, b, (state[vara], state[varb]));
                        self.set_pth(p, Some(op));
                    }
                }
                Some(op) if op.is_diagonal() => {
                    if denominator + 1.0 > numerator
                        || rng.gen::<f64>() < ((denominator + 1.0) / numerator)
                    {
                        self.set_pth(p, None);
                    }
                }
                _ => (),
            };
        }
    }
}

pub trait LoopUpdater<Node: OpNode>: OpContainer {
    fn get_node_ref(&self, p: usize) -> Option<&Node>;
    fn get_node_mut(&mut self, p: usize) -> Option<&mut Node>;

    fn get_first_p(&self) -> Option<usize>;
    fn get_last_p(&self) -> Option<usize>;
    fn get_first_p_for_var(&self, var: usize) -> Option<usize>;
    fn get_last_p_for_var(&self, var: usize) -> Option<usize>;

    fn get_previous_p(&self, node: &Node) -> Option<usize>;
    fn get_next_p(&self, node: &Node) -> Option<usize>;

    fn get_previous_p_for_var(&self, var: usize, node: &Node) -> Option<usize>;
    fn get_next_p_for_var(&self, var: usize, node: &Node) -> Option<usize>;

    fn get_nth_p(&self, n: usize) -> usize {
        let acc = self
            .get_first_p()
            .map(|p| (p, self.get_node_ref(p).unwrap()))
            .unwrap();
        (0..n)
            .fold(acc, |(_, opnode), _| {
                let p = self.get_next_p(opnode).unwrap();
                (p, self.get_node_ref(p).unwrap())
            })
            .0
    }

    fn does_var_have_ops(&self, var: usize) -> bool {
        self.get_first_p_for_var(var).is_some()
    }

    fn make_loop_update<H>(
        &mut self,
        initial_n: Option<usize>,
        hamiltonian: H,
    ) -> Vec<(usize, bool)>
    where
        H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64,
    {
        self.make_loop_update_with_rng(initial_n, hamiltonian, &mut rand::thread_rng())
    }

    fn make_loop_update_with_rng<H, R: Rng>(
        &mut self,
        initial_n: Option<usize>,
        hamiltonian: H,
        rng: &mut R,
    ) -> Vec<(usize, bool)>
    where
        H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64,
    {
        let h = |op: &Op, entrance: Leg, exit: Leg| -> f64 {
            let (inputs, outputs) = adjust_states(op.inputs, op.outputs, entrance);
            let (inputs, outputs) = adjust_states(inputs, outputs, exit);
            // Call the supplied hamiltonian.
            hamiltonian(op.vara, op.varb, op.bond, inputs, outputs)
        };

        if self.get_n() > 0 {
            let initial_n = initial_n
                .map(|n| min(n, self.get_n()))
                .unwrap_or_else(|| rng.gen_range(0, self.get_n()));
            let nth_p = self.get_nth_p(initial_n);
            // Get the starting leg (vara/b, top/bottom).
            let initial_var = if rng.gen() { Variable::A } else { Variable::B };
            let initial_direction = if rng.gen() {
                OpSide::Inputs
            } else {
                OpSide::Outputs
            };
            let initial_leg = (initial_var, initial_direction);

            let updates = self.recursive_looper(
                (nth_p, initial_leg),
                nth_p,
                initial_leg,
                h,
                rng,
                vec![None; self.get_nvars()],
            );
            updates
                .into_iter()
                .enumerate()
                .fold(vec![], |mut acc, (i, v)| {
                    if let Some(v) = v {
                        acc.push((i, v))
                    };
                    acc
                })
        } else {
            vec![]
        }
    }

    fn recursive_looper<H, R: Rng>(
        &mut self,
        initial_op_and_leg: (usize, Leg),
        sel_op_pos: usize,
        entrance_leg: Leg,
        h: H,
        rng: &mut R,
        mut acc: Vec<Option<bool>>,
    ) -> Vec<Option<bool>>
    where
        H: Fn(&Op, Leg, Leg) -> f64,
    {
        let sel_opnode = self.get_node_mut(sel_op_pos).unwrap();
        let sel_op = sel_opnode.get_op();
        let weights = [
            h(&sel_op, entrance_leg, LEGS[0]),
            h(&sel_op, entrance_leg, LEGS[1]),
            h(&sel_op, entrance_leg, LEGS[2]),
            h(&sel_op, entrance_leg, LEGS[3]),
        ];
        let total_weight: f64 = weights.iter().sum();
        let choice = rng.gen_range(0.0, total_weight);
        let exit_leg = *weights
            .iter()
            .zip(LEGS.iter())
            .try_fold(choice, |c, (weight, leg)| {
                if c < *weight {
                    Err(leg)
                } else {
                    Ok(c - *weight)
                }
            })
            .unwrap_err();
        let (inputs, outputs) = adjust_states(
            sel_opnode.get_op_ref().inputs,
            sel_opnode.get_op_ref().outputs,
            entrance_leg,
        );
        let (inputs, outputs) = adjust_states(inputs, outputs, exit_leg);

        // Change the op now that we passed through.
        let sel_op_mut = sel_opnode.get_op_mut();
        sel_op_mut.inputs = inputs;
        sel_op_mut.outputs = outputs;

        // No longer need mutability.
        let sel_opnode = self.get_node_ref(sel_op_pos).unwrap();
        let sel_op = sel_opnode.get_op_ref();

        // Check if we closed the loop before going to next opnode.
        if (sel_op_pos, exit_leg) == initial_op_and_leg {
            acc
        } else {
            // Get the next opnode and entrance leg, let us know if it changes the initial/final.
            let (next_op_pos, var_to_match) = match exit_leg {
                (Variable::A, OpSide::Outputs) => {
                    let next_vara_op = self.get_next_p_for_var(sel_op.vara, sel_opnode);
                    let next = next_vara_op.unwrap_or_else(|| {
                        acc[sel_op.vara] = Some(sel_op.outputs.0);
                        self.get_first_p_for_var(sel_op.vara).unwrap()
                    });
                    (next, sel_op.vara)
                }
                (Variable::A, OpSide::Inputs) => {
                    let prev_vara_op = self.get_previous_p_for_var(sel_op.vara, sel_opnode);
                    let next = prev_vara_op.unwrap_or_else(|| {
                        acc[sel_op.vara] = Some(sel_op.inputs.0);
                        self.get_last_p_for_var(sel_op.vara).unwrap()
                    });
                    (next, sel_op.vara)
                }
                (Variable::B, OpSide::Outputs) => {
                    let next_varb_op = self.get_next_p_for_var(sel_op.varb, sel_opnode);
                    let next = next_varb_op.unwrap_or_else(|| {
                        acc[sel_op.varb] = Some(sel_op.outputs.1);
                        self.get_first_p_for_var(sel_op.varb).unwrap()
                    });
                    (next, sel_op.varb)
                }
                (Variable::B, OpSide::Inputs) => {
                    let prev_varb_op = self.get_previous_p_for_var(sel_op.varb, sel_opnode);
                    let next = prev_varb_op.unwrap_or_else(|| {
                        acc[sel_op.varb] = Some(sel_op.inputs.1);
                        self.get_last_p_for_var(sel_op.varb).unwrap()
                    });
                    (next, sel_op.varb)
                }
            };

            let next_node = self.get_node_ref(next_op_pos);
            let new_entrance_leg = match next_node.map(|opnode| opnode.get_op_ref()) {
                Some(Op { vara, .. }) if *vara == var_to_match => {
                    (Variable::A, exit_leg.1.reverse())
                }
                Some(Op { varb, .. }) if *varb == var_to_match => {
                    (Variable::B, exit_leg.1.reverse())
                }
                _ => unreachable!(),
            };

            // If back where we started, close loop and return state changes.
            if (next_op_pos, new_entrance_leg) == initial_op_and_leg {
                acc
            } else {
                self.recursive_looper(
                    initial_op_and_leg,
                    next_op_pos,
                    new_entrance_leg,
                    h,
                    rng,
                    acc,
                )
            }
        }
    }
}

//fn debug_print_looper<L: LoopUpdater<Node>, Node: OpNode, H>(looper: L, h: H)
//    where
//        H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64,
//{
//    let mut last_p = 0;
//    let nvars = looper.get_nvars();
//    for i in 0..nvars {
//        print!("=");
//    }
//    println!();
//    let p_ends = looper.get_first_p().and_then(|first_p| looper.get_last_p().map(|last_p| (first_p, last_p)));
//    if let Some((p_start, p_end)) = p_ends {
//        let mut next_p = Some(p_start);
//        while next_p.is_some() {
//            let np = next_p.unwrap();
//            for p in last_p + 1..np {
//                for i in 0..nvars {
//                    print!("|");
//                }
//                println!("\tp={}", p);
//            }
//            let opnode = looper.get_node_ref(np).unwrap();
//            let op = opnode.get_op_ref();
//            for v in 0..op.vara {
//                print!("|");
//            }
//            print!("{}", if op.inputs.0 { 1 } else { 0 });
//            for v in op.vara + 1..op.varb {
//                print!("|");
//            }
//            print!("{}", if op.inputs.1 { 1 } else { 0 });
//            for v in op.varb + 1..nvars {
//                print!("|");
//            }
//            println!("\tp={}\t\tW: {:?}", np, looper.p_matrix_weight(np, &h));
//
//            for v in 0..op.vara {
//                print!("|");
//            }
//            print!("{}", if op.outputs.0 { 1 } else { 0 });
//            for v in op.vara + 1..op.varb {
//                print!("|");
//            }
//            print!("{}", if op.outputs.1 { 1 } else { 0 });
//            for v in op.varb + 1..nvars {
//                print!("|");
//            }
//            println!("\top: {:?}", &op);
//            last_p = np;
//            next_p = looper.get_next_p(opnode);
//        }
//    }
//}

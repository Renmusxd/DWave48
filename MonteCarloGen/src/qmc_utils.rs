extern crate num;
use rand::Rng;
use std::cmp::{max, min};
use std::fmt::{Debug, Error, Formatter};
use std::iter::FromIterator;
use std::process::exit;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Op {
    vara: usize,
    varb: usize,
    bond: usize,
    inputs: (bool, bool),
    outputs: (bool, bool),
}

impl Op {
    fn diagonal(vara: usize, varb: usize, bond: usize, state: (bool, bool)) -> Self {
        Self {
            vara,
            varb,
            bond,
            inputs: state,
            outputs: state,
        }
    }
    fn offdiagonal(
        vara: usize,
        varb: usize,
        bond: usize,
        inputs: (bool, bool),
        outputs: (bool, bool),
    ) -> Self {
        Self {
            vara,
            varb,
            bond,
            inputs,
            outputs,
        }
    }
    fn is_diagonal(&self) -> bool {
        self.inputs == self.outputs
    }
}

#[derive(Debug)]
pub(crate) struct OpNode {
    op: Op,
    prev_vara_op: Option<usize>,
    next_vara_op: Option<usize>,
    prev_varb_op: Option<usize>,
    next_varb_op: Option<usize>,
    prev_p_op: Option<usize>,
    next_p_op: Option<usize>,
}

impl OpNode {
    pub(crate) fn point_prev_past_vara(
        &mut self,
        opnode: &OpNode,
    ) -> Option<usize> {
        let (other_vara, other_varb) = (opnode.op.vara, opnode.op.varb);
        let (vara, varb) = (self.op.vara, self.op.varb);

        if vara == other_vara {
            self.prev_vara_op = opnode.prev_vara_op;
            self.prev_vara_op
        } else if varb == other_vara {
            self.prev_varb_op = opnode.prev_vara_op;
            self.prev_varb_op
        } else {
            None
        }
    }
    pub(crate) fn point_prev_past_varb(
        &mut self,
        opnode: &OpNode,
    ) -> Option<usize> {
        let (other_vara, other_varb) = (opnode.op.vara, opnode.op.varb);
        let (vara, varb) = (self.op.vara, self.op.varb);

        if vara == other_varb {
            self.prev_vara_op = opnode.prev_varb_op;
            self.prev_vara_op
        } else if varb == other_varb {
            self.prev_varb_op = opnode.prev_varb_op;
            self.prev_varb_op
        } else {
            None
        }
    }

    pub(crate) fn point_next_past_vara(
        &mut self,
        opnode: &OpNode,
    ) -> Option<usize> {
        let (other_vara, other_varb) = (opnode.op.vara, opnode.op.varb);
        let (vara, varb) = (self.op.vara, self.op.varb);

        if vara == other_vara {
            self.next_vara_op = opnode.next_vara_op;
            self.next_vara_op
        } else if varb == other_vara {
            self.next_varb_op = opnode.next_vara_op;
            self.next_varb_op
        } else {
            None
        }
    }

    pub(crate) fn point_next_past_varb(
        &mut self,
        opnode: &OpNode,
    ) -> Option<usize> {
        let (other_vara, other_varb) = (opnode.op.vara, opnode.op.varb);
        let (vara, varb) = (self.op.vara, self.op.varb);

        if vara == other_varb {
            self.next_vara_op = opnode.next_varb_op;
            self.next_vara_op
        } else if varb == other_varb {
            self.next_varb_op = opnode.next_varb_op;
            self.next_varb_op
        } else {
            None
        }
    }
}

pub(crate) struct FastOps {
    ops: Vec<Option<OpNode>>,
    p_ends: Option<(usize, usize)>,
    var_ends: Vec<Option<(usize, usize)>>,
    n: usize,
}

impl Debug for FastOps {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.write_str(format!("p_ends:\t{:?}\tn: {}\n", self.p_ends, self.n).as_ref())?;
        f.write_str(format!("var_ends:\t{:?}\n", self.var_ends).as_ref())?;
        self.ops
            .iter()
            .enumerate()
            .try_for_each(|(p, opnode)| f.write_str(format!("{}:\t{:?}\n", p, opnode).as_ref()))
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Variable {
    A,
    B,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum OpSide {
    Inputs,
    Outputs,
}

impl OpSide {
    fn reverse(self) -> Self {
        match self {
            OpSide::Inputs => OpSide::Outputs,
            OpSide::Outputs => OpSide::Inputs,
        }
    }
}

type Leg = (Variable, OpSide);

static LEGS: [Leg; 4] = [
    (Variable::A, OpSide::Inputs),
    (Variable::A, OpSide::Outputs),
    (Variable::B, OpSide::Inputs),
    (Variable::B, OpSide::Outputs),
];

impl FastOps {
    pub(crate) fn new(nvars: usize) -> Self {
        Self {
            ops: Vec::default(),
            p_ends: None,
            var_ends: vec![None; nvars],
            n: 0,
        }
    }

    pub(crate) fn set_min_size(&mut self, n: usize) {
        if self.ops.len() < n {
            self.ops.resize_with(n, || None)
        }
    }

    pub fn does_var_have_ops(&self, var: usize) -> bool {
        self.var_ends[var].is_some()
    }

    pub(crate) fn get_size(&self) -> usize {
        self.n
    }

    pub(crate) fn get_pth(&self, p: usize) -> Option<Op> {
        if p >= self.ops.len() {
            None
        } else {
            self.ops[p].as_ref().map(|opnode| opnode.op)
        }
    }

    pub(crate) fn set_pth(&mut self, p: usize, op: Option<Op>) -> Option<Op> {
        self.set_min_size(p + 1);

        // Check if we can do this efficiently
        let opvars = op.map(|op| (op.vara, op.varb));
        let other_opvars = self.ops[p]
            .as_ref()
            .map(|opnode| (opnode.op.vara, opnode.op.varb));
        match (opvars, other_opvars) {
            (Some((vara, varb)), Some((varc, vard))) if vara == varc && varb == vard => {
                return self.swap_pth(p, op);
            }
            _ => (),
        };

        let old_opinfo = match self.ops[p].take() {
            Some(opnode) => {
                let retval = Some((opnode.op, opnode.prev_p_op, opnode.next_p_op));
                self.untie_op_node(opnode);
                retval
            }
            None => None,
        };

        // If we are just inserting an Identity (None), then we are done.
        // Otherwise we need to insert some stuff.
        if let Some(op) = op {
            // Makes an opnode for the op, ties it into the graph.
            self.tie_op(p, op, old_opinfo.map(|(_, p, n)| (p, n)))
        };

        if old_opinfo.is_some() {
            self.n -= 1;
        }
        if op.is_some() {
            self.n += 1;
        }
        old_opinfo.map(|(op, _, _)| op)
    }

    pub(crate) fn swap_pth(&mut self, p: usize, op: Option<Op>) -> Option<Op> {
        match (self.ops[p].as_mut(), op) {
            (None, None) => None,
            (Some(mut opnode), Some(op)) => {
                let oldop = opnode.op;
                opnode.op = op;
                Some(oldop)
            }
            _ => unreachable!(),
        }
    }

    fn tie_op(&mut self, p: usize, op: Op, p_hints: Option<(Option<usize>, Option<usize>)>) {
        let (vara, varb) = (op.vara, op.varb);

        // Get the previous and next ops
        let (prev_p_op, next_p_op) = p_hints.unwrap_or_else(|| {
            find_wrapping_indices_from_back(self.p_ends, p, |p| {
                self.ops[p].as_ref().unwrap().prev_p_op.unwrap()
            })
        });
        let (prev_vara_op, next_vara_op) =
            find_wrapping_indices_from_back(self.var_ends[vara], p, |p| {
                let o = self.ops[p].as_ref().unwrap();
                if vara == o.op.vara {
                    o.prev_vara_op.unwrap()
                } else if vara == o.op.varb {
                    o.prev_varb_op.unwrap()
                } else {
                    unreachable!()
                }
            });
        let (prev_varb_op, next_varb_op) =
            find_wrapping_indices_from_back(self.var_ends[varb], p, |p| {
                let o = self.ops[p].as_ref().unwrap();
                if varb == o.op.vara {
                    o.prev_vara_op.unwrap()
                } else if varb == o.op.varb {
                    o.prev_varb_op.unwrap()
                } else {
                    unreachable!()
                }
            });

        let opnode = OpNode {
            op,
            prev_vara_op,
            next_vara_op,
            prev_varb_op,
            next_varb_op,
            prev_p_op,
            next_p_op,
        };

        self.ops[p] = Some(opnode);

        // Update linked list heads and tails.
        self.p_ends = Some(match self.p_ends {
            None => (p, p),
            Some((p_start, p_end)) => (min(p, p_start), max(p, p_end)),
        });
        self.var_ends[vara] = Some(match self.var_ends[vara] {
            None => (p, p),
            Some((p_start, p_end)) => (min(p, p_start), max(p, p_end)),
        });
        self.var_ends[varb] = Some(match self.var_ends[varb] {
            None => (p, p),
            Some((p_start, p_end)) => (min(p, p_start), max(p, p_end)),
        });

        // Update linked list internals.
        // Make previous op in p order point to this one
        let prev_p_opnode = prev_p_op.map(|prev_p| self.ops[prev_p].as_mut().unwrap());
        if let Some(prev_p_opnode) = prev_p_opnode {
            prev_p_opnode.next_p_op = Some(p);
        }
        // Make next op in p order point to this one
        let next_p_opnode = next_p_op.map(|next_p| self.ops[next_p].as_mut().unwrap());
        if let Some(next_p_opnode) = next_p_opnode {
            next_p_opnode.prev_p_op = Some(p);
        }

        // Make previous op with vara point to this one
        let prev_vara_opnode = prev_vara_op.map(|prev_vara| self.ops[prev_vara].as_mut().unwrap());
        if let Some(prev_vara_opnode) = prev_vara_opnode {
            let (prev_vara, prev_varb) = (prev_vara_opnode.op.vara, prev_vara_opnode.op.varb);
            if prev_vara == vara {
                prev_vara_opnode.next_vara_op = Some(p);
            } else if prev_varb == vara {
                prev_vara_opnode.next_varb_op = Some(p);
            } else {
                unreachable!()
            }
        }

        // Make next op with vara point to this one
        let next_vara_opnode = next_vara_op.map(|next_vara| self.ops[next_vara].as_mut().unwrap());
        if let Some(next_vara_opnode) = next_vara_opnode {
            let (next_vara, next_varb) = (next_vara_opnode.op.vara, next_vara_opnode.op.varb);
            if next_vara == vara {
                next_vara_opnode.prev_vara_op = Some(p);
            } else if next_varb == vara {
                next_vara_opnode.prev_varb_op = Some(p);
            } else {
                unreachable!()
            }
        }

        // Make previous op with varb point to this one
        let prev_varb_opnode = prev_varb_op.map(|prev_varb| self.ops[prev_varb].as_mut().unwrap());
        if let Some(prev_varb_opnode) = prev_varb_opnode {
            let (prev_vara, prev_varb) = (prev_varb_opnode.op.vara, prev_varb_opnode.op.varb);
            if prev_vara == varb {
                prev_varb_opnode.next_vara_op = Some(p);
            } else if prev_varb == varb {
                prev_varb_opnode.next_varb_op = Some(p);
            } else {
                unreachable!()
            }
        }

        // Make next op with varb point to this one
        let next_varb_opnode = next_varb_op.map(|next_varb| self.ops[next_varb].as_mut().unwrap());
        if let Some(next_varb_opnode) = next_varb_opnode {
            let (next_vara, next_varb) = (next_varb_opnode.op.vara, next_varb_opnode.op.varb);
            if next_vara == varb {
                next_varb_opnode.prev_vara_op = Some(p);
            } else if next_varb == varb {
                next_varb_opnode.prev_varb_op = Some(p);
            } else {
                unreachable!()
            }
        }
    }

    fn untie_op_node(&mut self, opnode: OpNode) {
        // First untie the p ordering
        if let Some(prev_opnode_pos) = opnode.prev_p_op {
            // If there's a previous, set it to point to the next.
            self.ops[prev_opnode_pos].as_mut().unwrap().next_p_op = opnode.next_p_op;
        } else if let Some(next_opnode_pos) = opnode.next_p_op {
            // If there's no previous, set the new first p entry to opnode's next
            self.p_ends.as_mut().unwrap().0 = next_opnode_pos;
        }
        // the final else case is handled below.

        if let Some(next_opnode_pos) = opnode.next_p_op {
            // If there's a next, point it to the previous.
            self.ops[next_opnode_pos].as_mut().unwrap().prev_p_op = opnode.prev_p_op;
        } else if let Some(prev_opnode_pos) = opnode.prev_p_op {
            // If there's no next, set the new last p entry to opnode's previous
            self.p_ends.as_mut().unwrap().1 = prev_opnode_pos;
        } else {
            // If there's also no previous then just set it to none.
            self.p_ends = None;
        }

        let (vara, varb) = (opnode.op.vara, opnode.op.varb);

        // Then untie the variable ordering
        // VARA
        if let Some(prev_opnode_pos) = opnode.prev_vara_op {
            let next_a = self.ops[prev_opnode_pos]
                .as_mut()
                .unwrap()
                .point_next_past_vara(&opnode);
            // Nones mean end of sequence, should overwrite variable stuff.
            let opref = &self.ops[prev_opnode_pos].as_ref().unwrap().op;
            let (sel_a, sel_b) = (opref.vara, opref.varb);
            if next_a.is_none() {
                self.var_ends[vara].as_mut().unwrap().1 = prev_opnode_pos;
            }
        } else if let Some(next_opnode_pos) = opnode.next_vara_op {
            // If no previous, but there's a next, then we should update the variable ends.
            self.var_ends[vara].as_mut().unwrap().0 = next_opnode_pos;
        }
        // Last else case handled below
        if let Some(next_opnode_pos) = opnode.next_vara_op {
            let prev_a = self.ops[next_opnode_pos]
                .as_mut()
                .unwrap()
                .point_prev_past_vara(&opnode);
            // Nones mean beginning of sequence, overwrite variable stuff
            let opref = &self.ops[next_opnode_pos].as_ref().unwrap().op;
            let (sel_a, sel_b) = (opref.vara, opref.varb);
            if prev_a.is_none() {
                self.var_ends[vara].as_mut().unwrap().0 = next_opnode_pos;
            }
        } else if let Some(prev_opnode_pos) = opnode.prev_vara_op {
            // If no next, but there's a previous, then we should update the variable ends.
            self.var_ends[vara].as_mut().unwrap().1 = prev_opnode_pos;
        } else {
            self.var_ends[vara] = None;
        }

        // VARB
        if let Some(prev_opnode_pos) = opnode.prev_varb_op {
            let next_b = self.ops[prev_opnode_pos]
                .as_mut()
                .unwrap()
                .point_next_past_varb(&opnode);
            let opref = &self.ops[prev_opnode_pos].as_ref().unwrap().op;
            let (sel_a, sel_b) = (opref.vara, opref.varb);
            // Nones mean end of sequence, should overwrite variable stuff.
            if next_b.is_none() {
                self.var_ends[varb].as_mut().unwrap().1 = prev_opnode_pos;
            }
        } else if let Some(next_opnode_pos) = opnode.next_varb_op {
            // If no previous, but there's a next, then we should update the variable ends.
            self.var_ends[varb].as_mut().unwrap().0 = next_opnode_pos;
        }
        // Last else case handled below
        if let Some(next_opnode_pos) = opnode.next_varb_op {
            let prev_b = self.ops[next_opnode_pos]
                .as_mut()
                .unwrap()
                .point_prev_past_varb(&opnode);
            // Nones mean beginning of sequence, overwrite variable stuff
            let opref = &self.ops[next_opnode_pos].as_ref().unwrap().op;
            let (sel_a, sel_b) = (opref.vara, opref.varb);
            if prev_b.is_none() {
                self.var_ends[varb].as_mut().unwrap().0 = next_opnode_pos;
            }
        } else if let Some(prev_opnode_pos) = opnode.prev_varb_op {
            // If no next, but there's a previous, then we should update the variable ends.
            self.var_ends[varb].as_mut().unwrap().1 = prev_opnode_pos;
        } else {
            self.var_ends[varb] = None;
        }
    }

    fn get_p_states(&self, initial_state: &[bool]) -> Vec<(usize, Vec<bool>)> {
        match self.p_ends {
            None => vec![],
            Some((p_start, _)) => {
                let state = initial_state.to_vec();
                let (states, _) = (0..self.n).fold(
                    (vec![], (Some(p_start), state)),
                    |(mut states, (ppos, mut state)), _| {
                        if let Some(ppos) = ppos {
                            let opnode = self.ops[ppos].as_ref().unwrap();
                            state[opnode.op.vara] = opnode.op.outputs.0;
                            state[opnode.op.varb] = opnode.op.outputs.1;
                            states.push((ppos, state.clone()));
                            let ppos = opnode.next_p_op;
                            (states, (ppos, state))
                        } else {
                            // This shouldn't happen.
                            unreachable!();
                        }
                    },
                );
                states
            }
        }
    }

    fn get_ops_for_var(&self, var: usize) -> Vec<Op> {
        fn rec_traverse(
            var: usize,
            ops: &[Option<OpNode>],
            mut acc: Vec<Op>,
            opnode: Option<&OpNode>,
        ) -> Vec<Op> {
            match opnode {
                None => acc,
                Some(opnode) => {
                    acc.push(opnode.op);
                    match (opnode.op.vara, opnode.op.varb) {
                        (vara, varb) if vara == var => rec_traverse(
                            var,
                            ops,
                            acc,
                            opnode.next_vara_op.map(|indx| ops[indx].as_ref().unwrap()),
                        ),
                        (vara, varb) if varb == var => rec_traverse(
                            var,
                            ops,
                            acc,
                            opnode.next_varb_op.map(|indx| ops[indx].as_ref().unwrap()),
                        ),
                        _ => unreachable!(),
                    }
                }
            }
        }
        rec_traverse(
            var,
            &self.ops,
            vec![],
            self.var_ends[var].map(|(start, _)| self.ops[start].as_ref().unwrap()),
        )
    }

    pub fn make_diagonal_update_with_rng<H, E, R: Rng>(
        &mut self,
        cutoff: usize,
        beta: f64,
        state: &[bool],
        hamiltonian: H,
        num_edges: usize,
        edges_fn: E,
        rng: &mut R,
    ) where
        H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64,
        E: Fn(usize) -> (usize, usize),
    {
        let mut state = state.to_vec();
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
                    state[vara] = outputs.0;
                    state[varb] = outputs.1;
                    continue;
                }
            };
            let (vara, varb) = edges_fn(b);

            let substate = (state[vara], state[varb]);
            let mat_element = hamiltonian(vara, varb, b, substate, substate);
            let numerator = beta * num_edges as f64 * mat_element;
            let denominator = (cutoff - self.get_size()) as f64;

            match op {
                None => {
                    let prob = numerator / denominator;
                    if rng.gen::<f64>() < prob {
                        let op = Op::diagonal(vara, varb, b, (state[vara], state[varb]));
                        self.set_pth(p, Some(op));
                    }
                }
                Some(op) if op.is_diagonal() => {
                    let prob = (denominator + 1.0) / numerator;
                    if rng.gen::<f64>() < prob {
                        self.set_pth(p, None);
                    }
                }
                _ => (),
            };
        }
    }

    pub fn make_diagonal_update<H, E>(
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
            num_edges,
            edges_fn,
            &mut rand::thread_rng(),
        )
    }

    pub fn make_loop_update_with_rng<H, R: Rng>(
        &mut self,
        initial_n: Option<usize>,
        hamiltonian: H,
        rng: &mut R,
    ) -> Vec<(usize, bool)>
    where
        H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64,
    {
        let h = |op: Op, entrance: Leg, exit: Leg| -> f64 {
            let (inputs, outputs) = adjust_states(op.inputs, op.outputs, entrance);
            let (inputs, outputs) = adjust_states(inputs, outputs, exit);
            // Call the supplied hamiltonian.
            hamiltonian(op.vara, op.varb, op.bond, inputs, outputs)
        };

        let opnode = self
            .p_ends
            .map(|(p_start, _)| (p_start, self.ops[p_start].as_ref().unwrap()));
        if let Some((p_start, opnode)) = opnode {
            let initial_n = initial_n
                .map(|n| min(n, self.n))
                .unwrap_or_else(|| rng.gen_range(0, self.n));

            let initial_opnode = (0..initial_n).fold(opnode, |acc, _| {
                self.ops[opnode.next_p_op.unwrap()].as_ref().unwrap()
            });
            // Get the starting leg (vara/b, top/bottom).
            let initial_var = if rng.gen() { Variable::A } else { Variable::B };
            let initial_direction = if rng.gen() {
                OpSide::Inputs
            } else {
                OpSide::Outputs
            };
            let initial_leg = (initial_var, initial_direction);

            let updates = self.recursive_looper(
                (p_start, initial_leg),
                p_start,
                initial_leg,
                h,
                rng,
                vec![None; self.var_ends.len()],
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

    pub fn make_loop_update<H>(
        &mut self,
        initial_n: Option<usize>,
        hamiltonian: H,
    ) -> Vec<(usize, bool)>
    where
        H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64,
    {
        self.make_loop_update_with_rng(initial_n, hamiltonian, &mut rand::thread_rng())
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
        H: Fn(Op, Leg, Leg) -> f64,
    {
        let sel_opnode = self.ops[sel_op_pos].as_mut().unwrap();
        let sel_op = sel_opnode.op;
        let weights = [
            h(sel_op, entrance_leg, LEGS[0]),
            h(sel_op, entrance_leg, LEGS[1]),
            h(sel_op, entrance_leg, LEGS[2]),
            h(sel_op, entrance_leg, LEGS[3]),
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
        let (inputs, outputs) =
            adjust_states(sel_opnode.op.inputs, sel_opnode.op.outputs, entrance_leg);
        let (inputs, outputs) = adjust_states(inputs, outputs, exit_leg);
        // Change the op now that we passed through.
        sel_opnode.op.inputs = inputs;
        sel_opnode.op.outputs = outputs;
        let sel_op = &sel_opnode.op;
        let var_ends = &self.var_ends;

        // Check if we closed the loop before going to next opnode.
        if (sel_op_pos, exit_leg) == initial_op_and_leg {
            acc
        } else {
            // Get the next opnode and entrance leg, let us know if it changes the initial/final.
            let (next_op_pos, var_to_match) = match exit_leg {
                (Variable::A, OpSide::Outputs) => {
                    let next = sel_opnode.next_vara_op.unwrap_or_else(|| {
                        acc[sel_op.vara] = Some(sel_op.outputs.0);
                        var_ends[sel_op.vara].unwrap().0
                    });
                    (next, sel_op.vara)
                }
                (Variable::A, OpSide::Inputs) => {
                    let next = sel_opnode.prev_vara_op.unwrap_or_else(|| {
                        acc[sel_op.vara] = Some(sel_op.inputs.0);
                        var_ends[sel_op.vara].unwrap().1
                    });
                    (next, sel_op.vara)
                }
                (Variable::B, OpSide::Outputs) => {
                    let next = sel_opnode.next_varb_op.unwrap_or_else(|| {
                        acc[sel_op.varb] = Some(sel_op.outputs.1);
                        var_ends[sel_op.varb].unwrap().0
                    });
                    (next, sel_op.varb)
                }
                (Variable::B, OpSide::Inputs) => {
                    let next = sel_opnode.prev_varb_op.unwrap_or_else(|| {
                        acc[sel_op.varb] = Some(sel_op.inputs.1);
                        var_ends[sel_op.varb].unwrap().1
                    });
                    (next, sel_op.varb)
                }
            };

            let new_entrance_leg = match self.ops[next_op_pos].as_ref().map(|opnode| opnode.op) {
                Some(Op { vara, .. }) if vara == var_to_match => {
                    (Variable::A, exit_leg.1.reverse())
                }
                Some(Op { varb, .. }) if varb == var_to_match => {
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

    fn p_matrix_weight<H>(&self, p: usize, h: &H) -> f64
        where H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64
    {
        let op = self.ops[p].as_ref().unwrap();
        h(op.op.vara, op.op.varb, op.op.bond, op.op.inputs, op.op.outputs)
    }

    pub fn total_matrix_weight<H>(&self, h: H) -> f64
    where H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64
    {
        let mut t = 1.0;
        let mut p = self.p_ends.map(|(p, _)| p);
        while p.is_some() {
            let op = self.ops[p.unwrap()].as_ref().unwrap();
            t *= h(op.op.vara, op.op.varb, op.op.bond, op.op.inputs, op.op.outputs);
            p = op.next_p_op;
        }
        t
    }

    pub fn debug_print<H>(&self, h: H)
    where H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64 {
        let mut last_p = 0;
        let nvars = self.var_ends.len();
        for i in 0..nvars {
            print!("=");
        }
        println!();
        if let Some((p_start, p_end)) = self.p_ends {
            let mut next_p = Some(p_start);
            while next_p.is_some() {
                let np = next_p.unwrap();
                for p in last_p+1..np {
                    for i in 0..nvars {
                        print!("|");
                    }
                    println!("\tp={}", p);
                }
                let op = self.ops[np].as_ref().unwrap();
                for v in 0 .. op.op.vara {
                    print!("|");
                }
                print!("{}", if op.op.inputs.0 { 1 } else {0});
                for v in op.op.vara+1 .. op.op.varb {
                    print!("|");
                }
                print!("{}", if op.op.inputs.1 {1} else {0});
                for v in op.op.varb+1 .. nvars {
                    print!("|");
                }
                println!("\tp={}\t\tW: {:?}", np, self.p_matrix_weight(np, &h));

                for v in 0 .. op.op.vara {
                    print!("|");
                }
                print!("{}", if op.op.outputs.0 {1} else {0});
                for v in op.op.vara+1 .. op.op.varb {
                    print!("|");
                }
                print!("{}", if op.op.outputs.1 {1} else {0});
                for v in op.op.varb+1 .. nvars {
                    print!("|");
                }
                println!("\top: {:?}", &op);
                last_p = np;
                next_p = op.next_p_op;
            }
        }
    }

}

fn adjust_states(
    before: (bool, bool),
    after: (bool, bool),
    leg: Leg,
) -> ((bool, bool), (bool, bool)) {
    let (mut a_bef, mut b_bef) = before;
    let (mut a_aft, mut b_aft) = after;
    match leg {
        (Variable::A, OpSide::Inputs) => {
            a_bef = !a_bef;
        }
        (Variable::A, OpSide::Outputs) => {
            a_aft = !a_aft;
        }
        (Variable::B, OpSide::Inputs) => {
            b_bef = !b_bef;
        }
        (Variable::B, OpSide::Outputs) => {
            b_aft = !b_aft;
        }
    };
    ((a_bef, b_bef), (a_aft, b_aft))
}

fn find_wrapping_indices_from_back<F: Fn(usize) -> usize>(
    ends: Option<(usize, usize)>,
    p: usize,
    prev_fn: F,
) -> (Option<usize>, Option<usize>) {
    match ends {
        None => (None, None),
        Some((p_start, _)) if p < p_start => (None, Some(p_start)),
        Some((_, p_end)) if p_end < p => (Some(p_end), None),
        Some((p_start, p_end)) if p_start == p_end => {
            // If p_start == p_end, it's there's also equal to p
            // If equal to p it should have been removed during the untie phase.
            unreachable!()
        }
        Some((p_start, p_end)) => {
            // We know p_before can be unwrapped because p_start != p_end
            let (mut p_before, mut p_after) = (prev_fn(p_end), p_end);
            while p < p_before {
                p_after = p_before;
                p_before = prev_fn(p_after);
            }
            // Now p_before < p < p_after
            (Some(p_before), Some(p_after))
        }
    }
}

#[cfg(test)]
mod fastops_tests {
    use super::*;
    use rand_chacha::ChaCha20Rng;

    #[test]
    fn add_single_item() {
        let mut f = FastOps::new(2);
        let op = Op::diagonal(0, 1, 0, (true, true));
        f.set_pth(0, Some(op));
        println!("{:?}", f);
        assert_eq!(f.n, 1);
        assert_eq!(f.get_ops_for_var(0), vec![op]);
        assert_eq!(f.get_ops_for_var(1), vec![op]);
    }

    #[test]
    fn add_unrelated_item() {
        let mut f = FastOps::new(4);
        let opa = Op::diagonal(0, 1, 0, (true, true));
        let opb = Op::diagonal(2, 3, 1, (true, true));
        f.set_pth(0, Some(opa));
        f.set_pth(1, Some(opb));
        println!("{:?}", f);
        assert_eq!(f.n, 2);

        assert_eq!(f.get_ops_for_var(0), vec![opa]);
        assert_eq!(f.get_ops_for_var(1), vec![opa]);
        assert_eq!(f.get_ops_for_var(2), vec![opb]);
        assert_eq!(f.get_ops_for_var(3), vec![opb]);
    }

    #[test]
    fn add_identical_item() {
        let mut f = FastOps::new(2);
        let opa = Op::diagonal(0, 1, 0, (true, true));
        f.set_pth(0, Some(opa));
        f.set_pth(1, Some(opa));
        println!("{:?}", f);
        assert_eq!(f.n, 2);

        assert_eq!(f.get_ops_for_var(0), vec![opa, opa]);
        assert_eq!(f.get_ops_for_var(1), vec![opa, opa]);
    }

    #[test]
    fn add_overlapping_item() {
        let mut f = FastOps::new(3);
        let opa = Op::diagonal(0, 1, 0, (true, true));
        let opb = Op::diagonal(1, 2, 0, (true, true));
        f.set_pth(0, Some(opa));
        f.set_pth(1, Some(opb));
        println!("{:?}", f);
        assert_eq!(f.n, 2);

        assert_eq!(f.get_ops_for_var(0), vec![opa]);
        assert_eq!(f.get_ops_for_var(1), vec![opa, opb]);
        assert_eq!(f.get_ops_for_var(2), vec![opb]);
    }

    #[test]
    fn add_skipping_item() {
        let mut f = FastOps::new(3);
        let opa = Op::diagonal(0, 1, 0, (true, true));
        let opb = Op::diagonal(1, 2, 0, (true, true));
        f.set_pth(0, Some(opa));
        f.set_pth(2, Some(opb));
        println!("{:?}", f);
        assert_eq!(f.n, 2);

        assert_eq!(f.get_ops_for_var(0), vec![opa]);
        assert_eq!(f.get_ops_for_var(1), vec![opa, opb]);
        assert_eq!(f.get_ops_for_var(2), vec![opb]);
    }

    #[test]
    fn add_skipping_item_and_remove() {
        let mut f = FastOps::new(3);
        let opa = Op::diagonal(0, 1, 0, (true, true));
        let opb = Op::diagonal(1, 2, 0, (true, true));
        f.set_pth(0, Some(opa));
        f.set_pth(2, Some(opb));
        f.set_pth(2, None);
        println!("{:?}", f);
        assert_eq!(f.n, 1);

        assert_eq!(f.get_ops_for_var(0), vec![opa]);
        assert_eq!(f.get_ops_for_var(1), vec![opa]);
        assert_eq!(f.get_ops_for_var(2), vec![]);
    }

    #[test]
    fn add_skipping_item_and_remove_first() {
        let mut f = FastOps::new(3);
        let opa = Op::diagonal(0, 1, 0, (true, true));
        let opb = Op::diagonal(1, 2, 0, (true, true));
        f.set_pth(0, Some(opa));
        f.set_pth(2, Some(opb));
        f.set_pth(0, None);
        println!("{:?}", f);
        assert_eq!(f.n, 1);

        assert_eq!(f.get_ops_for_var(0), vec![]);
        assert_eq!(f.get_ops_for_var(1), vec![opb]);
        assert_eq!(f.get_ops_for_var(2), vec![opb]);
    }

    #[test]
    fn add_and_remove_all() {
        let mut f = FastOps::new(3);
        f.set_pth(0, Some(Op::diagonal(0, 1, 0, (true, true))));
        f.set_pth(2, Some(Op::diagonal(1, 2, 0, (true, true))));
        f.set_pth(0, None);
        f.set_pth(2, None);
        println!("{:?}", f);
        assert_eq!(f.n, 0);

        assert_eq!(f.get_ops_for_var(0), vec![]);
        assert_eq!(f.get_ops_for_var(1), vec![]);
    }

    #[test]
    fn get_states_simple() {
        let mut f = FastOps::new(3);
        f.set_pth(0, Some(Op::diagonal(0, 1, 0, (true, true))));
        f.set_pth(2, Some(Op::diagonal(1, 2, 0, (true, true))));
        println!("{:?}", f);
        let states = f.get_p_states(&[true, false, true]);
        println!("{:?}", states);
    }

    #[test]
    fn get_states_offsimple() {
        let mut f = FastOps::new(3);
        f.set_pth(
            0,
            Some(Op::offdiagonal(0, 1, 0, (false, true), (true, false))),
        );
        f.set_pth(
            1,
            Some(Op::offdiagonal(1, 2, 0, (false, true), (true, false))),
        );
        f.set_pth(
            2,
            Some(Op::offdiagonal(0, 2, 0, (false, true), (true, false))),
        );
        println!("{:?}", f);
        let states = f.get_p_states(&[false, true, true]);
        println!("{:?}", states);
    }

    #[test]
    fn test_loop_update() {
        let mut f = FastOps::new(2);
        f.set_pth(0, Some(Op::diagonal(0, 1, 0, (false, false))));
        println!("{:?}", f);
        let updates = f.make_loop_update(
            Some(0),
            |vara, varb, bond, input, output| {
                if input == output {
                    1.0
                } else {
                    0.0
                }
            },
        );
        println!("{:?}", f);
        println!("{:?}", updates);
    }

    #[test]
    fn test_larger_loop_update() {
        let mut f = FastOps::new(3);
        f.set_pth(0, Some(Op::diagonal(0, 1, 0, (false, false))));
        f.set_pth(1, Some(Op::diagonal(1, 2, 1, (false, false))));

        println!("{:?}", f);
        let mut rng: ChaCha20Rng = rand::SeedableRng::seed_from_u64(12345678);
        let updates =
            f.make_loop_update_with_rng(Some(0), |vara, varb, bond, input, output| 1.0, &mut rng);
        println!("{:?}", f);
        println!("{:?}", updates);
    }

    #[test]
    fn test_diagonal_update() {
        let edges = vec![(0, 1), (1, 2)];

        let h = |_, _, _, _, _| 1.0;

        let mut f = FastOps::new(3);
        let mut rng: ChaCha20Rng = rand::SeedableRng::seed_from_u64(12345678);
        f.make_diagonal_update_with_rng(10, 1.0, &[false, false, false], h, edges.len(), |i| edges[i], &mut rng);
    }

    #[test]
    fn test_diagonal_updates() {
        let edges = vec![(0, 1), (1, 2)];

        let h = |_, _, _, _, _| 1.0;

        let mut f = FastOps::new(3);
        let mut rng: ChaCha20Rng = rand::SeedableRng::seed_from_u64(12345678);
        f.make_diagonal_update_with_rng(10, 1.0, &[false, false, false], h, edges.len(), |i| edges[i], &mut rng);
        f.make_diagonal_update_with_rng(10, 1.0, &[false, false, false], h, edges.len(), |i| edges[i], &mut rng);
        f.make_diagonal_update_with_rng(10, 1.0, &[false, false, false], h, edges.len(), |i| edges[i], &mut rng);
        f.make_diagonal_update_with_rng(10, 1.0, &[false, false, false], h, edges.len(), |i| edges[i], &mut rng);
    }
}

extern crate num;
use rand::Rng;
use std::cmp::{max, min};
use std::fmt::{Debug, Error, Formatter};
use std::iter::FromIterator;
use test::stats::Stats;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    // vara, varb, edge_index, (vara_value, varb_value)
    Diagonal(usize, usize, usize, (bool, bool)),
    // vara, varb, edge_index, input_values, output_values
    OffDiagonal(usize, usize, usize, (bool, bool), (bool, bool)),
}

impl Op {
    pub(crate) fn get_vars(&self) -> (usize, usize) {
        match *self {
            Op::Diagonal(vara, varb, _, _) => (vara, varb),
            Op::OffDiagonal(vara, varb, _, _, _) => (vara, varb),
        }
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
    pub(crate) fn point_vars_prev_past(
        &mut self,
        opnode: &OpNode,
    ) -> (Option<usize>, Option<usize>) {
        let (other_vara, other_varb) = opnode.op.get_vars();
        let (vara, varb) = self.op.get_vars();

        let a = if vara == other_vara {
            self.prev_vara_op = opnode.prev_vara_op;
            self.prev_vara_op
        } else if vara == other_varb {
            self.prev_vara_op = opnode.prev_varb_op;
            self.prev_vara_op
        } else {
            None
        };
        let b = if varb == other_vara {
            self.prev_varb_op = opnode.prev_vara_op;
            self.prev_varb_op
        } else if varb == other_varb {
            self.prev_varb_op = opnode.prev_varb_op;
            self.prev_varb_op
        } else {
            None
        };
        (a, b)
    }

    pub(crate) fn point_vars_next_past(
        &mut self,
        opnode: &OpNode,
    ) -> (Option<usize>, Option<usize>) {
        let (other_vara, other_varb) = opnode.op.get_vars();
        let (vara, varb) = self.op.get_vars();

        let a = if vara == other_vara {
            self.next_vara_op = opnode.next_vara_op;
            self.next_vara_op
        } else if vara == other_varb {
            self.next_vara_op = opnode.next_varb_op;
            self.next_vara_op
        } else {
            None
        };
        let b = if varb == other_vara {
            self.next_varb_op = opnode.next_vara_op;
            self.next_varb_op
        } else if varb == other_varb {
            self.next_varb_op = opnode.next_varb_op;
            self.next_varb_op
        } else {
            None
        };
        (a, b)
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

// (variable, side)
type Leg = (bool, bool);
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

    pub(crate) fn get_size(&self) -> usize {
        self.n
    }

    pub(crate) fn get_pth(&self, p: usize) -> Option<Op> {
        self.ops[p].as_ref().map(|opnode| opnode.op)
    }

    pub(crate) fn set_pth(&mut self, p: usize, op: Option<Op>) -> Option<Op> {
        self.set_min_size(p + 1);

        // Check if we can do this efficiently
        let opvars = op.map(|op| op.get_vars());
        let other_opvars = self.ops[p].as_ref().map(|opnode| opnode.op.get_vars());
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
        let (vara, varb) = op.get_vars();

        // Get the previous and next ops
        let (prev_p_op, next_p_op) = p_hints.unwrap_or_else(|| {
            find_wrapping_indices_from_back(self.p_ends, p, |p| {
                self.ops[p].as_ref().unwrap().prev_p_op.unwrap()
            })
        });
        let (prev_vara_op, next_vara_op) =
            find_wrapping_indices_from_back(self.var_ends[vara], p, |p| {
                let o = self.ops[p].as_ref().unwrap();
                if vara == o.op.get_vars().0 {
                    o.prev_vara_op.unwrap()
                } else if vara == o.op.get_vars().1 {
                    o.prev_varb_op.unwrap()
                } else {
                    unreachable!()
                }
            });
        let (prev_varb_op, next_varb_op) =
            find_wrapping_indices_from_back(self.var_ends[varb], p, |p| {
                let o = self.ops[p].as_ref().unwrap();
                if varb == o.op.get_vars().0 {
                    o.prev_vara_op.unwrap()
                } else if varb == o.op.get_vars().1 {
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
            let (prev_vara, prev_varb) = prev_vara_opnode.op.get_vars();
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
            let (next_vara, next_varb) = next_vara_opnode.op.get_vars();
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
            let (prev_vara, prev_varb) = prev_varb_opnode.op.get_vars();
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
            let (next_vara, next_varb) = next_varb_opnode.op.get_vars();
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

        let (vara, varb) = opnode.op.get_vars();

        // Then untie the variable ordering
        // VARA
        if let Some(prev_opnode_pos) = opnode.prev_vara_op {
            let (next_a, next_b) = self.ops[prev_opnode_pos]
                .as_mut()
                .unwrap()
                .point_vars_next_past(&opnode);
            // Nones mean end of sequence, should overwrite variable stuff.
            let (sel_a, sel_b) = self.ops[prev_opnode_pos].as_ref().unwrap().op.get_vars();
            if next_a.is_none() {
                self.var_ends[sel_a].as_mut().unwrap().1 = prev_opnode_pos;
            }
            if next_b.is_none() {
                self.var_ends[sel_b].as_mut().unwrap().1 = prev_opnode_pos;
            }
        } else if let Some(next_opnode_pos) = opnode.next_vara_op {
            // If no previous, but there's a next, then we should update the variable ends.
            self.var_ends[vara].as_mut().unwrap().0 = next_opnode_pos;
        }
        // Last else case handled below
        if let Some(next_opnode_pos) = opnode.next_vara_op {
            let (prev_a, prev_b) = self.ops[next_opnode_pos]
                .as_mut()
                .unwrap()
                .point_vars_prev_past(&opnode);
            // Nones mean beginning of sequence, overwrite variable stuff
            let (sel_a, sel_b) = self.ops[next_opnode_pos].as_ref().unwrap().op.get_vars();
            if prev_a.is_none() {
                self.var_ends[sel_a].as_mut().unwrap().0 = next_opnode_pos;
            }
            if prev_b.is_none() {
                self.var_ends[sel_b].as_mut().unwrap().0 = next_opnode_pos;
            }
        } else if let Some(prev_opnode_pos) = opnode.prev_vara_op {
            // If no next, but there's a previous, then we should update the variable ends.
            self.var_ends[vara].as_mut().unwrap().1 = prev_opnode_pos;
        } else {
            self.var_ends[vara] = None;
        }

        // VARB
        if let Some(prev_opnode_pos) = opnode.prev_varb_op {
            let (next_a, next_b) = self.ops[prev_opnode_pos]
                .as_mut()
                .unwrap()
                .point_vars_next_past(&opnode);
            let (sel_a, sel_b) = self.ops[prev_opnode_pos].as_ref().unwrap().op.get_vars();
            // Nones mean end of sequence, should overwrite variable stuff.
            if next_a.is_none() {
                self.var_ends[sel_a].as_mut().unwrap().1 = prev_opnode_pos;
            }
            if next_b.is_none() {
                self.var_ends[sel_b].as_mut().unwrap().1 = prev_opnode_pos;
            }
        } else if let Some(next_opnode_pos) = opnode.next_varb_op {
            // If no previous, but there's a next, then we should update the variable ends.
            self.var_ends[varb].as_mut().unwrap().0 = next_opnode_pos;
        }
        // Last else case handled below
        if let Some(next_opnode_pos) = opnode.next_varb_op {
            let (prev_a, prev_b) = self.ops[next_opnode_pos]
                .as_mut()
                .unwrap()
                .point_vars_prev_past(&opnode);
            // Nones mean beginning of sequence, overwrite variable stuff
            let (sel_a, sel_b) = self.ops[next_opnode_pos].as_ref().unwrap().op.get_vars();
            if prev_a.is_none() {
                self.var_ends[sel_a].as_mut().unwrap().0 = next_opnode_pos;
            }
            if prev_b.is_none() {
                self.var_ends[sel_b].as_mut().unwrap().0 = next_opnode_pos;
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
                            let op = self.ops[ppos].as_ref().unwrap();
                            match op.op {
                                Op::Diagonal(_, _, _, _) => states.push((ppos, state.clone())),
                                Op::OffDiagonal(vara, varb, _, _, (output_a, output_b)) => {
                                    state[vara] = output_a;
                                    state[varb] = output_b;
                                    states.push((ppos, state.clone()));
                                }
                            };
                            let ppos = op.next_p_op;
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
                    match opnode.op.get_vars() {
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

    fn make_loop_update<H>(
        &mut self,
        initial_n: Option<usize>,
        hamiltonian: H,
    ) -> Vec<(usize, bool)>
    where
        H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64,
    {
        let h = |op: Op, entrance: Leg, exit: Leg| -> f64 {
            let (entrance_var, entrace_side) = entrance;
            let (exit_var, exit_side) = exit;
            let (vara, varb, b, (mut a_bef, mut b_bef), (mut a_aft, mut b_aft)) = match op {
                Op::Diagonal(vara, varb, b, state) => (vara, varb, b, state, state),
                Op::OffDiagonal(vara, varb, b, bef, aft) => (vara, varb, b, bef, aft),
            };

            let mut adjust = |leg: Leg| {
                match leg {
                    (false, false) => {
                        a_bef = !a_bef;
                    }
                    (false, true) => {
                        a_aft = !a_aft;
                    }
                    (true, false) => {
                        b_bef = !b_bef;
                    }
                    (true, true) => {
                        b_aft = !b_aft;
                    }
                };
            };
            adjust(entrance);
            adjust(exit);
            // Call the supplied hamiltonian.
            hamiltonian(vara, varb, b, (a_bef, b_bef), (a_aft, b_aft))
        };

        let opnode = self
            .p_ends
            .map(|(p_start, _)| self.ops[p_start].as_ref().unwrap());
        if let Some(opnode) = opnode {
            let mut rng = rand::thread_rng();
            let initial_n = initial_n
                .map(|n| min(n, self.n))
                .unwrap_or_else(|| rng.gen_range(0, self.n));

            let initial_opnode = (0..initial_n).fold(opnode, |acc, _| {
                self.ops[opnode.next_p_op.unwrap()].as_ref().unwrap()
            });
            // Get the starting leg (vara/b, top/bottom).
            let initial_leg: Leg = (rng.gen(), rng.gen());
            let mut entrance_leg = initial_leg;
            //            while {
            //                let weights = [
            //                    h(opnode.op, entrance_leg, (false, false)),
            //                    h(opnode.op, entrance_leg, (false, true)),
            //                    h(opnode.op, entrance_leg, (true, false)),
            //                    h(opnode.op, entrance_leg, (true, true)),
            //                ];
            //                let total_weight = weights.sum();
            //            } {};
        }
        unimplemented!()
    }
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
                p_before = prev_fn(p_end);
            }
            // Now p_before < p < p_after
            (Some(p_before), Some(p_after))
        }
    }
}

#[cfg(test)]
mod fastops_tests {
    use super::*;

    #[test]
    fn add_single_item() {
        let mut f = FastOps::new(2);
        let op = Op::Diagonal(0, 1, 0, (true, true));
        f.set_pth(0, Some(op));
        println!("{:?}", f);
        assert_eq!(f.n, 1);
        assert_eq!(f.get_ops_for_var(0), vec![op]);
        assert_eq!(f.get_ops_for_var(1), vec![op]);
    }

    #[test]
    fn add_unrelated_item() {
        let mut f = FastOps::new(4);
        let opa = Op::Diagonal(0, 1, 0, (true, true));
        let opb = Op::Diagonal(2, 3, 1, (true, true));
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
        let opa = Op::Diagonal(0, 1, 0, (true, true));
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
        let opa = Op::Diagonal(0, 1, 0, (true, true));
        let opb = Op::Diagonal(1, 2, 0, (true, true));
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
        let opa = Op::Diagonal(0, 1, 0, (true, true));
        let opb = Op::Diagonal(1, 2, 0, (true, true));
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
        let opa = Op::Diagonal(0, 1, 0, (true, true));
        let opb = Op::Diagonal(1, 2, 0, (true, true));
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
        let opa = Op::Diagonal(0, 1, 0, (true, true));
        let opb = Op::Diagonal(1, 2, 0, (true, true));
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
        f.set_pth(0, Some(Op::Diagonal(0, 1, 0, (true, true))));
        f.set_pth(2, Some(Op::Diagonal(1, 2, 0, (true, true))));
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
        f.set_pth(0, Some(Op::Diagonal(0, 1, 0, (true, true))));
        f.set_pth(2, Some(Op::Diagonal(1, 2, 0, (true, true))));
        println!("{:?}", f);
        let states = f.get_p_states(&[true, false, true]);
        println!("{:?}", states);
    }

    #[test]
    fn get_states_offsimple() {
        let mut f = FastOps::new(3);
        f.set_pth(
            0,
            Some(Op::OffDiagonal(0, 1, 0, (false, true), (true, false))),
        );
        f.set_pth(
            1,
            Some(Op::OffDiagonal(1, 2, 0, (false, true), (true, false))),
        );
        f.set_pth(
            2,
            Some(Op::OffDiagonal(0, 2, 0, (false, true), (true, false))),
        );
        println!("{:?}", f);
        let states = f.get_p_states(&[false, true, true]);
        println!("{:?}", states);
    }
}

extern crate num;
use crate::qmc_traits::*;
use crate::qmc_types::*;
use std::cmp::{max, min};
use std::fmt::{Debug, Error, Formatter};

#[derive(Debug)]
pub(crate) struct LiveOpNode {
    op: Op,
    prev_vara_op: Option<usize>,
    next_vara_op: Option<usize>,
    prev_varb_op: Option<usize>,
    next_varb_op: Option<usize>,
    prev_p_op: Option<usize>,
    next_p_op: Option<usize>,
}

impl LiveOpNode {
    pub(crate) fn point_prev_past_vara(&mut self, opnode: &LiveOpNode) -> Option<usize> {
        let other_vara = opnode.op.vara;
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
    pub(crate) fn point_prev_past_varb(&mut self, opnode: &LiveOpNode) -> Option<usize> {
        let other_varb = opnode.op.varb;
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

    pub(crate) fn point_next_past_vara(&mut self, opnode: &LiveOpNode) -> Option<usize> {
        let other_vara = opnode.op.vara;
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

    pub(crate) fn point_next_past_varb(&mut self, opnode: &LiveOpNode) -> Option<usize> {
        let other_varb = opnode.op.varb;
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

impl OpNode for LiveOpNode {
    fn get_op(&self) -> Op {
        self.op.clone()
    }

    fn get_op_ref(&self) -> &Op {
        &self.op
    }

    fn get_op_mut(&mut self) -> &mut Op {
        &mut self.op
    }
}

pub(crate) struct LiveOps {
    ops: Vec<Option<LiveOpNode>>,
    p_ends: Option<(usize, usize)>,
    var_ends: Vec<Option<(usize, usize)>>,
    n: usize,
}

impl Debug for LiveOps {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.write_str(format!("p_ends:\t{:?}\tn: {}\n", self.p_ends, self.n).as_ref())?;
        f.write_str(format!("var_ends:\t{:?}\n", self.var_ends).as_ref())?;
        self.ops
            .iter()
            .enumerate()
            .try_for_each(|(p, opnode)| f.write_str(format!("{}:\t{:?}\n", p, opnode).as_ref()))
    }
}

impl LiveOps {
    pub fn new(nvars: usize) -> Self {
        Self {
            ops: Vec::default(),
            p_ends: None,
            var_ends: vec![None; nvars],
            n: 0,
        }
    }

    pub fn set_min_size(&mut self, n: usize) {
        if self.ops.len() < n {
            self.ops.resize_with(n, || None)
        }
    }

    pub fn does_var_have_ops(&self, var: usize) -> bool {
        self.var_ends[var].is_some()
    }

    pub(crate) fn swap_pth(&mut self, p: usize, op: Option<Op>) -> Option<Op> {
        match (self.ops[p].as_mut(), op) {
            (None, None) => None,
            (Some(mut opnode), Some(op)) => {
                let oldop = opnode.op.clone();
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

        let opnode = LiveOpNode {
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

    fn untie_op_node(&mut self, opnode: LiveOpNode) {
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
            ops: &[Option<LiveOpNode>],
            mut acc: Vec<Op>,
            opnode: Option<&LiveOpNode>,
        ) -> Vec<Op> {
            match opnode {
                None => acc,
                Some(opnode) => {
                    acc.push(opnode.op.clone());
                    match (opnode.op.vara, opnode.op.varb) {
                        (vara, _) if vara == var => rec_traverse(
                            var,
                            ops,
                            acc,
                            opnode.next_vara_op.map(|indx| ops[indx].as_ref().unwrap()),
                        ),
                        (_, varb) if varb == var => rec_traverse(
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

    fn p_matrix_weight<H>(&self, p: usize, h: &H) -> f64
    where
        H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64,
    {
        let op = self.ops[p].as_ref().unwrap();
        h(
            op.op.vara,
            op.op.varb,
            op.op.bond,
            op.op.inputs,
            op.op.outputs,
        )
    }
}

impl OpContainer for LiveOps {
    fn get_n(&self) -> usize {
        self.n
    }

    fn get_nvars(&self) -> usize {
        self.var_ends.len()
    }

    fn get_pth(&self, p: usize) -> Option<&Op> {
        if p >= self.ops.len() {
            None
        } else {
            self.ops[p].as_ref().map(|opnode| &opnode.op)
        }
    }

    fn weight<H>(&self, h: H) -> f64
    where
        H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64,
    {
        let mut t = 1.0;
        let mut p = self.p_ends.map(|(p, _)| p);
        while p.is_some() {
            let op = self.ops[p.unwrap()].as_ref().unwrap();
            t *= h(
                op.op.vara,
                op.op.varb,
                op.op.bond,
                op.op.inputs,
                op.op.outputs,
            );
            p = op.next_p_op;
        }
        t
    }
}

impl DiagonalUpdater for LiveOps {
    fn set_pth(&mut self, p: usize, op: Option<Op>) -> Option<Op> {
        self.set_min_size(p + 1);

        // Check if we can do this efficiently
        let opvars = op.as_ref().map(|op| (op.vara, op.varb));
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
                let retval = Some((opnode.op.clone(), opnode.prev_p_op, opnode.next_p_op));
                self.untie_op_node(opnode);
                retval
            }
            None => None,
        };

        // If we are just inserting an Identity (None), then we are done.
        // Otherwise we need to insert some stuff.
        if let Some(op) = &op {
            // Makes an opnode for the op, ties it into the graph.
            self.tie_op(p, op.clone(), old_opinfo.as_ref().map(|(_, p, n)| (*p, *n)))
        };

        if old_opinfo.is_some() {
            self.n -= 1;
        }
        if op.is_some() {
            self.n += 1;
        }
        old_opinfo.map(|(op, _, _)| op)
    }
}

impl LoopUpdater<LiveOpNode> for LiveOps {
    fn get_node_ref(&self, p: usize) -> Option<&LiveOpNode> {
        self.ops[p].as_ref()
    }

    fn get_node_mut(&mut self, p: usize) -> Option<&mut LiveOpNode> {
        self.ops[p].as_mut()
    }

    fn get_first_p(&self) -> Option<usize> {
        self.p_ends.map(|(p, _)| p)
    }

    fn get_last_p(&self) -> Option<usize> {
        self.p_ends.map(|(_, p)| p)
    }

    fn get_first_p_for_var(&self, var: usize) -> Option<usize> {
        self.var_ends[var].map(|(p, _)| p)
    }

    fn get_last_p_for_var(&self, var: usize) -> Option<usize> {
        self.var_ends[var].map(|(_, p)| p)
    }

    fn get_previous_p(&self, node: &LiveOpNode) -> Option<usize> {
        node.prev_p_op
    }

    fn get_next_p(&self, node: &LiveOpNode) -> Option<usize> {
        node.next_p_op
    }

    fn get_previous_p_for_var(&self, var: usize, node: &LiveOpNode) -> Option<usize> {
        if var == node.op.vara {
            node.prev_vara_op
        } else if var == node.op.varb {
            node.prev_varb_op
        } else {
            unreachable!()
        }
    }

    fn get_next_p_for_var(&self, var: usize, node: &LiveOpNode) -> Option<usize> {
        if var == node.op.vara {
            node.next_vara_op
        } else if var == node.op.varb {
            node.next_varb_op
        } else {
            unreachable!()
        }
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
        Some((_, p_end)) => {
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
        let mut f = LiveOps::new(2);
        let op = Op::diagonal(0, 1, 0, (true, true));
        f.set_pth(0, Some(op.clone()));
        println!("{:?}", f);
        assert_eq!(f.n, 1);
        assert_eq!(f.get_ops_for_var(0), vec![op.clone()]);
        assert_eq!(f.get_ops_for_var(1), vec![op]);
    }

    #[test]
    fn add_unrelated_item() {
        let mut f = LiveOps::new(4);
        let opa = Op::diagonal(0, 1, 0, (true, true));
        let opb = Op::diagonal(2, 3, 1, (true, true));
        f.set_pth(0, Some(opa.clone()));
        f.set_pth(1, Some(opb.clone()));
        println!("{:?}", f);
        assert_eq!(f.n, 2);

        assert_eq!(f.get_ops_for_var(0), vec![opa.clone()]);
        assert_eq!(f.get_ops_for_var(1), vec![opa]);
        assert_eq!(f.get_ops_for_var(2), vec![opb.clone()]);
        assert_eq!(f.get_ops_for_var(3), vec![opb]);
    }

    #[test]
    fn add_identical_item() {
        let mut f = LiveOps::new(2);
        let opa = Op::diagonal(0, 1, 0, (true, true));
        f.set_pth(0, Some(opa.clone()));
        f.set_pth(1, Some(opa.clone()));
        println!("{:?}", f);
        assert_eq!(f.n, 2);

        assert_eq!(f.get_ops_for_var(0), vec![opa.clone(), opa.clone()]);
        assert_eq!(f.get_ops_for_var(1), vec![opa.clone(), opa.clone()]);
    }

    #[test]
    fn add_overlapping_item() {
        let mut f = LiveOps::new(3);
        let opa = Op::diagonal(0, 1, 0, (true, true));
        let opb = Op::diagonal(1, 2, 0, (true, true));
        f.set_pth(0, Some(opa.clone()));
        f.set_pth(1, Some(opb.clone()));
        println!("{:?}", f);
        assert_eq!(f.n, 2);

        assert_eq!(f.get_ops_for_var(0), vec![opa.clone()]);
        assert_eq!(f.get_ops_for_var(1), vec![opa, opb.clone()]);
        assert_eq!(f.get_ops_for_var(2), vec![opb]);
    }

    #[test]
    fn add_skipping_item() {
        let mut f = LiveOps::new(3);
        let opa = Op::diagonal(0, 1, 0, (true, true));
        let opb = Op::diagonal(1, 2, 0, (true, true));
        f.set_pth(0, Some(opa.clone()));
        f.set_pth(2, Some(opb.clone()));
        println!("{:?}", f);
        assert_eq!(f.n, 2);

        assert_eq!(f.get_ops_for_var(0), vec![opa.clone()]);
        assert_eq!(f.get_ops_for_var(1), vec![opa, opb.clone()]);
        assert_eq!(f.get_ops_for_var(2), vec![opb]);
    }

    #[test]
    fn add_skipping_item_and_remove() {
        let mut f = LiveOps::new(3);
        let opa = Op::diagonal(0, 1, 0, (true, true));
        let opb = Op::diagonal(1, 2, 0, (true, true));
        f.set_pth(0, Some(opa.clone()));
        f.set_pth(2, Some(opb));
        f.set_pth(2, None);
        println!("{:?}", f);
        assert_eq!(f.n, 1);

        assert_eq!(f.get_ops_for_var(0), vec![opa.clone()]);
        assert_eq!(f.get_ops_for_var(1), vec![opa]);
        assert_eq!(f.get_ops_for_var(2), vec![]);
    }

    #[test]
    fn add_skipping_item_and_remove_first() {
        let mut f = LiveOps::new(3);
        let opa = Op::diagonal(0, 1, 0, (true, true));
        let opb = Op::diagonal(1, 2, 0, (true, true));
        f.set_pth(0, Some(opa));
        f.set_pth(2, Some(opb.clone()));
        f.set_pth(0, None);
        println!("{:?}", f);
        assert_eq!(f.n, 1);

        assert_eq!(f.get_ops_for_var(0), vec![]);
        assert_eq!(f.get_ops_for_var(1), vec![opb.clone()]);
        assert_eq!(f.get_ops_for_var(2), vec![opb]);
    }

    #[test]
    fn add_and_remove_all() {
        let mut f = LiveOps::new(3);
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
        let mut f = LiveOps::new(3);
        f.set_pth(0, Some(Op::diagonal(0, 1, 0, (true, true))));
        f.set_pth(2, Some(Op::diagonal(1, 2, 0, (true, true))));
        println!("{:?}", f);
        let states = f.get_p_states(&[true, false, true]);
        println!("{:?}", states);
    }

    #[test]
    fn get_states_offsimple() {
        let mut f = LiveOps::new(3);
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
        let mut f = LiveOps::new(2);
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
        let mut f = LiveOps::new(3);
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

        let mut f = LiveOps::new(3);
        let mut rng: ChaCha20Rng = rand::SeedableRng::seed_from_u64(12345678);
        f.make_diagonal_update_with_rng(
            10,
            1.0,
            &[false, false, false],
            h,
            (edges.len(), |i| edges[i]),
            &mut rng,
        );
    }

    #[test]
    fn test_diagonal_updates() {
        let edges = vec![(0, 1), (1, 2)];

        let h = |_, _, _, _, _| 1.0;

        let mut f = LiveOps::new(3);
        let mut rng: ChaCha20Rng = rand::SeedableRng::seed_from_u64(12345678);
        f.make_diagonal_update_with_rng(
            10,
            1.0,
            &[false, false, false],
            h,
            (edges.len(), |i| edges[i]),
            &mut rng,
        );
        f.make_diagonal_update_with_rng(
            10,
            1.0,
            &[false, false, false],
            h,
            (edges.len(), |i| edges[i]),
            &mut rng,
        );
        f.make_diagonal_update_with_rng(
            10,
            1.0,
            &[false, false, false],
            h,
            (edges.len(), |i| edges[i]),
            &mut rng,
        );
        f.make_diagonal_update_with_rng(
            10,
            1.0,
            &[false, false, false],
            h,
            (edges.len(), |i| edges[i]),
            &mut rng,
        );
    }
}

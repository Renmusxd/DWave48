use crate::sse::qmc_traits::*;
use crate::sse::qmc_types::TwoSiteOp;

pub struct SimpleOpDiagonal {
    ops: Vec<Option<TwoSiteOp>>,
    n: usize,
    nvars: usize,
}

impl SimpleOpDiagonal {
    pub fn new(nvars: usize) -> Self {
        Self {
            ops: vec![],
            n: 0,
            nvars,
        }
    }

    pub(crate) fn set_min_size(&mut self, n: usize) {
        if self.ops.len() < n {
            self.ops.resize_with(n, || None)
        }
    }

    pub fn convert_to_looper(self) -> SimpleOpLooper {
        let mut p_ends = None;
        let mut var_ends = vec![None; self.nvars];
        let mut opnodes = self
            .ops
            .iter()
            .map(|op| op.clone().map(SimpleOpNode::new_empty))
            .collect::<Vec<_>>();
        let mut nth_ps = vec![];
        self.ops
            .iter()
            .enumerate()
            .filter(|op| op.1.is_some())
            .map(|(p, op)| (p, op.as_ref().unwrap()))
            .for_each(|(p, op)| {
                nth_ps.push(p);
                match p_ends {
                    None => p_ends = Some((p, p)),
                    Some((_, last_p)) => {
                        let last_op = opnodes[last_p].as_mut().unwrap();
                        last_op.next_p = Some(p);

                        p_ends.as_mut().unwrap().1 = p;

                        let this_opnode = opnodes[p].as_mut().unwrap();
                        this_opnode.previous_p = Some(last_p);
                    }
                }
                match var_ends[op.vara] {
                    None => var_ends[op.vara] = Some((p, p)),
                    Some((_, last_p)) => {
                        let last_op = opnodes[last_p].as_mut().unwrap();
                        if last_op.op.vara == op.vara {
                            last_op.next_vara = Some(p);
                        } else if last_op.op.varb == op.vara {
                            last_op.next_varb = Some(p);
                        } else {
                            unreachable!()
                        }
                        var_ends[op.vara].as_mut().unwrap().1 = p;
                        let this_opnode = opnodes[p].as_mut().unwrap();
                        this_opnode.previous_vara = Some(last_p);
                    }
                }
                match var_ends[op.varb] {
                    None => var_ends[op.varb] = Some((p, p)),
                    Some((_, last_p)) => {
                        let last_op = opnodes[last_p].as_mut().unwrap();
                        if last_op.op.vara == op.varb {
                            last_op.next_vara = Some(p);
                        } else if last_op.op.varb == op.varb {
                            last_op.next_varb = Some(p);
                        } else {
                            unreachable!()
                        }
                        var_ends[op.varb].as_mut().unwrap().1 = p;
                        let this_opnode = opnodes[p].as_mut().unwrap();
                        this_opnode.previous_varb = Some(last_p);
                    }
                }
            });
        SimpleOpLooper {
            ops: opnodes,
            nth_ps,
            p_ends,
            var_ends,
        }
    }
}

impl OpContainer for SimpleOpDiagonal {
    fn get_n(&self) -> usize {
        self.n
    }

    fn get_nvars(&self) -> usize {
        self.nvars
    }

    fn get_pth(&self, p: usize) -> Option<&TwoSiteOp> {
        if p >= self.ops.len() {
            None
        } else {
            self.ops[p].as_ref()
        }
    }

    fn weight<H>(&self, h: H) -> f64
    where
        H: Fn(usize, usize, usize, (bool, bool), (bool, bool)) -> f64,
    {
        self.ops
            .iter()
            .filter(|op| op.is_some())
            .fold(1.0, |t, op| {
                let op = op.as_ref().unwrap();
                h(op.vara, op.varb, op.bond, op.inputs, op.outputs) * t
            })
    }
}

impl DiagonalUpdater for SimpleOpDiagonal {
    fn set_pth(&mut self, p: usize, op: Option<TwoSiteOp>) -> Option<TwoSiteOp> {
        self.set_min_size(p + 1);
        let temp = self.ops[p].take();
        self.ops[p] = op;
        temp
    }
}

pub struct SimpleOpNode {
    op: TwoSiteOp,
    previous_p: Option<usize>,
    next_p: Option<usize>,
    previous_vara: Option<usize>,
    next_vara: Option<usize>,
    previous_varb: Option<usize>,
    next_varb: Option<usize>,
}

impl SimpleOpNode {
    fn new_empty(op: TwoSiteOp) -> Self {
        Self {
            op,
            previous_p: None,
            next_p: None,
            previous_vara: None,
            next_vara: None,
            previous_varb: None,
            next_varb: None,
        }
    }
}

impl OpNode for SimpleOpNode {
    fn get_op(&self) -> TwoSiteOp {
        self.op.clone()
    }

    fn get_op_ref(&self) -> &TwoSiteOp {
        &self.op
    }

    fn get_op_mut(&mut self) -> &mut TwoSiteOp {
        &mut self.op
    }
}

pub struct SimpleOpLooper {
    ops: Vec<Option<SimpleOpNode>>,
    nth_ps: Vec<usize>,
    p_ends: Option<(usize, usize)>,
    var_ends: Vec<Option<(usize, usize)>>,
}

impl SimpleOpLooper {
    pub fn convert_to_diagonal(self) -> SimpleOpDiagonal {
        let n = self.get_n();
        let nvars = self.get_nvars();
        let ops = self
            .ops
            .into_iter()
            .map(|opnode| opnode.map(|opnode| opnode.op))
            .collect();
        SimpleOpDiagonal { ops, n, nvars }
    }
}

impl OpContainer for SimpleOpLooper {
    fn get_n(&self) -> usize {
        self.nth_ps.len()
    }

    fn get_nvars(&self) -> usize {
        self.var_ends.len()
    }

    fn get_pth(&self, p: usize) -> Option<&TwoSiteOp> {
        self.ops[p].as_ref().map(|opnode| &opnode.op)
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
            p = op.next_p;
        }
        t
    }
}

impl LoopUpdater<SimpleOpNode> for SimpleOpLooper {
    fn get_node_ref(&self, p: usize) -> Option<&SimpleOpNode> {
        self.ops[p].as_ref()
    }

    fn get_node_mut(&mut self, p: usize) -> Option<&mut SimpleOpNode> {
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

    fn get_previous_p(&self, node: &SimpleOpNode) -> Option<usize> {
        node.previous_p
    }

    fn get_next_p(&self, node: &SimpleOpNode) -> Option<usize> {
        node.next_p
    }

    fn get_previous_p_for_var(&self, var: usize, node: &SimpleOpNode) -> Option<usize> {
        if var == node.op.vara {
            node.previous_vara
        } else if var == node.op.varb {
            node.previous_varb
        } else {
            unreachable!()
        }
    }

    fn get_next_p_for_var(&self, var: usize, node: &SimpleOpNode) -> Option<usize> {
        if var == node.op.vara {
            node.next_vara
        } else if var == node.op.varb {
            node.next_varb
        } else {
            unreachable!()
        }
    }

    fn get_nth_p(&self, n: usize) -> usize {
        self.nth_ps[n]
    }
}

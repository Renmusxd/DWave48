#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Op {
    pub vara: usize,
    pub varb: usize,
    pub bond: usize,
    pub inputs: (bool, bool),
    pub outputs: (bool, bool),
}

impl Op {
    pub fn diagonal(vara: usize, varb: usize, bond: usize, state: (bool, bool)) -> Self {
        Self {
            vara,
            varb,
            bond,
            inputs: state,
            outputs: state,
        }
    }
    pub fn offdiagonal(
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
    pub fn is_diagonal(&self) -> bool {
        self.inputs == self.outputs
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Variable {
    A,
    B,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum OpSide {
    Inputs,
    Outputs,
}

impl OpSide {
    pub fn reverse(self) -> Self {
        match self {
            OpSide::Inputs => OpSide::Outputs,
            OpSide::Outputs => OpSide::Inputs,
        }
    }
}

pub type Leg = (Variable, OpSide);

pub static LEGS: [Leg; 4] = [
    (Variable::A, OpSide::Inputs),
    (Variable::A, OpSide::Outputs),
    (Variable::B, OpSide::Inputs),
    (Variable::B, OpSide::Outputs),
];

pub fn adjust_states(
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

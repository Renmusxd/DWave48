use rand::prelude::*;
use std::fmt::{Debug, Error, Formatter};

pub struct GraphState {
    binding_mat: Vec<Vec<(usize, f64)>>,
    biases: Vec<f64>,
    state: Option<Vec<bool>>,
}

impl Debug for GraphState {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        if let Some(state) = &self.state {
            let s = state
                .iter()
                .map(|b| if *b { "1" } else { "0" })
                .collect::<Vec<_>>()
                .join("");
            let e = self.get_energy();
            f.write_str(&format!("{}\t{}", s, e))
        } else {
            f.write_str("Error")
        }
    }
}

pub type Edge = (usize, usize);
impl GraphState {
    pub fn new(edges: &[(Edge, f64)], biases: &[f64]) -> Self {
        // Matrix of all bonds.
        let mut binding_mat: Vec<Vec<(usize, f64)>> = vec![vec![]; biases.len() * biases.len()];

        edges.iter().for_each(|((va, vb), j)| {
            binding_mat[*va].push((*vb, *j));
            binding_mat[*vb].push((*va, *j));
        });
        // Sort just in case
        binding_mat.iter_mut().for_each(|vs| {
            vs.sort_by_key(|(i, _)| *i);
        });

        GraphState {
            binding_mat,
            biases: biases.to_vec(),
            state: Some(GraphState::make_random_spin_state(biases.len())),
        }
    }

    pub fn do_time_step(&mut self, beta: f64) -> Result<(), String> {
        let mut rng = rand::thread_rng();
        let random_index = rng.gen_range(0, self.biases.len());

        // Energy cost of this flip
        if let Some(mut spin_state) = self.state.take() {
            let curr_value = spin_state[random_index];
            // new - old
            let binding_slice = &self.binding_mat[random_index];
            let delta_e: f64 = binding_slice
                .iter()
                .map(|(indx, j)| {
                    let old_coupling = if (curr_value ^ spin_state[*indx]) == false {
                        1.0
                    } else {
                        -1.0
                    };
                    // j*new - j*old = j*(-old) - j*(old) = -2j*(old)
                    -2.0 * (*j) * old_coupling
                })
                .sum();
            let delta_e =
                delta_e + (2.0 * self.biases[random_index] * if curr_value { 1.0 } else { -1.0 });
            // If dE < 0 then it will always flip, don't bother calculating odds.
            let should_flip = if delta_e > 0.0 {
                let chance = (-beta * delta_e).exp();
                rng.gen::<f64>() < chance
            } else {
                true
            };
            if should_flip {
                spin_state[random_index] = !spin_state[random_index]
            }
            self.state = Some(spin_state);
            Ok(())
        } else {
            Err("No state to edit".to_string())
        }
    }

    pub fn get_state(self) -> Vec<bool> {
        self.state.unwrap()
    }

    pub fn get_energy(&self) -> f64 {
        if let Some(state) = &self.state {
            state.iter().enumerate().fold(0.0, |acc, (i, si)| {
                let binding_slice = &self.binding_mat[i];
                let total_e: f64 = binding_slice
                    .iter()
                    .map(|(indx, j)| -> f64 {
                        let old_coupling = if (si ^ state[*indx]) == false {
                            1.0
                        } else {
                            -1.0
                        };
                        j * old_coupling / 2.0
                    })
                    .sum();
                acc + total_e
            })
        } else {
            std::f64::NAN
        }
    }

    fn make_random_spin_state(n: usize) -> Vec<bool> {
        (0..n).map(|_| -> bool { rand::random() }).collect()
    }
}

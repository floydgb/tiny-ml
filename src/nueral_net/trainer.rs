// Imports --------------------------------------------------------------------
use super::*;

// Types ----------------------------------------------------------------------
#[derive(Default)]
pub struct Trainer<const N: usize, const O: usize> {
    pub inputs: Vec<[f32; N]>,
    pub labels: Vec<[f32; O]>,
}

// Public Functions -----------------------------------------------------------
impl<const N: usize, const O: usize> Trainer<N, O> {
    pub fn train(&self, net: &mut NeuralNet<N, O>, iterations: usize) -> f32 {
        let mut cur_err = self.compute_error(net);
        for _ in 0..=iterations {
            net.random_edit();
            let next_err = self.compute_error(net);
            if cur_err < next_err {
                net.undo_edit();
            } else {
                cur_err = next_err;
            }
        }
        cur_err
    }

    pub fn compute_error(&self, net: &NeuralNet<N, O>) -> f32 {
        self.inputs
            .par_iter()
            .zip(&self.labels)
            .map(|(input, label)| {
                label
                    .iter()
                    .zip(net.run(input))
                    .fold(0.0, |acc, (a, b)| acc + (a - b).abs())
            })
            .sum::<f32>()
            / self.inputs.len() as f32
    }
}

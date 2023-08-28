// Imports --------------------------------------------------------------------
use crate::nueral_net::NeuralNet;
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::usize;

// Types ----------------------------------------------------------------------
pub struct Trainer<const N: usize, const O: usize> {
    training_data: TrainingData<N, O>,
}

#[derive(Default)]
pub struct TrainingData<const N: usize, const O: usize> {
    pub inputs: Vec<[f32; N]>,
    pub outputs: Vec<[f32; O]>,
}

// Public Functions -----------------------------------------------------------
impl<const N: usize, const O: usize> Trainer<N, O> {
    pub fn new(data: TrainingData<N, O>) -> Self {
        Self {
            training_data: data,
        }
    }

    pub fn train(&self, net: &mut NeuralNet<N, O>, iterations: usize) -> f32 {
        let mut pre = self.compute_total_error(net);
        for _ in 0..=iterations {
            net.random_edit();
            let post = self.compute_total_error(net);
            if pre < post {
                net.undo_edit();
            } else {
                pre = post;
            }
        }
        pre
    }

    pub fn compute_total_error(&self, net: &NeuralNet<N, O>) -> f32 {
        self.training_data
            .inputs
            .par_iter()
            .zip(&self.training_data.outputs)
            .map(|(input, output)| {
                output
                    .iter()
                    .zip(&mut net.run(input))
                    .fold(0.0, |dist, x| dist + (*x.0 - *x.1).abs())
            })
            .sum()
    }
}

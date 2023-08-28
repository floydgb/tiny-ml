// Exports --------------------------------------------------------------------
mod neuron;

// Imports --------------------------------------------------------------------
use self::neuron::Neuron;
use rand::Rng;
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};
use serde_derive::{Deserialize, Serialize};

// Types ----------------------------------------------------------------------
#[derive(Serialize, Deserialize)]
pub struct NeuralNet<const I: usize, const O: usize> {
    layers: Vec<Vec<Neuron>>,
    last_edit: Option<Edit>,
    longest_layer: usize,
}

#[derive(Serialize, Deserialize)]
struct Edit {
    old: Neuron,
    layer: usize,
    row: usize,
}

#[derive(Serialize, Deserialize, Debug, Default, Clone, Copy)]
pub enum Fn {
    #[default]
    ReLU,
    Linear,
}

// Public Functions -----------------------------------------------------------
impl<const I: usize, const O: usize> Default for NeuralNet<I, O> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const I: usize, const O: usize> NeuralNet<I, O> {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            last_edit: None,
            longest_layer: 0,
        }
    }

    pub fn add_layer(mut self, n: usize, func: Fn) -> Self {
        let n_inputs = self.get_last_layer_len();
        self.layers.push(vec![Neuron::new(n_inputs, 0.0, func); n]);
        self.set_longest_layer(n);
        self
    }

    pub fn set_longest_layer(&mut self, n: usize) {
        if n > self.longest_layer {
            self.longest_layer = n;
        }
    }

    pub fn add_random_layer(mut self, n: usize, func: Fn) -> Self {
        let mut layer: Vec<Neuron> = vec![];
        for _ in 0..n {
            layer.push(Neuron::random(self.get_last_layer_len(), func))
        }
        self.layers.push(layer);
        self.set_longest_layer(n);
        self
    }

    pub fn random_edit(&mut self) {
        let mut rng = rand::thread_rng();
        let layer = rng.gen_range(0..self.layers.len());
        let row = rng.gen_range(0..self.layers[layer].len());
        let neuron = &mut self.layers[layer][row];
        let mut change: f32 = rng.gen::<f32>() / 10.0;
        if rng.gen_bool(0.5) {
            change *= -1.0;
        }
        self.last_edit = Some(Edit {
            old: neuron.clone(),
            layer,
            row,
        });
        if rng.gen_bool(0.95) {
            let index = rng.gen_range(0..neuron.get_weights_len());
            neuron.change_weight(index, change);
        } else {
            neuron.change_bias(change);
        }
    }

    pub fn undo_edit(&mut self) {
        match &self.last_edit {
            Some(edit) => {
                self.layers[edit.layer][edit.row] = edit.old.clone();
            }
            None => {}
        }
    }

    fn get_last_layer_len(&self) -> usize {
        if self.layers.is_empty() {
            return I;
        }
        self.layers[self.layers.len() - 1].len()
    }

    pub fn with_weights(mut self, weights: Vec<Vec<f32>>) -> Self {
        match self.layers.last_mut() {
            None => panic!("tried to add weights before layers!"),
            Some(layer) => layer
                .iter_mut()
                .zip(weights)
                .for_each(|(neuron, weight)| neuron.set_weights(weight)),
        }
        self
    }

    pub fn with_bias(mut self, biases: Vec<f32>) -> Self {
        match self.layers.last_mut() {
            None => panic!("tried to add biases before layers!"),
            Some(layer) => layer
                .iter_mut()
                .zip(biases)
                .for_each(|(neuron, bias)| neuron.set_bias(bias)),
        }
        self
    }

    pub fn run(&self, input: &[f32; I]) -> [f32; O] {
        let mut data = vec![0.0; self.longest_layer];
        let mut temp = vec![0.0; self.longest_layer];
        let mut out = [0.0; O];
        data[..input.len()].copy_from_slice(&input[..]);
        for layer in &self.layers {
            for (i, neuron) in layer.iter().enumerate() {
                temp[i] = neuron.compute(&data);
            }
            (data, temp) = (temp, data);
        }
        out[..O].copy_from_slice(&data[..O]);
        out
    }

    pub fn par_run(&self, inputs: &Vec<[f32; I]>) -> Vec<[f32; O]> {
        inputs.par_iter().map(|input| self.run(input)).collect()
    }
}

// Tests ----------------------------------------------------------------------
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn minimal_test() {
        let net = NeuralNet::new()
            .add_layer(10, Fn::ReLU)
            .add_layer(5, Fn::Linear);
        assert_eq!(net.run(&[1.0]), [10.0; 5])
    }

    #[test]
    fn better_test() {
        let net: NeuralNet<2, 1> = NeuralNet::new()
            .add_layer(4, Fn::ReLU)
            .with_weights(vec![
                vec![1.1, -0.93],
                vec![-0.9, -0.96],
                vec![1.2, 0.81],
                vec![-0.91, 0.95],
            ])
            .with_bias(vec![0.048, 0.12, 0.083, -0.02])
            .add_layer(1, Fn::Linear)
            .with_weights(vec![vec![-1.4, 1.3, 1.4, -1.3]]);
        assert!(net.run(&[3.0, 3.0])[0] > 0.0);
        assert!(net.run(&[-3.0, -3.0])[0] > 0.0);
        assert!(net.run(&[3.0, -3.0])[0] < 0.0);
        assert!(net.run(&[-3.0, 3.0])[0] < 0.0);
    }
}

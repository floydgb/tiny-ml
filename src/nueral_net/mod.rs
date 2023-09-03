// Exports --------------------------------------------------------------------
pub mod neuron;
pub mod trainer;

// Imports --------------------------------------------------------------------
use {
    self::neuron::Neuron,
    rand::{random, thread_rng, Rng},
    rayon::prelude::*,
    serde_derive::{Deserialize, Serialize},
    std::mem::swap,
};

// Constants ------------------------------------------------------------------
pub const LEARNING_RATE: f32 = 0.055731735;
pub const WEIGHT_TO_BIAS_RATIO: f64 = 0.95;

// Types ----------------------------------------------------------------------
#[derive(Default, Serialize, Deserialize)]
pub struct NeuralNet<const I: usize, const O: usize> {
    layers: Vec<Vec<Neuron>>,
    last_edit: Edit,
}

#[derive(Default, Serialize, Deserialize)]
struct Edit {
    old: Neuron,
    layer: usize,
    row: usize,
}

#[derive(Default, Serialize, Deserialize, Debug, Clone, Copy)]
pub enum ActivationFn {
    #[default]
    ReLU,
    Linear,
    Sigmoid,
    Tanh,
}

// Public Functions -----------------------------------------------------------
impl<const I: usize, const O: usize> NeuralNet<I, O> {
    pub fn add_layer(mut self, n: usize, func: ActivationFn) -> Self {
        let new_layer = vec![Neuron::new(self.last_layer_len(), 0.0, func); n];
        self.layers.push(new_layer);
        self
    }

    pub fn random_edit(&mut self) {
        match thread_rng().gen_bool(WEIGHT_TO_BIAS_RATIO) {
            true => self.random_neuron().change_weight(),
            false => self.random_neuron().change_bias(),
        }
    }

    pub fn undo_edit(&mut self) {
        match &self.last_edit {
            Edit { layer, row, old } => self.layers[*layer][*row] = old.clone(),
        }
    }

    pub fn with_weights(mut self, weights: Vec<Vec<f32>>) -> Self {
        match self.layers.last_mut() {
            None => panic!("tried to add weights before layers!"),
            Some(layer) => layer
                .iter_mut()
                .zip(weights)
                .for_each(|(neuron, weights)| neuron.weights = weights),
        }
        self
    }

    pub fn with_bias(mut self, biases: Vec<f32>) -> Self {
        match self.layers.last_mut() {
            None => panic!("tried to add biases before layers!"),
            Some(layer) => layer
                .iter_mut()
                .zip(biases)
                .for_each(|(neuron, bias)| neuron.bias = bias),
        }
        self
    }

    pub fn max_len(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.len())
            .max()
            .unwrap_or(0)
    }

    pub fn run(&self, input: &[f32; I]) -> [f32; O] {
        let (mut buf1, mut buf2) = (vec![0.0; self.max_len()], vec![0.0; self.max_len()]);
        buf1[..input.len()].copy_from_slice(input);
        for layer in &self.layers {
            for (i, neuron) in layer.iter().enumerate() {
                buf2[i] = neuron.compute(&buf1);
            }
            swap(&mut buf1, &mut buf2);
        }
        let mut out = [0.0; O];
        out.copy_from_slice(&buf1[..O]);
        out
    }

    // Private Functions ------------------------------------------------------
    fn random_neuron(&mut self) -> &mut Neuron {
        let (layer, row) = self.random_index();
        let neuron = &mut self.layers[layer][row];
        self.last_edit = Edit {
            old: neuron.clone(),
            layer,
            row,
        };
        neuron
    }

    fn random_index(&mut self) -> (usize, usize) {
        let layer = thread_rng().gen_range(0..self.layers.len());
        let row = thread_rng().gen_range(0..self.layers[layer].len());
        (layer, row)
    }

    fn last_layer_len(&self) -> usize {
        if self.layers.is_empty() {
            return I;
        }
        self.layers[self.layers.len() - 1].len()
    }
}

// Tests ----------------------------------------------------------------------
#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn minimal_test() {
        let net = NeuralNet::default()
            .add_layer(10, ActivationFn::ReLU)
            .add_layer(5, ActivationFn::Linear);
        assert_eq!(net.run(&[1.0]), [0.0; 5])
    }

    #[test]
    fn better_test() {
        let net: NeuralNet<2, 1> = NeuralNet::default()
            .add_layer(4, ActivationFn::ReLU)
            .with_weights(vec![
                vec![1.1, -0.93],
                vec![-0.9, -0.96],
                vec![1.2, 0.81],
                vec![-0.91, 0.95],
            ])
            .with_bias(vec![0.048, 0.12, 0.083, -0.02])
            .add_layer(1, ActivationFn::Linear)
            .with_weights(vec![vec![-1.4, 1.3, 1.4, -1.3]]);
        assert!(net.run(&[3.0, 3.0]) > [0.0]);
        assert!(net.run(&[-3.0, -3.0]) > [0.0]);
        assert!(net.run(&[3.0, -3.0]) < [0.0]);
        assert!(net.run(&[-3.0, 3.0]) < [0.0]);
    }
}

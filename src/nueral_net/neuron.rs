// Imports --------------------------------------------------------------------
use {
    super::*,
    std::simd::{f32x16, f32x2, f32x4, f32x8, SimdFloat},
};

// Types ----------------------------------------------------------------------
#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct Neuron {
    weights: Vec<f32>,
    bias: f32,
    activation: ActivationFn,
}

// Public Functions -----------------------------------------------------------
impl Neuron {
    pub fn new(n_inputs: usize, bias: f32, func: ActivationFn) -> Self {
        Self {
            weights: vec![1.0; n_inputs],
            bias,
            activation: func,
        }
    }

    pub fn random(n_inputs: usize, func: ActivationFn) -> Self {
        let mut weights = vec![];
        for _ in 0..n_inputs {
            weights.push(random())
        }
        Self {
            weights,
            bias: random(),
            activation: func,
        }
    }

    pub fn set_weights(&mut self, weights: Vec<f32>) {
        self.weights = weights
    }

    pub fn change_weight(&mut self) {
        let random_index = rand::thread_rng().gen_range(0..self.get_weights_len());
        self.weights[random_index] += random::<f32>() * random_sign() * LEARNING_RATE;
    }

    pub fn get_weights_len(&self) -> usize {
        self.weights.len()
    }

    pub fn set_bias(&mut self, bias: f32) {
        self.bias = bias
    }

    pub fn change_bias(&mut self) {
        self.bias += random::<f32>() * random_sign() * LEARNING_RATE;
    }

    pub fn compute(&self, x: &[f32]) -> f32 {
        let mut remaining_length = self.weights.len();
        let mut i = 0;
        let mut res = self.bias;
        while remaining_length >= 16 {
            let simd_weights = f32x16::from_slice(&self.weights[i..i + 16]);
            let simd_input = f32x16::from_slice(&x[i..i + 16]);
            res += (simd_input * simd_weights).reduce_sum();
            i += 16;
            remaining_length -= 16;
        }
        while remaining_length >= 8 {
            let simd_weights = f32x8::from_slice(&self.weights[i..i + 8]);
            let simd_input = f32x8::from_slice(&x[i..i + 8]);
            res += (simd_input * simd_weights).reduce_sum();
            i += 8;
            remaining_length -= 8;
        }
        while remaining_length >= 4 {
            let simd_weights = f32x4::from_slice(&self.weights[i..i + 4]);
            let simd_input = f32x4::from_slice(&x[i..i + 4]);
            res += (simd_input * simd_weights).reduce_sum();
            i += 4;
            remaining_length -= 4;
        }
        while remaining_length >= 2 {
            let simd_weights = f32x2::from_slice(&self.weights[i..i + 2]);
            let simd_input = f32x2::from_slice(&x[i..i + 2]);
            res += (simd_input * simd_weights).reduce_sum();
            i += 2;
            remaining_length -= 2;
        }
        if remaining_length == 1 {
            res += x[i] * self.weights[i];
        }
        match self.activation {
            ActivationFn::Linear => res,
            ActivationFn::ReLU => res.max(0.0),
        }
    }
}

fn random_sign() -> f32 {
    match rand::thread_rng().gen_bool(POSITIVE_BIAS) {
        true => -1.0,
        false => 1.0,
    }
}

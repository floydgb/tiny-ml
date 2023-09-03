// Imports --------------------------------------------------------------------
use {
    super::*,
    crate::{compute_simd, compute_with_simd},
    std::simd::{f32x16, f32x2, f32x4, f32x8, SimdFloat},
};

// Constants ------------------------------------------------------------------
const SIMD_SIZES: [usize; 5] = [16, 8, 4, 2, 1];

// Types ----------------------------------------------------------------------
#[derive(Serialize, Deserialize, Default, Debug, Clone)]
pub struct Neuron {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub activation: ActivationFn,
}

// Public Functions -----------------------------------------------------------
impl Neuron {
    pub fn new(n_inputs: usize, bias: f32, func: ActivationFn) -> Self {
        Self {
            weights: vec![0.0; n_inputs],
            bias,
            activation: func,
        }
    }

    pub fn new_random(n_inputs: usize, func: ActivationFn) -> Neuron {
        Self {
            weights: thread_rng()
                .sample_iter(Standard)
                .take(n_inputs)
                .collect::<Vec<f32>>(),
            bias: random(),
            activation: func,
        }
    }

    pub fn change_weight(&mut self) {
        let random_index = rand::thread_rng().gen_range(0..self.weights.len());
        self.weights[random_index] += random::<f32>() * random_sign() * LEARNING_RATE;
    }

    pub fn change_bias(&mut self) {
        self.bias += random::<f32>() * random_sign() * LEARNING_RATE;
    }

    pub fn compute(&self, x: &[f32]) -> f32 {
        let (mut i, mut result) = (0, self.bias);
        for size in SIMD_SIZES {
            while i + size <= self.weights.len() {
                result += compute_with_simd!(size, &self.weights[i..i + size], &x[i..i + size]);
                i += size;
            }
        }
        self.activate(result)
    }

    pub fn activate(&self, res: f32) -> f32 {
        match self.activation {
            ActivationFn::Linear => res,
            ActivationFn::ReLU => res.max(0.0),
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-res).exp()),
            ActivationFn::Tanh => res.tanh(),
        }
    }
}

// Private Functions ----------------------------------------------------------
fn random_sign() -> f32 {
    match rand::thread_rng().gen_bool(POSITIVE_BIAS) {
        true => -1.0,
        false => 1.0,
    }
}

// Macros ----------------------------------------------------------------------
#[macro_export]
macro_rules! compute_simd {
    ($simd_type:ty, $weights:expr, $input:expr) => {{
        let simd_weights: $simd_type = <$simd_type>::from_slice($weights);
        let simd_input: $simd_type = <$simd_type>::from_slice($input);
        (simd_input * simd_weights).reduce_sum()
    }};
}

#[macro_export]
macro_rules! compute_with_simd {
    ($size:expr, $weights:expr, $input:expr) => {{
        match $size {
            16 => compute_simd!(f32x16, $weights, $input),
            8 => compute_simd!(f32x8, $weights, $input),
            4 => compute_simd!(f32x4, $weights, $input),
            2 => compute_simd!(f32x2, $weights, $input),
            1 => $input[0] * $weights[0],
            _ => unreachable!(),
        }
    }};
}

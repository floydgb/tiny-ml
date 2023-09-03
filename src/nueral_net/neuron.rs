// Imports --------------------------------------------------------------------
use {
    super::*,
    crate::dot_simd,
    std::simd::{f32x16, f32x2, f32x4, f32x8, SimdFloat},
};

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
        let random_index = thread_rng().gen_range(0..self.weights.len());
        self.weights[random_index] += random::<f32>() * LEARNING_RATE * random_sign();
    }

    pub fn change_bias(&mut self) {
        self.bias += random::<f32>() * LEARNING_RATE * random_sign();
    }

    pub fn compute(&self, inputs: &[f32]) -> f32 {
        self.activate(dot(&self.weights, inputs) + self.bias)
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
fn dot(x: &[f32], y: &[f32]) -> f32 {
    let (mut i, mut result) = (0, 0.0);
    for simd_size in [16, 8, 4, 2, 1] {
        while i + simd_size <= x.len() {
            let (x_slice, y_slice) = (&y[i..i + simd_size], &x[i..i + simd_size]);
            result += match simd_size {
                16 => dot_simd!(f32x16, x_slice, y_slice),
                8 => dot_simd!(f32x8, x_slice, y_slice),
                4 => dot_simd!(f32x4, x_slice, y_slice),
                2 => dot_simd!(f32x2, x_slice, y_slice),
                1 => x_slice[0] * y_slice[0],
                _ => unreachable!(),
            };
            i += simd_size;
        }
    }
    result
}

fn random_sign() -> f32 {
    match thread_rng().gen_bool(0.5) {
        true => 1.0,
        false => -1.0,
    }
}

// Macros ---------------------------------------------------------------------
#[macro_export]
macro_rules! dot_simd {
    ($simd_type:ty, $x_slice:expr, $y_slice:expr) => {{
        let x: $simd_type = <$simd_type>::from_slice($x_slice);
        let y: $simd_type = <$simd_type>::from_slice($y_slice);
        (x * y).reduce_sum()
    }};
}

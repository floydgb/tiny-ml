// Imports --------------------------------------------------------------------
use crate::{
    nueral_net::{Fn, NeuralNet},
    training::{Trainer, TrainingData},
};

// Public Functions -----------------------------------------------------------
pub fn train_circle() {
    let mut net: NeuralNet<2, 1> = NeuralNet::new()
        .add_layer(3, Fn::ReLU)
        .add_layer(3, Fn::ReLU)
        .add_layer(1, Fn::Linear);
    let trainer = Trainer::new(training_data::<2, 1>());
    let mut epochs = 0;
    while trainer.train(&mut net, 10) > 4000.0 {
        epochs += 1;
    }
    assert!(epochs < 5000);
}

// Private Functions ----------------------------------------------------------
fn training_data<const I: usize, const O: usize>() -> TrainingData<2, 1> {
    let mut data = TrainingData {
        inputs: vec![],
        outputs: vec![],
    };
    for x in 0..=100 {
        for y in 0..=100 {
            data.inputs.push([x as f32, y as f32]);
            data.outputs.push([inside_circle(x, y, 30.0)])
        }
    }
    data
}

fn inside_circle(x: i32, y: i32, radius: f32) -> f32 {
    match (x as f32).powi(2) + (y as f32).powi(2) < radius {
        true => 1.0,
        false => -1.0,
    }
}

// Tests ----------------------------------------------------------------------
#[test]
pub fn test_circle() {
    train_circle();
}

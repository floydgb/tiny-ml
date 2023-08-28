// Imports --------------------------------------------------------------------
use crate::{
    nueral_net::{Fn, NeuralNet},
    training::{Trainer, TrainingData},
};

// Public Functions -----------------------------------------------------------
pub fn train_and_run_trivial() {
    // Create a neural network.
    let mut net = NeuralNet::new().add_layer(1, Fn::Linear);

    // Create training data.
    let mut data = TrainingData {
        inputs: vec![],
        outputs: vec![],
    };
    for i in -50..50 {
        data.inputs.push([i as f32]);
        data.outputs.push([i as f32 * 3.0]);
    }

    // Train the model.
    let trainer = Trainer::new(data);
    for _ in 0..10 {
        trainer.train(&mut net, 10);
        // Lower is better.
        println!("{}", trainer.compute_total_error(&net))
    }

    // Show that this actually works!
    println!("########");
    for i in -5..5 {
        println!("{}", &net.run(&[i as f32 + 0.5])[0]);
    }
}

// Tests ----------------------------------------------------------------------
#[test]
fn test_trivial() {
    train_and_run_trivial();
}

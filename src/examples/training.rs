// Imports --------------------------------------------------------------------
use crate::nueral_net::{trainer::Trainer, ActivationFn, NeuralNet};

// Public Functions -----------------------------------------------------------
pub fn train_and_run_trivial() {
    // Create a neural network.
    let mut trivial_net = NeuralNet::default().add_layer(1, ActivationFn::Linear);

    // Create training data.
    let mut trivial_trainer = Trainer {
        inputs: vec![],
        labels: vec![],
    };
    for i in -50..50 {
        trivial_trainer.inputs.push([i as f32]);
        trivial_trainer.labels.push([i as f32 * 3.0]);
    }

    // Set the error tolerance.
    const TOLERANCE: f32 = 0.1;

    // Train the model.
    let mut err = trivial_trainer.train(&mut trivial_net, 10);
    while err > TOLERANCE {
        err = trivial_trainer.train(&mut trivial_net, 10);
        println!("{}", err)
    }

    // Show that this actually works!
    println!("########");
    for i in -5..5 {
        let value = &trivial_net.run(&[i as f32 + 0.5])[0];
        println!("{}", value);
    }

    // Check that the error is low.
    let correct = vec![-13.5, -10.5, -7.5, -4.5, -1.5, 1.5, 4.5, 7.5, 10.5, 13.5];
    for i in -5..5 {
        let value = &trivial_net.run(&[i as f32 + 0.5])[0];
        assert!((value - correct[(i + 5) as usize]).abs() <= TOLERANCE.powi(2));
    }
}

// Tests ----------------------------------------------------------------------
#[test]
fn test_trivial() {
    train_and_run_trivial();
}

// Imports --------------------------------------------------------------------
use crate::nueral_net::{trainer::Trainer, ActivationFn, NeuralNet};

// Public Functions -----------------------------------------------------------
pub fn train_circle() {
    let mut net: NeuralNet<2, 1> = NeuralNet::default()
        .add_layer(3, ActivationFn::ReLU)
        .add_layer(3, ActivationFn::ReLU)
        .add_layer(1, ActivationFn::Linear);
    let trainer = training_map::<2, 1>();
    let mut epochs = 0;
    const TOLERANCE: f32 = 0.3;
    let mut err = trainer.train(&mut net, 10);
    while err > TOLERANCE {
        err = trainer.train(&mut net, 10);
        epochs += 1;
        println!("{} {}", epochs, err);
    }
    println!("run: (5,5) {:?}", net.run(&[5.0, 5.0]));
    println!("run: (-10,10) {:?}", net.run(&[-10.0, 10.0]));
    println!("run: (10,-10) {:?}", net.run(&[10.0, -10.0]));
    println!("run: (0,0) {:?}", net.run(&[0.0, 0.0]));
    println!("run: (25,1) {:?}", net.run(&[25.0, 1.0]));
    println!("run: (25,25) {:?}", net.run(&[25.0, 25.0]));
    println!("run: (50,50) {:?}", net.run(&[50.0, 50.0]));
    assert!(epochs < 5000);
}

// Private Functions ----------------------------------------------------------
fn training_map<const I: usize, const O: usize>() -> Trainer<2, 1> {
    let mut data = Trainer {
        inputs: vec![],
        labels: vec![],
    };
    for x in 0..=100 {
        for y in 0..=100 {
            data.inputs.push([x as f32, y as f32]);
            data.labels.push([inside_circle(x, y, 30.0)])
        }
    }
    data
}

fn inside_circle(x: i32, y: i32, radius: f32) -> f32 {
    match f32::sqrt((x as f32).powi(2) + (y as f32).powi(2)) < radius {
        true => 0.0,
        false => 1.0,
    }
}

// Tests ----------------------------------------------------------------------
#[test]
pub fn test_circle() {
    train_circle();
}

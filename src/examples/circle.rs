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
    const TOLERANCE: f32 = 0.155;
    let mut err = trainer.train(&mut net, 100);
    while err > TOLERANCE {
        err = trainer.train(&mut net, 100);
        epochs += 1;
        println!("{} {}", epochs, err);
        assert!(epochs < 100);
    }
    println!("In:");
    println!("run: (5,5) {:?}", net.run(&[5.0, 5.0]));
    println!("run: (-10,10) {:?}", net.run(&[-10.0, 10.0]));
    println!("run: (0,0) {:?}", net.run(&[0.0, 0.0]));
    println!("run: (25,1) {:?}", net.run(&[25.0, 1.0]));
    println!("Out:");
    println!("run: (25,25) {:?}", net.run(&[25.0, 25.0]));
    println!("run: (50,-50) {:?}", net.run(&[50.0, -50.0]));
    println!("run: (-75,75) {:?}", net.run(&[-75.0, 75.0]));
    println!("run: (99,99) {:?}", net.run(&[99.0, 99.0]));
    println!("run: (-99,-99) {:?}", net.run(&[-99.0, -99.0]));
    assert!(net.run(&[5.0, 5.0]) < [1.0 + TOLERANCE * 2.0]);
    assert!(net.run(&[-10.0, 10.0]) < [1.0 + TOLERANCE * 2.0]);
    assert!(net.run(&[0.0, 0.0]) < [1.0 + TOLERANCE * 2.0]);
    assert!(net.run(&[25.0, 1.0]) < [1.0 + TOLERANCE * 2.0]);
    assert!(net.run(&[25.0, 25.0]) > [1.0 - TOLERANCE * 2.0]);
    assert!(net.run(&[50.0, -50.0]) > [1.0 - TOLERANCE * 2.0]);
    assert!(net.run(&[-75.0, 75.0]) > [1.0 - TOLERANCE * 2.0]);
    assert!(net.run(&[99.0, 99.0]) > [1.0 - TOLERANCE * 2.0]);
    assert!(net.run(&[-99.0, -99.0]) > [1.0 - TOLERANCE * 2.0]);
}

// Private Functions ----------------------------------------------------------
fn training_map<const I: usize, const O: usize>() -> Trainer<2, 1> {
    let mut data = Trainer {
        inputs: vec![],
        labels: vec![],
    };
    for x in -100..=100 {
        for y in -100..=100 {
            data.inputs.push([x as f32, y as f32]);
            data.labels.push([inside_circle(x, y, 30.0)])
        }
    }
    data
}

fn inside_circle(x: i32, y: i32, radius: f32) -> f32 {
    match (x as f32).powi(2) + (y as f32).powi(2) < radius.powi(2) {
        true => -1.0,
        false => 1.0,
    }
}

// Tests ----------------------------------------------------------------------
#[test]
pub fn test_circle() {
    train_circle();
}

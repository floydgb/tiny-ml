// Imports --------------------------------------------------------------------
use crate::nueral_net::{Fn, NeuralNet};

// Public Functions -----------------------------------------------------------
pub fn speed() {
    let net: NeuralNet<1, 1> = NeuralNet::new()
        .add_layer(5, Fn::ReLU)
        .add_layer(5, Fn::ReLU)
        .add_layer(5, Fn::ReLU)
        .add_layer(5, Fn::ReLU)
        .add_layer(5, Fn::ReLU)
        .add_layer(5, Fn::ReLU)
        .add_layer(1, Fn::Linear);
    let mut sum = 0.0;
    for i in 0..10_000 {
        sum += net.run(&[i as f32])[0];
    }
    assert_eq!(sum, 781171900000.0);
}

// Tests ----------------------------------------------------------------------
#[test]
pub fn test_speed() {
    speed();
}

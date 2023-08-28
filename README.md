# Tiny ML

A simple, fast rust crate for simple basic neural networks.

## What is this for?

- Learning about ML
- Evolution simulations
- Whatever else you want to use this for

## What is this **not**?

- A large scale ML library like Tensorflow or PyTorch. This is simple and basic, or just 'tiny'.

## How to use this?

As an example, here is how to make a model that can tell if a point is in a circle or not!

```rust
    // Create a neural network.
    let mut net = NeuralNet::new().add_layer(1, Fn::Linear);

    // Create training data.
    let mut data = DataSet { inputs: vec![], outputs: vec![] };
    for i in -50..50 {
        data.inputs.push([i as f32]);
        data.outputs.push([i as f32 * 3.0]);
    }

    // Train the model.
    let trainer = BasicTrainer::new(data);
    for _ in 0..10 {
        trainer.train(&mut net, 10);
        // Lower is better.
        println!("{}", trainer.get_total_error(&net))
    }

    // Show that this actually works!
    println!("########");
    for i in -5..5 {
        println!("{}", &net.run(&[i as f32 + 0.5])[0]);
    }
```

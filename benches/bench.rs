#![feature(test)]
// Imports --------------------------------------------------------------------
extern crate test;
use {
    test::Bencher,
    tiny_ml::examples::{circle, speed, training},
};

// Benchmarks -----------------------------------------------------------------
mod bench {
    use super::*;

    #[bench]
    fn bench_trivial(b: &mut Bencher) {
        b.iter(|| training::train_and_run_trivial());
    }

    #[bench]
    fn bench_speed(b: &mut Bencher) {
        b.iter(|| speed::speed());
    }

    #[bench]
    fn bench_circle(b: &mut Bencher) {
        b.iter(|| circle::train_circle());
    }
}

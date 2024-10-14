mod config;
mod dataset;
mod device;
mod model;
mod training;

mod wgpu {
    use burn::backend::{
        wgpu::{Wgpu, WgpuDevice},
        Autodiff,
    };

    use crate::training;

    pub fn run() {
        let device = WgpuDevice::default();
        training::run::<Autodiff<Wgpu>>(device);
    }
}

mod ndarray {
    use crate::training;
    use burn::backend::{
        ndarray::{NdArray, NdArrayDevice},
        Autodiff,
    };

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        training::run::<Autodiff<NdArray>>(device);
    }
}

fn main() {
    // wgpu::run();
    ndarray::run();
}

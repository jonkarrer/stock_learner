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
fn main() {
    wgpu::run();
}

mod dataset;
mod device;
mod inference;
mod model;
mod training;

mod wgpu {
    use burn::{
        backend::{
            wgpu::{Wgpu, WgpuDevice},
            Autodiff,
        },
        data::dataset::Dataset,
    };

    use crate::{
        dataset::{DailyLinearDataset, DailyLinearItem},
        inference, training,
    };

    pub fn train() {
        let device = WgpuDevice::default();
        training::run::<Autodiff<Wgpu>>(device);
    }

    pub fn inference() {
        let device = WgpuDevice::default();
        let data = DailyLinearDataset::test();
        let samples: Vec<DailyLinearItem> = data.iter().take(5).collect();
        inference::infer::<Autodiff<Wgpu>>(device, samples);
    }
}
fn main() {
    // wgpu::train();
    wgpu::inference();
}

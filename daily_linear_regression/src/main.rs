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

    pub fn train(model_path: &str) {
        let device = WgpuDevice::default();
        training::run::<Autodiff<Wgpu>>(device, model_path);
    }

    pub fn inference(model_path: &str) {
        let device = WgpuDevice::default();
        let data = DailyLinearDataset::test();
        let samples: Vec<DailyLinearItem> = data.iter().take(5).collect();
        inference::infer::<Autodiff<Wgpu>>(device, samples, model_path);
    }
}

const MODEL_PATH: &str = "/tmp/burn/daily_linear_regression";
fn main() {
    // wgpu::train(MODEL_PATH);
    wgpu::inference(MODEL_PATH);
}

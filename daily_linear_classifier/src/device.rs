#![allow(dead_code)]

use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};

pub type MyDevice = Wgpu<f32, i32>;
pub type MyBackend = Autodiff<MyDevice>;

pub fn get_device() -> WgpuDevice {
    WgpuDevice::default()
}

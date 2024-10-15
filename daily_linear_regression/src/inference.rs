use std::{fs::File, path::PathBuf};

use burn::{
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};

use crate::{
    dataset::{DailyLinearBatcher, DailyLinearItem},
    model::Model,
};

pub fn infer<B: AutodiffBackend>(
    device: B::Device,
    samples: Vec<DailyLinearItem>,
    artifact_dir: &str,
) {
    // Data
    let batcher = DailyLinearBatcher::<B>::new(device.clone());

    // Load pre-trained model weights
    println!("Loading weights ...");

    let record = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model weights failed to load");

    // Create model using loaded weights
    println!("Creating model ...");
    let model = Model::<B>::new(&device).load_record(record);

    // Run inference on the given text samples
    println!("Running inference ...");
    let item = batcher.batch(samples.clone()); // Batch samples using the batcher
    let predictions = model.infer(item); // Get model predictions

    dbg!(predictions);
}

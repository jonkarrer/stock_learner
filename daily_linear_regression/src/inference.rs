use std::{fs::File, path::PathBuf};

use burn::{
    data::dataloader::batcher::Batcher,
    module::Module,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};

use crate::{
    dataset::{DailyLinearBatch, DailyLinearBatcher, DailyLinearInferBatch, DailyLinearItem},
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
    for sample in samples.clone() {
        dbg!(
            &sample.event_unix_timestamp,
            &sample.next_period_price - sample.close_price
        );
    }
    let item: DailyLinearInferBatch<B> = batcher.batch(samples.clone()); // Batch samples using the batcher

    // dbg!(&item.inputs.to_data().to_vec::<f32>().unwrap());
    let predictions = model.infer(item); // Get model predictions

    dbg!(predictions);
}

pub fn dry_run<B: AutodiffBackend>(device: B::Device, samples: Vec<DailyLinearItem>) {
    // Data
    let batcher = DailyLinearBatcher::<B>::new(device.clone());
    // Create model using loaded weights
    println!("Creating model ...");
    let model = Model::<B>::new(&device);

    // Run inference on the given text samples
    println!("Running inference ...");
    dbg!(&samples);
    // let item: DailyLinearInferBatch<B> = batcher.batch(samples.clone()); // Batch samples using the batcher
    let item: DailyLinearBatch<B> = batcher.batch(samples.clone());

    // dbg!(&item.inputs.to_data().to_vec::<f32>().unwrap());
    // dbg!(&item.targets.to_data().to_vec::<f32>().unwrap());

    // dbg!(&item.inputs.to_data().to_vec::<f32>().unwrap());
    // let predictions = model.infer(item); // Get model predictions

    // dbg!(predictions);
}

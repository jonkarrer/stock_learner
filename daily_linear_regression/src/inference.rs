use burn::{
    config::Config,
    data::dataloader::{batcher::Batcher, DataLoaderBuilder},
    grad_clipping::GradientClippingConfig,
    module::Module,
    optim::{decay::WeightDecayConfig, AdamConfig},
    record::{CompactRecorder, NoStdTrainingRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{
            store::{Aggregate, Direction, Split},
            AccuracyMetric, CpuMemory, CpuUse, LossMetric,
        },
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

use crate::{
    dataset::{DailyLinearBatcher, DailyLinearDataset, DailyLinearItem},
    model::Model,
};

static ARTIFACTS_DIR: &str = "/tmp/burn/daily_linear_classifier";

#[derive(Config)]
pub struct DailyLinearTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 1000)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn infer<B: AutodiffBackend>(device: B::Device, samples: Vec<DailyLinearItem>) {
    create_artifact_dir(ARTIFACTS_DIR);

    // Config
    let optimizer_config = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(5e-5)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));
    let config = DailyLinearTrainingConfig::new(optimizer_config);
    B::seed(config.seed);

    // Data
    let batcher = DailyLinearBatcher::<B>::new(device.clone());

    // Load pre-trained model weights
    println!("Loading weights ...");
    let record = CompactRecorder::new()
        .load(format!("{ARTIFACTS_DIR}/model_01").into(), &device)
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

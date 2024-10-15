use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    grad_clipping::GradientClippingConfig,
    module::Module,
    optim::{decay::WeightDecayConfig, AdamConfig},
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{
            store::{Aggregate, Direction, Split},
            CpuMemory, CpuUse, LossMetric,
        },
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

use crate::{
    dataset::{DailyLinearBatcher, DailyLinearDataset},
    model::{self, Model},
};

#[derive(Config)]
pub struct DailyLinearTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 5000)]
    pub batch_size: usize,

    #[config(default = 4)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,
}

pub fn run<B: AutodiffBackend>(device: B::Device, artifact_dir: &str) {
    // Config
    let optimizer_config = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(5e-5)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)));
    let config = DailyLinearTrainingConfig::new(optimizer_config);
    B::seed(config.seed);

    // Data
    let batcher_train = DailyLinearBatcher::<B>::new(device.clone());
    let batcher_valid = DailyLinearBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(DailyLinearDataset::train());
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(DailyLinearDataset::valid());

    // Model
    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 2 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(Model::<B>::new(&device), config.optimizer.init(), 4e-3);

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    config
        .save(format!("{artifact_dir}/config.json").as_str())
        .expect("Failed to save config");

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

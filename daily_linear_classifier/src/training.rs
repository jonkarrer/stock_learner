use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::{decay::WeightDecayConfig, AdamConfig},
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{
            store::{Aggregate, Direction, Split},
            AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, LossMetric,
        },
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

use crate::{
    dataset::{DailyLinearBatcher, DailyLinearDataset},
    model::Model,
};

static ARTIFACTS_DIR: &str = "/tmp/burn/daily_linear_classifier";

#[derive(Config)]
pub struct DailyLinearTrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 64)]
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

pub fn run<B: AutodiffBackend>(device: B::Device) {
    create_artifact_dir(ARTIFACTS_DIR);

    // Config
    let optimizer_config = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
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
    let learner = LearnerBuilder::new(ARTIFACTS_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 1 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(Model::<B>::new(&device), config.optimizer.init(), 1e-4);

    let model_trained = learner.fit(dataloader_train, dataloader_valid);

    config
        .save(format!("{ARTIFACTS_DIR}/config.json").as_str())
        .expect("Failed to save config");

    model_trained
        .save_file(
            format!("{ARTIFACTS_DIR}/model_01"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save model");
}

use burn::{
    module::Module,
    nn::{loss::BinaryCrossEntropyLossConfig, Linear, LinearConfig, Relu, Sigmoid, Tanh},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::dataset::DailyLinearBatch;

const INPUT_SIZE: usize = 25;
const HIDDEN_SIZE: usize = 64;
const OUTPUT_SIZE: usize = 2;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input_layer: Linear<B>,
    output_layer: Linear<B>,
    activation: Relu,
}

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        let device = B::Device::default();
        Self::new(&device)
    }
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let input_layer = LinearConfig::new(INPUT_SIZE, HIDDEN_SIZE)
            .with_bias(true)
            .init(device);
        let output_layer = LinearConfig::new(HIDDEN_SIZE, OUTPUT_SIZE)
            .with_bias(true)
            .init(device);

        let activation = Relu::new();

        Self {
            input_layer,
            output_layer,
            activation,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.detach();
        let x = self.input_layer.forward(x);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }

    pub fn forward_step(&self, item: DailyLinearBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.inputs);
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&output.device())
            .forward(output.clone(), targets.clone().unsqueeze_dim(1)); // bce loss requires targets to be of shape

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<DailyLinearBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: DailyLinearBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: AutodiffBackend> ValidStep<DailyLinearBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: DailyLinearBatch<B>) -> ClassificationOutput<B> {
        self.forward_step(item)
    }
}
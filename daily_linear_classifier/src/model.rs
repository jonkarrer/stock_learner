use burn::{
    module::Module,
    nn::{loss::CrossEntropyLossConfig, Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::{activation::softmax, backend::AutodiffBackend, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::dataset::DailyLinearBatch;

const INPUT_SIZE: usize = 44;
const HIDDEN_SIZE: usize = 512;
const OUTPUT_SIZE: usize = 2;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input_layer: Linear<B>,
    ln1: Linear<B>,
    output_layer: Linear<B>,
    dropout: Dropout,
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

        let ln1 = LinearConfig::new(HIDDEN_SIZE, HIDDEN_SIZE)
            .with_bias(true)
            .init(device);

        let dropout = DropoutConfig::new(0.5).init();
        let output_layer = LinearConfig::new(HIDDEN_SIZE, OUTPUT_SIZE)
            .with_bias(true)
            .init(device);

        let activation = Relu::new();

        Self {
            input_layer,
            ln1,
            output_layer,
            dropout,
            activation,
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.detach();
        let x = self.input_layer.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.ln1.forward(x);
        let x = self.activation.forward(x);

        self.output_layer.forward(x)
    }

    pub fn forward_step(&self, item: DailyLinearBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.inputs);

        let loss = CrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&output.device())
            .forward(output.clone(), targets.clone()); // bce loss requires targets to be of shape

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

impl<B: Backend> ValidStep<DailyLinearBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: DailyLinearBatch<B>) -> ClassificationOutput<B> {
        self.forward_step(item)
    }
}

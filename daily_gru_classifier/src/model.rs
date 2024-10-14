use burn::{
    module::Module,
    nn::{
        gru::{Gru, GruConfig},
        loss::CrossEntropyLossConfig,
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
    },
    prelude::Backend,
    tensor::{backend::AutodiffBackend, RangesArg, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::{
    config::{get_config, Config},
    dataset::DailyLinearBatch,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input_layer: Gru<B>,
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
        let Config {
            input_size,
            hidden_size,
            output_size,
            ..
        } = get_config();
        let input_layer = GruConfig::new(input_size, hidden_size, true).init(device);

        let ln1 = LinearConfig::new(hidden_size, hidden_size)
            .with_bias(true)
            .init(device);

        let dropout = DropoutConfig::new(0.5).init();
        let output_layer = LinearConfig::new(hidden_size, output_size)
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

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = input.detach();
        let x = self.input_layer.forward(x, None);

        let x: Tensor<B, 3> = x.slice([None, Some((-2, -1)), None]);

        let x = self.ln1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);
        self.output_layer.forward(x.squeeze(1))
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

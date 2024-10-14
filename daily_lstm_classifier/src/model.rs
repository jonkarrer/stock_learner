use burn::{
    module::Module,
    nn::{
        gru::{Gru, GruConfig},
        loss::CrossEntropyLossConfig,
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Lstm, LstmConfig,
        Relu,
    },
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::{
    config::{get_config, ModelConfig},
    dataset::DailyLinearBatch,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input_layer: Lstm<B>,
    ln1: Linear<B>,
    ln2: Linear<B>,
    output_layer: Linear<B>,
    dropout: Dropout,
    activation: Relu,
    layer_norm: LayerNorm<B>,
}

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        let device = B::Device::default();
        Self::new(&device)
    }
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let ModelConfig {
            input_size,
            hidden_size,
            output_size,
            dropout,
            ..
        } = get_config();
        let input_layer = LstmConfig::new(input_size, hidden_size, true).init(device);
        let layer_norm = LayerNormConfig::new(hidden_size).init(device);

        let ln1 = LinearConfig::new(hidden_size, hidden_size)
            .with_bias(true)
            .init(device);

        let ln2 = LinearConfig::new(hidden_size, hidden_size)
            .with_bias(true)
            .init(device);

        let dropout = DropoutConfig::new(dropout as f64).init();
        let output_layer = LinearConfig::new(hidden_size, output_size)
            .with_bias(true)
            .init(device);

        let activation = Relu::new();

        Self {
            input_layer,
            ln1,
            ln2,
            output_layer,
            dropout,
            activation,
            layer_norm,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = input.detach();
        let x = self.input_layer.forward(x, None);
        let x = self.layer_norm.forward(x.0);

        let x = self.ln1.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        let x = self.ln2.forward(x);
        let x = self.activation.forward(x);
        let x = self.dropout.forward(x);

        // Reshape from [batch_size, sequence_length, hidden_size] to [batch_size, hidden_size]
        // ... this is because we only need the last output of the sequence as for the classification.
        let x: Tensor<B, 3> = x.slice([None, Some((-2, -1)), None]);
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

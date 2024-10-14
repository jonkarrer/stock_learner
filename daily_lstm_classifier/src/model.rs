use burn::{
    module::Module,
    nn::{
        loss::CrossEntropyLossConfig, Dropout, DropoutConfig, Linear, LinearConfig, Lstm,
        LstmConfig, Relu,
    },
    prelude::Backend,
    tensor::{backend::AutodiffBackend, Tensor},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::dataset::DailyLinearBatch;

const INPUT_SIZE: usize = 45;
const HIDDEN_SIZE: usize = 512;
const SEQUENCE_LENGTH: usize = 9;
const OUTPUT_SIZE: usize = 2;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input_layer: Lstm<B>,
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
        let input_layer = LstmConfig::new(INPUT_SIZE, HIDDEN_SIZE, true).init(device);

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

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let x = input.detach();
        let (lstm_out, _) = self.input_layer.forward(x, None);
        let x = self.activation.forward(
            lstm_out
                .clone()
                .slice([lstm_out.dims()[0] - 1..lstm_out.dims()[0]]),
        );
        // let x = self.dropout.forward(x);

        let x = self.ln1.forward(x);
        let x = self.activation.forward(x);

        self.output_layer.forward(x.squeeze(0))
    }

    pub fn forward_step(&self, item: DailyLinearBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let inputs = item.inputs;
        let inputs = inputs.reshape([HIDDEN_SIZE, SEQUENCE_LENGTH, INPUT_SIZE]);

        let output = self.forward(inputs).squeeze(0);

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

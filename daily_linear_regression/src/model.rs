use burn::{
    module::Module,
    nn::{loss::MseLoss, Dropout, DropoutConfig, Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::{backend::AutodiffBackend, DataError, Tensor},
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::dataset::{DailyLinearBatch, DailyLinearInferBatch};

const INPUT_SIZE: usize = 48;
const HIDDEN_SIZES: [usize; 6] = [64, 128, 256, 512, 1024, 2048];
const OUTPUT_SIZE: usize = 1;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input_layer: Linear<B>,
    // ln1: Linear<B>,
    // ln2: Linear<B>,
    // ln3: Linear<B>,
    // ln4: Linear<B>,
    // ln5: Linear<B>,
    // ln6: Linear<B>,
    // ln7: Linear<B>,
    // ln8: Linear<B>,
    // ln9: Linear<B>,
    // ln10: Linear<B>,
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
        let h1 = HIDDEN_SIZES[0];
        let h2 = HIDDEN_SIZES[1];
        let h3 = HIDDEN_SIZES[2];
        let h4 = HIDDEN_SIZES[3];
        let h5 = HIDDEN_SIZES[4];
        let h6 = HIDDEN_SIZES[5];
        let input_layer = LinearConfig::new(INPUT_SIZE, h1)
            .with_bias(true)
            .init(device);

        // let ln1 = LinearConfig::new(h1, h1).with_bias(true).init(device);

        // let ln2 = LinearConfig::new(h1, h2).with_bias(true).init(device);

        // let ln3 = LinearConfig::new(h2, h2).with_bias(true).init(device);

        // let ln4 = LinearConfig::new(h2, h3).with_bias(true).init(device);

        // let ln5 = LinearConfig::new(h3, h3).with_bias(true).init(device);

        // let ln6 = LinearConfig::new(h3, h4).with_bias(true).init(device);

        // let ln7 = LinearConfig::new(h4, h4).with_bias(true).init(device);

        // let ln8 = LinearConfig::new(h4, h5).with_bias(true).init(device);

        // let ln9 = LinearConfig::new(h5, h5).with_bias(true).init(device);

        // let ln10 = LinearConfig::new(h5, h6).with_bias(true).init(device);

        let output_layer = LinearConfig::new(h1, OUTPUT_SIZE)
            .with_bias(true)
            .init(device);

        let dropout = DropoutConfig::new(0.5).init();
        let activation = Relu::new();

        Self {
            input_layer,
            // ln1,
            // ln2,
            // ln3,
            // ln4,
            // ln5,
            // ln6,
            // ln7,
            // ln8,
            // ln9,
            // ln10,
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

        self.output_layer.forward(x)
    }

    pub fn forward_step(&self, item: DailyLinearBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item.targets.unsqueeze();
        let output: Tensor<B, 2> = self.forward(item.inputs);

        let loss = MseLoss::new().forward(
            output.clone(),
            targets.clone(),
            burn::nn::loss::Reduction::Mean,
        ); // bce loss requires targets to be of shape

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }

    pub fn infer(&self, item: DailyLinearInferBatch<B>) -> Result<Vec<f32>, DataError> {
        let output = self.forward(item.inputs);
        dbg!(output.to_data());
        output.to_data().to_vec()
    }
}

impl<B: AutodiffBackend> TrainStep<DailyLinearBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: DailyLinearBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<DailyLinearBatch<B>, RegressionOutput<B>> for Model<B> {
    fn step(&self, item: DailyLinearBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}

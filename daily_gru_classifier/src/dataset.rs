use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, SqliteDataset},
    },
    prelude::Backend,
    tensor::{Int, Tensor},
};
use serde::{Deserialize, Serialize};

use crate::config::get_config;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DailyLinearItem {
    pub row_id: i32,
    pub open_price: f32,
    pub close_price: f32,
    pub high_price: f32,
    pub low_price: f32,
    pub volume: f32,
    pub volume_weighted_price: f32,
    pub bar_trend: i32,
    pub label: i32,
    pub hundred_day_sma: f32,
    pub hundred_day_ema: f32,
    pub fifty_day_sma: f32,
    pub fifty_day_ema: f32,
    pub twenty_day_sma: f32,
    pub twenty_day_ema: f32,
    pub nine_day_sma: f32,
    pub nine_day_ema: f32,
    pub hundred_day_high: f32,
    pub hundred_day_low: f32,
    pub fifty_day_high: f32,
    pub fifty_day_low: f32,
    pub ten_day_high: f32,
    pub ten_day_low: f32,
    pub fourteen_day_rsi: f32,
    pub top_bollinger_band: f32,
    pub middle_bollinger_band: f32,
    pub bottom_bollinger_band: f32,
    pub macd_signal: f32,
    pub previous_period_trend: i32,
    pub previous_five_day_trend: i32,
    pub previous_ten_day_trend: i32,
    pub future_three_day_trend: i32,
    pub future_five_day_trend: i32,
    pub future_ten_day_trend: i32,
    pub distance_to_hundred_day_sma: f32,
    pub distance_to_hundred_day_ema: f32,
    pub distance_to_fifty_day_sma: f32,
    pub distance_to_fifty_day_ema: f32,
    pub distance_to_twenty_day_sma: f32,
    pub distance_to_twenty_day_ema: f32,
    pub distance_to_nine_day_ema: f32,
    pub distance_to_nine_day_sma: f32,
    pub distance_to_hundred_day_high: f32,
    pub distance_to_hundred_day_low: f32,
    pub distance_to_fifty_day_high: f32,
    pub distance_to_fifty_day_low: f32,
    pub distance_to_ten_day_high: f32,
    pub distance_to_ten_day_low: f32,
    pub distance_to_top_bollinger_band: f32,
    pub distance_to_middle_bollinger_band: f32,
    pub distance_to_bottom_bollinger_band: f32,
}

pub struct DailyLinearDataset {
    dataset: SqliteDataset<DailyLinearItem>,
}

impl Dataset<DailyLinearItem> for DailyLinearDataset {
    fn get(&self, index: usize) -> Option<DailyLinearItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl DailyLinearDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn valid() -> Self {
        Self::new("valid")
    }

    pub fn new(split_type: &str) -> Self {
        let split = match split_type {
            "train" => "daily_training_set",
            "valid" => "daily_validation_set",
            _ => panic!("Invalid split type"),
        };

        let dataset = SqliteDataset::from_db_file(
            "/Volumes/karrer_ssd/datastores/sqlite/market_data/stocks.db",
            &split,
        )
        .expect("Failed to load dataset");

        Self { dataset }
    }
}

#[derive(Clone, Debug)]
pub struct DailyLinearBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct DailyLinearBatch<B: Backend> {
    pub inputs: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> DailyLinearBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn min_max_norm_inputs(&self, inp: &Tensor<B, 3>) -> Tensor<B, 3> {
        let min = inp.clone().min_dim(0);
        let max = inp.clone().max_dim(0);

        let denominator = (max.clone() - min.clone()).clamp(1e-8, f32::MAX);
        let normalized = (inp.clone() - min.clone()) / denominator;

        normalized * 2.0 - 1.0
    }
}

impl<B: Backend> Batcher<DailyLinearItem, DailyLinearBatch<B>> for DailyLinearBatcher<B> {
    fn batch(&self, items: Vec<DailyLinearItem>) -> DailyLinearBatch<B> {
        let config = get_config();
        let mut inputs: Vec<Tensor<B, 3>> = Vec::new();
        let mut sequence: Vec<Tensor<B, 2>> = Vec::new();
        let mut start_seq_len = config.sequence_length;

        for (i, item) in items.iter().enumerate() {
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    item.open_price,
                    item.close_price,
                    item.high_price,
                    item.low_price,
                    item.volume,
                    item.volume_weighted_price,
                    item.bar_trend as f32,
                    item.hundred_day_sma,
                    item.hundred_day_ema,
                    item.fifty_day_sma,
                    item.fifty_day_ema,
                    item.twenty_day_sma,
                    item.twenty_day_ema,
                    item.nine_day_sma,
                    item.nine_day_ema,
                    item.hundred_day_high,
                    item.hundred_day_low,
                    item.fifty_day_high,
                    item.fifty_day_low,
                    item.ten_day_high,
                    item.ten_day_low,
                    item.fourteen_day_rsi,
                    item.top_bollinger_band,
                    item.middle_bollinger_band,
                    item.bottom_bollinger_band,
                    item.macd_signal,
                    item.previous_period_trend as f32,
                    item.previous_five_day_trend as f32,
                    item.previous_ten_day_trend as f32,
                    item.distance_to_hundred_day_sma,
                    item.distance_to_hundred_day_ema,
                    item.distance_to_fifty_day_sma,
                    item.distance_to_fifty_day_ema,
                    item.distance_to_twenty_day_sma,
                    item.distance_to_twenty_day_ema,
                    item.distance_to_nine_day_ema,
                    item.distance_to_nine_day_sma,
                    item.distance_to_hundred_day_high,
                    item.distance_to_hundred_day_low,
                    item.distance_to_fifty_day_high,
                    item.distance_to_fifty_day_low,
                    item.distance_to_ten_day_high,
                    item.distance_to_ten_day_low,
                    item.distance_to_top_bollinger_band,
                    item.distance_to_middle_bollinger_band,
                    item.distance_to_bottom_bollinger_band,
                ],
                &self.device,
            );

            // make the tensor 2D for easy concat later
            // sequence = [
            //     Tensor([[10.5, 100.0, 50.2]]),
            //     Tensor([[20.1, 95.5, 60.0]]),
            //     Tensor([[15.3, 105.2, 55.7]]),
            // ];
            sequence.push(input_tensor.unsqueeze());

            if i == start_seq_len - 1 {
                // concat the tensors, now the shape is (sequence_length, feature length)
                // inputs = Tensor([
                //     [10.5, 100.0, 50.2],
                //     [20.1, 95.5, 60.0],
                //     [15.3, 105.2, 55.7]
                // ], ...)
                inputs.push(Tensor::cat(sequence, 0).unsqueeze::<3>());
                sequence = Vec::new();
                start_seq_len += config.sequence_length;
            }
        }

        // concat the tensors, now the shape is (batch_size, sequence_length, feature length)
        // inputs = Tensor([
        //     [10.5, 100.0, 50.2],
        //     [20.1, 95.5, 60.0],
        //     [15.3, 105.2, 55.7]
        // ], ...)
        let inputs = Tensor::cat(inputs, 0);

        // normalize the inputs so that they fit between 0 and 1
        let inputs = self.min_max_norm_inputs(&inputs);

        // create target tenser
        // targets = [
        //     Tensor([1.0]),
        //     Tensor([0.0]),
        //     Tensor([1.0]),
        //     Tensor([1.0]),
        //     Tensor([0.0])
        // ]
        let mut targets = Vec::new();

        for (i, item) in items.iter().enumerate() {
            if i != 0 && i % config.sequence_length == 0 {
                targets.push(Tensor::<B, 1, Int>::from_ints([item.label], &self.device));
            }
        }

        // do not need to unsqueeze here, just concat for a 1D tensor
        // targets = Tensor([1.0, 0.0, 1.0, 1.0, 0.0])
        // dbg!(targets.len());
        let targets = Tensor::cat(targets, 0);

        DailyLinearBatch { inputs, targets }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::{get_device, MyBackend};

    #[test]
    fn test_dataset() {
        let train = DailyLinearDataset::train();
        let valid = DailyLinearDataset::valid();

        assert_eq!(train.dataset.len(), 759839);
        assert_eq!(valid.dataset.len(), 219260);
    }

    #[test]
    fn test_batcher() {
        let device = get_device();
        let train = DailyLinearDataset::train();
        let batcher: DailyLinearBatcher<MyBackend> = DailyLinearBatcher::new(device);

        let items: Vec<DailyLinearItem> = train.dataset.iter().take(200).collect();
        let items = items[100..149].to_vec();

        let batch = batcher.batch(items);

        dbg!(batch.inputs.shape());
        dbg!(batch.targets.shape());
        assert!(false)
    }
}

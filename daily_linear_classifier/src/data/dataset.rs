use burn::data::dataset::{Dataset, SqliteDataset};
use serde::{Deserialize, Serialize};

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
}

pub struct DailyLinearDataset {
    dataset: SqliteDataset<DailyLinearItem>,
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
            "train" => "daily_linear_classifier_train",
            "valid" => "daily_linear_classifier_validation",
            _ => panic!("Invalid split type"),
        };

        dbg!(split);

        let dataset = SqliteDataset::from_db_file(
            "/Volumes/karrer_ssd/datastores/sqlite/market_data/stocks.db",
            &split,
        )
        .expect("Failed to load dataset");

        Self { dataset }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset() {
        let train = DailyLinearDataset::train();
        // let valid = DailyLinearDataset::valid();

        dbg!(train.dataset.len());
        assert!(false);
    }
}

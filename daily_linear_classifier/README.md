# Daily Linear Classifier

I wanted to get a baseline started for a machine learning model that would predict if a stock would go up or down the next period. The period in this case is daily, so if the stock will go up the next day. Building a baseline model is the first step in all my projects. There is magic in the 80 / 20 approach to everything. Here I start with a 2 layer linear neural network that tries to classify just one row of data as buy or sell.

## Data

A single row of data has the candle for that day, and some technical indicators. Here is a raw example of a row:

| Column Name | Value |
|-------------|-------|
| id | 200 |
| event_datetime | 2016-10-17 04:00:00 |
| event_unix_timestamp | 1476676800000 |
| open_price | 17.7999992370605 |
| close_price | 17.7700004577637 |
| high_price | 18.2000007629395 |
| low_price | 17.7049999237061 |
| volume | 4385696.0 |
| volume_weighted_price | 17.8098182678223 |
| stock_symbol | JBLU |
| timeframe | 1D |
| bar_trend | bearish |
| buy_or_sell | 1 |
| next_frame_price | 17.7800006866455 |
| next_frame_trend | bearish |
| next_frame_unix_timestamp | 1476763200000 |
| next_frame_event_datetime | 2016-10-18 04:00:00 |
| hundred_day_sma | 17.1679515838623 |
| hundred_day_ema | 17.1679515838623 |
| fifty_day_sma | 16.9481010437012 |
| fifty_day_ema | 16.9481010437012 |
| twenty_day_sma | 17.5162487030029 |
| twenty_day_ema | 17.5162487030029 |
| nine_day_ema | 17.8033351898193 |
| nine_day_sma | 17.8033351898193 |
| hundred_day_high | 18.9400005340576 |
| hundred_day_low | 14.7600002288818 |
| fifty_day_high | 18.4699993133545 |
| fifty_day_low | 15.6999998092651 |
| ten_day_high | 18.4699993133545 |
| ten_day_low | 17.1499996185303 |
| fourteen_day_rsi | 57.9687461853027 |
| top_bollinger_band | 18.182430267334 |
| middle_bollinger_band | 17.5162487030029 |
| bottom_bollinger_band | 16.8500671386719 |

### Data processing

The framework I use is called [Burn](https://burn.dev/). It provides a number of utilities that help create nueral nets and train them. They have a sqlite database utility that I use for this model. Their opinions on table format are [linked here](https://burn.dev/docs/burn/data/dataset/struct.SqliteDataset.html).

Along with formatting the data for the framework, I need to split it into training and validation. There are 1,000,000 rows in the dataset. I split it into 80% training and 20% validation. Also, since this is time series data, I want the validation to be after the training data. Here is an example of one of those tables after the split:

| Column Name | Value |
|-------------|-------|
| row_id | 1 |
| open_price | 17.7999992370605 |
| close_price | 17.7700004577637 |
| high_price | 18.2000007629395 |
| low_price | 17.7049999237061 |
| volume | 4385696.0 |
| volume_weighted_price | 17.8098182678223 |
| bar_trend | 1 |
| buy_or_sell | 1 |
| hundred_day_sma | 17.1679515838623 |
| hundred_day_ema | 17.1679515838623 |
| fifty_day_sma | 16.9481010437012 |
| fifty_day_ema | 16.9481010437012 |
| twenty_day_sma | 17.5162487030029 |
| twenty_day_ema | 17.5162487030029 |
| nine_day_ema | 17.8033351898193 |
| nine_day_sma | 17.8033351898193 |
| hundred_day_high | 18.9400005340576 |
| hundred_day_low | 14.7600002288818 |
| fifty_day_high | 18.4699993133545 |
| fifty_day_low | 15.6999998092651 |
| ten_day_high | 18.4699993133545 |
| ten_day_low | 17.1499996185303 |
| fourteen_day_rsi | 57.9687461853027 |
| top_bollinger_band | 18.182430267334 |
| middle_bollinger_band | 17.5162487030029 |
| bottom_bollinger_band | 16.8500671386719 |

## Training

Linear classifiers are usually the starting point for classification tasks on tabular data. I will walk through my experiment configs on each run and see how they perform. Hopefully adjusting hpyerparameters will improve performance with each run, but that's why it's called experimenting. First though, I will start with the most simple one that I can think of.

### Run 1

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-4 |
| batch_size | 64 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 25 |
| hidden_layers | 1 |
| hidden_layer_size | 64 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |

- Results

| Epoch | Loss | Accuracy |
|-------|------|----------|
| 0 | 51.0 | 50.0 |
| 1 | 52.0 | 51.0 |
| 2 | 52.0 | 51.0 |

Early stop... no improvement.

### Run 2

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 25 |
| hidden_layers | 1 |
| hidden_layer_size | 128 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |

- Results

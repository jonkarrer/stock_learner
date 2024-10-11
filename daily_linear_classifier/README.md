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
| weight_decay | 5e-5 |
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
| bias | true |

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
| learning_rate | 1e-3 |
| weight_decay | 5e-5 |
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
| bias | true |

- Results

Model {
  input_layer: Linear {d_input: 25, d_output: 128, bias: true, params: 3328}
  output_layer: Linear {d_input: 128, d_output: 2, bias: true, params: 258}
  activation: Relu
  params: 3586
}
Total Epochs: 5

| Split | Metric          | Min.     | Epoch    | Max.     | Epoch    |
|-------|-----------------|----------|----------|----------|----------|
| Train | CPU Usage       | 52.270   | 3        | 56.542   | 1        |
| Train | CPU Memory      | 19.332   | 5        | 19.482   | 2        |
| Train | Loss            | 0.692    | 5        | 0.692    | 1        |
| Train | Accuracy        | 51.839   | 1        | 52.129   | 5        |
| Valid | CPU Usage       | 50.215   | 4        | 55.582   | 1        |
| Valid | CPU Memory      | 18.967   | 5        | 19.404   | 2        |
| Valid | Loss            | 0.693    | 3        | 0.695    | 2        |
| Valid | Accuracy        | 51.061   | 1        | 51.369   | 5        |

### Run 3

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-5 |
| weight_decay | 5e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 25 |
| hidden_layers | 1 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |

- Results

Model {
  input_layer: Linear {d_input: 25, d_output: 256, bias: true, params: 6656}
  output_layer: Linear {d_input: 256, d_output: 2, bias: true, params: 514}
  activation: Relu
  params: 7170
}
Total Epochs: 3

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | Loss       | 0.692    | 3        | 0.692    | 1        |
| Train | CPU Memory | 19.209   | 3        | 19.498   | 1        |
| Train | Accuracy   | 51.713   | 1        | 51.851   | 3        |
| Train | CPU Usage  | 53.681   | 1        | 54.714   | 3        |
| Valid | Loss       | 0.693    | 1        | 0.693    | 2        |
| Valid | CPU Memory | 19.228   | 2        | 19.481   | 1        |
| Valid | Accuracy   | 51.387   | 3        | 51.414   | 1        |
| Valid | CPU Usage  | 52.290   | 1        | 53.637   | 3        |

### Run 4

- Notes
  
tried taking the log of the volume column and removing the min max norm. Spoiler alert, fail.

- Config

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-5 |
| weight_decay | 5e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 25 |
| hidden_layers | 1 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |

- Results

Model {
  input_layer: Linear {d_input: 25, d_output: 256, bias: true, params: 6656}
  output_layer: Linear {d_input: 256, d_output: 2, bias: true, params: 514}
  activation: Relu
  params: 7170
}
Total Epochs: 3

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | CPU Usage  | 54.374   | 3        | 55.855   | 1        |
| Train | Loss       | NaN      | 1        | NaN      | 3        |
| Train | CPU Memory | 19.632   | 1        | 19.952   | 3        |
| Train | Accuracy   | 48.180   | 2        | 48.184   | 1        |
| Valid | CPU Usage  | 50.668   | 3        | 53.622   | 1        |
| Valid | Loss       | NaN      | 1        | NaN      | 3        |
| Valid | CPU Memory | 19.915   | 2        | 19.958   | 3        |
| Valid | Accuracy   | 48.584   | 1        | 48.584   | 3        |

### Run 5

- Notes
  
tried taking the log of the volume column and removing the min max norm. Spoiler alert, fail.

- Config

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-5 |
| weight_decay | 5e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 25 |
| hidden_layers | 1 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |

- Results

Model {
  input_layer: Linear {d_input: 25, d_output: 256, bias: true, params: 6656}
  output_layer: Linear {d_input: 256, d_output: 2, bias: true, params: 514}
  activation: Relu
  params: 7170
}
Total Epochs: 3

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | CPU Usage  | 54.374   | 3        | 55.855   | 1        |
| Train | Loss       | NaN      | 1        | NaN      | 3        |
| Train | CPU Memory | 19.632   | 1        | 19.952   | 3        |
| Train | Accuracy   | 48.180   | 2        | 48.184   | 1        |
| Valid | CPU Usage  | 50.668   | 3        | 53.622   | 1        |
| Valid | Loss       | NaN      | 1        | NaN      | 3        |
| Valid | CPU Memory | 19.915   | 2        | 19.958   | 3        |
| Valid | Accuracy   | 48.584   | 1        | 48.584   | 3        |

### Run 6

- Notes

Going to add a dropout layer.

- Config

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-5 |
| weight_decay | 5e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 25 |
| hidden_layers | 1 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |
| dropout | 0.5 |

- Results

Model {
  input_layer: Linear {d_input: 25, d_output: 256, bias: true, params: 6656}
  output_layer: Linear {d_input: 256, d_output: 2, bias: true, params: 514}
  dropout: Dropout {prob: 0.5}
  activation: Relu
  params: 7170
}
Total Epochs: 3

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | Accuracy   | 50.903   | 1        | 51.391   | 3        |
| Train | Loss       | 0.693    | 3        | 0.694    | 1        |
| Train | CPU Memory | 19.722   | 1        | 19.949   | 2        |
| Train | CPU Usage  | 55.119   | 1        | 56.131   | 2        |
| Valid | Accuracy   | 51.397   | 3        | 51.417   | 1        |
| Valid | Loss       | 0.693    | 1        | 0.693    | 2        |
| Valid | CPU Memory | 19.340   | 3        | 20.021   | 2        |
| Valid | CPU Usage  | 52.641   | 2        | 54.060   | 3        |

### Run 7

- Notes

Added 2 more hidden layers.

- Config
| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-5 |
| weight_decay | 5e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 25 |
| hidden_layers | 3 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |
| dropout | 0.5 |

- Results

Model {
  input_layer: Linear {d_input: 25, d_output: 256, bias: true, params: 6656}
  ln1: Linear {d_input: 256, d_output: 256, bias: true, params: 65792}
  ln2: Linear {d_input: 256, d_output: 256, bias: true, params: 65792}
  output_layer: Linear {d_input: 256, d_output: 2, bias: true, params: 514}
  dropout: Dropout {prob: 0.5}
  activation: Relu
  params: 138754
}
Total Epochs: 3

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | Accuracy   | 50.975   | 1        | 51.648   | 3        |
| Train | CPU Usage  | 51.280   | 2        | 51.635   | 1        |
| Train | Loss       | 0.693    | 3        | 0.693    | 1        |
| Train | CPU Memory | 19.638   | 3        | 19.773   | 2        |
| Valid | Accuracy   | 51.416   | 1        | 51.416   | 3        |
| Valid | CPU Usage  | 48.600   | 3        | 49.028   | 1        |
| Valid | Loss       | 0.693    | 1        | 0.693    | 3        |
| Valid | CPU Memory | 19.627   | 2        | 19.733   | 1        |

### Run 8

- Notes

Taking out the bias.

- Config
| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-5 |
| weight_decay | 5e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 25 |
| hidden_layers | 3 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | false |

- Results
Model {
  input_layer: Linear {d_input: 25, d_output: 256, bias: true, params: 6656}
  ln1: Linear {d_input: 256, d_output: 256, bias: true, params: 65792}
  ln2: Linear {d_input: 256, d_output: 256, bias: true, params: 65792}
  output_layer: Linear {d_input: 256, d_output: 2, bias: true, params: 514}
  dropout: Dropout {prob: 0.5}
  activation: Relu
  params: 138754
}
Total Epochs: 3

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | CPU Usage  | 55.270   | 1        | 61.351   | 2        |
| Train | CPU Memory | 19.492   | 2        | 19.718   | 1        |
| Train | Loss       | 0.693    | 3        | 0.693    | 1        |
| Train | Accuracy   | 50.980   | 1        | 51.621   | 3        |
| Valid | CPU Usage  | 50.155   | 1        | 56.421   | 3        |
| Valid | CPU Memory | 19.342   | 2        | 19.663   | 3        |
| Valid | Loss       | 0.693    | 1        | 0.693    | 3        |
| Valid | Accuracy   | 51.416   | 1        | 51.416   | 3        |

### Run 9

- Notes

Seems I am stuck at a loss of 0.693. My learning rate or initialization probably off. Going to up my learning rate.

- Config

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 5e-1 |
| weight_decay | 5e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 25 |
| hidden_layers | 3 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |

- Results

Model {
  input_layer: Linear {d_input: 25, d_output: 256, bias: true, params: 6656}
  ln1: Linear {d_input: 256, d_output: 256, bias: true, params: 65792}
  ln2: Linear {d_input: 256, d_output: 256, bias: true, params: 65792}
  output_layer: Linear {d_input: 256, d_output: 2, bias: true, params: 514}
  dropout: Dropout {prob: 0.5}
  activation: Relu
  params: 138754
}
Total Epochs: 6

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | Loss       | 0.719    | 2        | NaN      | 6        |
| Train | CPU Usage  | 54.249   | 6        | 66.030   | 4        |
| Train | Accuracy   | 50.465   | 4        | 50.626   | 1        |
| Train | CPU Memory | 19.801   | 2        | 20.048   | 4        |
| Valid | Loss       | 0.693    | 4        | NaN      | 6        |
| Valid | CPU Usage  | 51.482   | 5        | 57.649   | 2        |
| Valid | Accuracy   | 48.584   | 1        | 51.416   | 5        |
| Valid | CPU Memory | 19.713   | 1        | 20.040   | 6        |

### Run 10

- Notes

The learning rate increase was fine, still only getting my losst to around 0.7. Let's add more layers.

- Config

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 5e-1 |
| weight_decay | 5e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 25 |
| hidden_layers | 7 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |

- Results

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | Accuracy   | 51.278   | 2        | 51.310   | 3        |
| Train | CPU Usage  | 60.536   | 1        | 61.883   | 2        |
| Train | CPU Memory | 19.877   | 3        | 20.124   | 2        |
| Train | Loss       | 0.694    | 2        | 0.698    | 1        |
| Valid | Accuracy   | 48.584   | 1        | 51.416   | 3        |
| Valid | CPU Usage  | 54.317   | 2        | 57.741   | 3        |
| Valid | CPU Memory | 19.761   | 3        | 20.103   | 1        |
| Valid | Loss       | 0.694    | 1        | 0.756    | 2        |

### Run 11

- Notes

Not budging. Going to take out the shuffle and add 2 more workers, and slow wieght decay.

- Config

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 5e-2 |
| weight_decay | 2e-5 |
| batch_size | 256 |
| num_workers | 6 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 25 |
| hidden_layers | 7 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | false |
| bias | true |

- Results

Model {
  input_layer: Linear {d_input: 25, d_output: 256, bias: true, params: 6656}
  ln1: Linear {d_input: 256, d_output: 256, bias: true, params: 65792}
  ln2: Linear {d_input: 256, d_output: 256, bias: true, params: 65792}
  ln3: Linear {d_input: 256, d_output: 256, bias: true, params: 65792}
  ln4: Linear {d_input: 256, d_output: 256, bias: true, params: 65792}
  ln5: Linear {d_input: 256, d_output: 256, bias: true, params: 65792}
  ln6: Linear {d_input: 256, d_output: 256, bias: true, params: 65792}
  output_layer: Linear {d_input: 256, d_output: 2, bias: true, params: 514}
  dropout: Dropout {prob: 0.5}
  activation: Relu
  params: 401922
}
Total Epochs: 5

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | Accuracy   | 51.259   | 5        | 51.362   | 1        |
| Train | CPU Memory | 19.808   | 3        | 20.172   | 4        |
| Train | CPU Usage  | 74.068   | 1        | 77.307   | 4        |
| Train | Loss       | 0.693    | 5        | 0.721    | 1        |
| Valid | Accuracy   | 51.416   | 1        | 51.416   | 5        |
| Valid | CPU Memory | 19.788   | 2        | 20.195   | 4        |
| Valid | CPU Usage  | 72.220   | 4        | 78.686   | 3        |
| Valid | Loss       | 0.693    | 3        | 0.693    | 1        |

### Run 12

- Notes

Last run. Going to add gradient clipping, and reduce the number of layers. Trying to combat maybe a vanishing gradient problem.

- Config

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-2 |
| weight_decay | 5e-5 |
| batch_size | 512 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | SGD |
| input_size | 25 |
| hidden_layers | 2 |
| hidden_layer_size | 512 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |

- Results

Model {
  input_layer: Linear {d_input: 25, d_output: 512, bias: true, params: 13312}
  ln1: Linear {d_input: 512, d_output: 512, bias: true, params: 262656}
  output_layer: Linear {d_input: 512, d_output: 2, bias: true, params: 1026}
  dropout: Dropout {prob: 0.5}
  activation: Relu
  params: 276994
}
Total Epochs: 5

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | CPU Usage  | 49.157   | 1        | 51.136   | 4        |
| Train | Accuracy   | 51.389   | 5        | 51.486   | 2        |
| Train | CPU Memory | 20.042   | 4        | 20.456   | 3        |
| Train | Loss       | 0.693    | 3        | 0.705    | 1        |
| Valid | CPU Usage  | 48.439   | 2        | 51.192   | 3        |
| Valid | Accuracy   | 51.416   | 1        | 51.416   | 5        |
| Valid | CPU Memory | 19.971   | 4        | 20.622   | 3        |
| Valid | Loss       | 0.696    | 3        | 0.701    | 2        |

### Run 13

- Notes

Added two more features, previous bar trend and macd signal. This was more of the same result wise.

- Config

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-2 |
| weight_decay | 5e-5 |
| batch_size | 512 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | SGD |
| input_size | 27 |
| hidden_layers | 2 |
| hidden_layer_size | 512 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |

- Results

Model {
  input_layer: Linear {d_input: 27, d_output: 512, bias: true, params: 14336}
  ln1: Linear {d_input: 512, d_output: 512, bias: true, params: 262656}
  output_layer: Linear {d_input: 512, d_output: 2, bias: true, params: 1026}
  dropout: Dropout {prob: 0.5}
  activation: Relu
  params: 278018
}
Total Epochs: 8

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | CPU Memory | 20.956   | 7        | 21.549   | 2        |
| Train | CPU Usage  | 53.669   | 8        | 58.540   | 3        |
| Train | Loss       | 0.693    | 2        | 0.709    | 1        |
| Train | Accuracy   | 51.283   | 6        | 51.380   | 4        |
| Valid | CPU Memory | 20.979   | 7        | 21.453   | 2        |
| Valid | CPU Usage  | 52.584   | 8        | 59.560   | 6        |
| Valid | Loss       | 0.692    | 6        | 0.698    | 3        |
| Valid | Accuracy   | 48.155   | 1        | 51.845   | 8        |

## Training Round 1 Conclusion

Still not moving the needle. This was expected, as predicting stocks is hard. I may need to rework my data, but a simple model like this was expected to not be very accurate. Random guessing is all that it can do, and my assumption is there is not much predictive power in the data. So my next step it do do some feature engineering and try to improve the dataset.

## Feature Engineering

It seems the data does not have enough predictive power juice in the current set up. A few low hanging fruit features could be added.

### Brainstorm

- Add distance to high/low
- Add distance to bollinger bands
- Add distance to averages

## Training Round 2

### Run 2.0

- Notes

Added the distances as features, about 17 more

- Config

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-5 |
| weight_decay | 5e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 44 |
| hidden_layers | 2 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |

- Results

Model {
  input_layer: Linear {d_input: 44, d_output: 512, bias: true, params: 23040}
  ln1: Linear {d_input: 512, d_output: 512, bias: true, params: 262656}
  output_layer: Linear {d_input: 512, d_output: 2, bias: true, params: 1026}
  dropout: Dropout {prob: 0.5}
  activation: Relu
  params: 286722
}
Total Epochs: 5

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | Accuracy   | 51.184   | 5        | 51.400   | 3        |
| Train | CPU Memory | 21.660   | 1        | 21.842   | 5        |
| Train | CPU Usage  | 55.364   | 4        | 57.356   | 3        |
| Train | Loss       | 0.693    | 2        | 0.731    | 1        |
| Valid | Accuracy   | 51.845   | 1        | 51.845   | 5        |
| Valid | CPU Memory | 21.638   | 1        | 21.885   | 3        |
| Valid | CPU Usage  | 53.882   | 1        | 54.789   | 3        |
| Valid | Loss       | 0.692    | 3        | 0.693    | 4        |

### Run 2.1

- Notes

Last run was more of the same. So let's take out min max norm and add a leaky relu. Also removed volume feature as it is varying too much.

- Config

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-5 |
| weight_decay | 5e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 43 |
| hidden_layers | 2 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | LeakyRelu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |

- Results

Model {
  input_layer: Linear {d_input: 43, d_output: 512, bias: true, params: 22528}
  ln1: Linear {d_input: 512, d_output: 512, bias: true, params: 262656}
  output_layer: Linear {d_input: 512, d_output: 2, bias: true, params: 1026}
  dropout: Dropout {prob: 0.5}
  activation: LeakyRelu {negative_slope: 0.01}
  params: 286210
}
Total Epochs: 6

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | CPU Memory | 20.843   | 5        | 21.023   | 4        |
| Train | CPU Usage  | 50.155   | 5        | 60.643   | 6        |
| Train | Accuracy   | 50.047   | 5        | 50.165   | 6        |
| Train | Loss       | 107.629  | 5        | 155.673  | 6        |
| Valid | CPU Memory | 20.749   | 6        | 21.020   | 5        |
| Valid | CPU Usage  | 48.772   | 4        | 52.440   | 1        |
| Valid | Accuracy   | 48.152   | 1        | 51.865   | 3        |
| Valid | Loss       | 16.754   | 4        | 179.804  | 6        |

### Run 2.3

- Notes

Remove negative slope

- Config

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-5 |
| weight_decay | 5e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 44 |
| hidden_layers | 2 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |

- Results

Model {
  input_layer: Linear {d_input: 43, d_output: 512, bias: true, params: 22528}
  ln1: Linear {d_input: 512, d_output: 512, bias: true, params: 262656}
  output_layer: Linear {d_input: 512, d_output: 2, bias: true, params: 1026}
  dropout: Dropout {prob: 0.5}
  activation: LeakyRelu {negative_slope: 0.01}
  params: 286210
}
Total Epochs: 6

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | CPU Memory | 20.843   | 5        | 21.023   | 4        |
| Train | CPU Usage  | 50.155   | 5        | 60.643   | 6        |
| Train | Accuracy   | 50.047   | 5        | 50.165   | 6        |
| Train | Loss       | 107.629  | 5        | 155.673  | 6        |
| Valid | CPU Memory | 20.749   | 6        | 21.020   | 5        |
| Valid | CPU Usage  | 48.772   | 4        | 52.440   | 1        |
| Valid | Accuracy   | 48.152   | 1        | 51.865   | 3        |
| Valid | Loss       | 16.754   | 4        | 179.804  | 6        |

### Run 2.4

- Notes

Ok at this point I'm wondering if this thing is learning at all, so let's put the target in the features, and only th target.

- Config

| Hyperparameters | Value |
|-----------------|-------|
| epochs | 10 |
| learning_rate | 1e-5 |
| weight_decay | 5e-5 |
| batch_size | 256 |
| num_workers | 4 |
| seed | 42 |
| device | wgpu |
| loss | CrossEntropyLoss |
| optimizer | Adam |
| input_size | 1 |
| hidden_layers | 2 |
| hidden_layer_size | 256 |
| output_size | 2 |
| hidden_layer_activation | Relu |
| output_activation | with logits |
| shuffle_batch | true |
| bias | true |

- Results

Model {
  input_layer: Linear {d_input: 1, d_output: 512, bias: true, params: 1024}
  ln1: Linear {d_input: 512, d_output: 512, bias: true, params: 262656}
  output_layer: Linear {d_input: 512, d_output: 2, bias: true, params: 1026}
  dropout: Dropout {prob: 0.5}
  activation: Relu
  params: 264706
}
Total Epochs: 3

| Split | Metric     | Min.     | Epoch    | Max.     | Epoch    |
|-------|------------|----------|----------|----------|----------|
| Train | CPU Memory | 20.355   | 1        | 20.462   | 2        |
| Train | Accuracy   | 99.890   | 1        | 100.000  | 3        |
| Train | Loss       | 4.286e-5 | 2        | 0.024    | 1        |
| Train | CPU Usage  | 49.532   | 3        | 50.228   | 2        |
| Valid | CPU Memory | 20.354   | 1        | 20.422   | 3        |
| Valid | Accuracy   | 100.000  | 1        | 100.000  | 3        |
| Valid | Loss       | 1.240e-5 | 1        | 3.078e-5 | 3        |
| Valid | CPU Usage  | 47.896   | 3        | 49.296   | 1        |

## Training Round 2 Conclusion

Well good news is that the model can learn, as it was 100 percent for the situation of putting the target in the features, and only the target. So even with the 27 additional features, not much is giving predictive power, at least for a linear model that is just trying to predict the target on one row. The nature of this data is time series, so maybe a time series focused model would be better.

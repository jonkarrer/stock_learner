# Sql

## Create daily_stock_bars for training

```sql
SELECT
 id AS row_id,
 open_price,
 close_price,
 high_price,
 low_price,
 volume,
 volume_weighted_price,
 CASE WHEN bar_trend = 'bullish' THEN 1 ELSE 0 END AS bar_trend,
 buy_or_sell AS label,
 hundred_day_sma,
 hundred_day_ema,
 fifty_day_sma,
 fifty_day_ema,
 twenty_day_sma,
 twenty_day_ema,
 nine_day_sma,
 nine_day_ema,
 hundred_day_high,
 hundred_day_low,
 fifty_day_high,
 fifty_day_low,
 ten_day_high,
 ten_day_low,
 fourteen_day_rsi,
 top_bollinger_band,
 middle_bollinger_band,
 bottom_bollinger_band
FROM daily_stock_bars;
```

## Add Primary key

```sql
INSERT INTO daily_linear_classifier (
 row_id,
 open_price,
 close_price,
 high_price,
 low_price,
 volume,
 volume_weighted_price,
 bar_trend,
 label,
 hundred_day_sma,
 hundred_day_ema,
 fifty_day_sma,
 fifty_day_ema,
 twenty_day_sma,
 twenty_day_ema,
 nine_day_sma,
 nine_day_ema,
 hundred_day_high,
 hundred_day_low,
 fifty_day_high,
 fifty_day_low,
 ten_day_high,
 ten_day_low,
 fourteen_day_rsi,
 top_bollinger_band,
 middle_bollinger_band,
 bottom_bollinger_band
)
SELECT
 id AS row_id,
 open_price,
 close_price,
 high_price,
 low_price,
 volume,
 volume_weighted_price,
 CASE WHEN bar_trend = 'bullish' THEN 1 ELSE 0 END AS bar_trend,
 buy_or_sell AS label,
 hundred_day_sma,
 hundred_day_ema,
 fifty_day_sma,
 fifty_day_ema,
 twenty_day_sma,
 twenty_day_ema,
 nine_day_sma,
 nine_day_ema,
 hundred_day_high,
 hundred_day_low,
 fifty_day_high,
 fifty_day_low,
 ten_day_high,
 ten_day_low,
 fourteen_day_rsi,
 top_bollinger_band,
 middle_bollinger_band,
 bottom_bollinger_band
FROM daily_stock_bars;
```

## Split data into train and validation with validation having future data

create tables

```sql
CREATE TABLE daily_training_set (
  row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    open_price REAL NOT NULL DEFAULT 0.0,
    close_price REAL NOT NULL DEFAULT 0.0,
    high_price REAL NOT NULL DEFAULT 0.0,
    low_price REAL NOT NULL DEFAULT 0.0,
    volume REAL NOT NULL DEFAULT 0.0,
    volume_weighted_price REAL DEFAULT 0.0,
    bar_trend INTEGER NOT NULL,
    label INTEGER NOT NULL,
    hundred_day_sma REAL NOT NULL,
    hundred_day_ema REAL NOT NULL,
    fifty_day_sma REAL NOT NULL,
    fifty_day_ema REAL NOT NULL,
    twenty_day_sma REAL NOT NULL,
    twenty_day_ema REAL NOT NULL,
    nine_day_ema REAL NOT NULL,
    nine_day_sma REAL NOT NULL,
    hundred_day_high REAL NOT NULL,
    hundred_day_low REAL NOT NULL,
    fifty_day_high REAL NOT NULL,
    fifty_day_low REAL NOT NULL,
    ten_day_high REAL NOT NULL,
    ten_day_low REAL NOT NULL,
    fourteen_day_rsi REAL NOT NULL,
    top_bollinger_band REAL NOT NULL,
    middle_bollinger_band REAL NOT NULL,
    bottom_bollinger_band REAL NOT NULL,
    previous_period_trend INTEGER NOT NULL,
    macd_signal INTEGER NOT NULL
);
```

Insert data

```sql
INSERT INTO daily_training_set (
 open_price,
 close_price,
 high_price,
 low_price,
 volume,
 volume_weighted_price,
 bar_trend,
 label,
 hundred_day_sma,
 hundred_day_ema,
 fifty_day_sma,
 fifty_day_ema,
 twenty_day_sma,
 twenty_day_ema,
 nine_day_sma,
 nine_day_ema,
 hundred_day_high,
 hundred_day_low,
 fifty_day_high,
 fifty_day_low,
 ten_day_high,
 ten_day_low,
 fourteen_day_rsi,
 top_bollinger_band,
 middle_bollinger_band,
 bottom_bollinger_band,
 previous_period_trend,
 macd_signal
)
SELECT
 open_price,
 close_price,
 high_price,
 low_price,
 volume,
 volume_weighted_price,
 CASE WHEN bar_trend = 'bullish' THEN 1 ELSE 0 END AS bar_trend,
 buy_or_sell AS label,
 hundred_day_sma,
 hundred_day_ema,
 fifty_day_sma,
 fifty_day_ema,
 twenty_day_sma,
 twenty_day_ema,
 nine_day_sma,
 nine_day_ema,
 hundred_day_high,
 hundred_day_low,
 fifty_day_high,
 fifty_day_low,
 ten_day_high,
 ten_day_low,
 fourteen_day_rsi,
 top_bollinger_band,
 middle_bollinger_band,
 bottom_bollinger_band,
 CASE WHEN previous_period_trend = 'bullish' THEN 1 ELSE 0 END AS previous_period_trend,
 macd_signal
FROM daily_stock_bars
WHERE event_unix_timestamp < 1678910064000;
```

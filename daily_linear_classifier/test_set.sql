CREATE TABLE daily_test_set (
 	row_id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_unix_timestamp INTEGER NOT NULL,
    next_period_price REAL NOT NULL,
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
    macd_signal INTEGER NOT NULL,
    previous_five_day_trend INTEGER NOT NULL,
    previous_ten_day_trend INTEGER NOT NULL,
    future_three_day_trend INTEGER NOT NULL,
    future_five_day_trend INTEGER NOT NULL,
    future_ten_day_trend INTEGER NOT NULL,
    distance_to_hundred_day_sma REAL NOT NULL,
    distance_to_hundred_day_ema REAL NOT NULL,
    distance_to_fifty_day_sma REAL NOT NULL,
    distance_to_fifty_day_ema REAL NOT NULL,
    distance_to_twenty_day_sma REAL NOT NULL,
    distance_to_twenty_day_ema REAL NOT NULL,
    distance_to_nine_day_ema REAL NOT NULL,
    distance_to_nine_day_sma REAL NOT NULL,
    distance_to_hundred_day_high REAL NOT NULL,
    distance_to_hundred_day_low REAL NOT NULL,
    distance_to_fifty_day_high REAL NOT NULL,
    distance_to_fifty_day_low REAL NOT NULL,
    distance_to_ten_day_high REAL NOT NULL,
    distance_to_ten_day_low REAL NOT NULL,
    distance_to_top_bollinger_band REAL NOT NULL,
    distance_to_middle_bollinger_band REAL NOT NULL,
    distance_to_bottom_bollinger_band REAL NOT NULL
);

INSERT INTO daily_test_set (
    event_unix_timestamp,
    next_period_price,
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
    macd_signal,
    previous_five_day_trend,
    previous_ten_day_trend,
    future_three_day_trend,
    future_five_day_trend,
    future_ten_day_trend,
    distance_to_hundred_day_sma,
    distance_to_hundred_day_ema,
    distance_to_fifty_day_sma,
    distance_to_fifty_day_ema,
    distance_to_twenty_day_sma,
    distance_to_twenty_day_ema,
    distance_to_nine_day_ema,
    distance_to_nine_day_sma,
    distance_to_hundred_day_high,
    distance_to_hundred_day_low,
    distance_to_fifty_day_high,
    distance_to_fifty_day_low,
    distance_to_ten_day_high,
    distance_to_ten_day_low,
    distance_to_top_bollinger_band,
    distance_to_middle_bollinger_band,
    distance_to_bottom_bollinger_band
)
SELECT
    event_unix_timestamp,
    next_period_price,
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
    macd_signal,
    CASE WHEN previous_five_day_trend = 'bullish' THEN 1 ELSE 0 END AS previous_five_day_trend,
    CASE WHEN previous_ten_day_trend = 'bullish' THEN 1 ELSE 0 END AS previous_ten_day_trend,
    CASE WHEN future_three_day_trend = 'bullish' THEN 1 ELSE 0 END AS future_three_day_trend,
    CASE WHEN future_five_day_trend = 'bullish' THEN 1 ELSE 0 END AS future_five_day_trend,
    CASE WHEN future_ten_day_trend = 'bullish' THEN 1 ELSE 0 END AS future_ten_day_trend,
    distance_to_hundred_day_sma,
    distance_to_hundred_day_ema,
    distance_to_fifty_day_sma,
    distance_to_fifty_day_ema,
    distance_to_twenty_day_sma,
    distance_to_twenty_day_ema,
    distance_to_nine_day_ema,
    distance_to_nine_day_sma,
    distance_to_hundred_day_high,
    distance_to_hundred_day_low,
    distance_to_fifty_day_high,
    distance_to_fifty_day_low,
    distance_to_ten_day_high,
    distance_to_ten_day_low,
    distance_to_top_bollinger_band,
    distance_to_middle_bollinger_band,
    distance_to_bottom_bollinger_band
FROM daily_stock_bars
WHERE event_unix_timestamp >= 1688910064000;
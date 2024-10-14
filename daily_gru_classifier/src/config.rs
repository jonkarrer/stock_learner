pub struct Config {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub sequence_length: usize,
    pub batch_size: usize,
}

impl Config {
    pub fn new() -> Self {
        Self {
            input_size: 46,
            hidden_size: 512,
            output_size: 2,
            sequence_length: 9,
            batch_size: 512,
        }
    }
}

pub fn get_config() -> Config {
    Config::new()
}

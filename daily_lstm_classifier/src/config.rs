pub struct ModelConfig {
    pub input_size: usize,
    pub hidden_size: usize,
    pub output_size: usize,
    pub sequence_length: usize,
    pub dropout: f32,
}

impl ModelConfig {
    pub fn new() -> Self {
        Self {
            input_size: 46,
            hidden_size: 512,
            output_size: 2,
            sequence_length: 3,
            dropout: 0.3,
        }
    }
}

pub fn get_config() -> ModelConfig {
    ModelConfig::new()
}

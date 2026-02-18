use crate::generation::{streamer::TokenStreamer, GenerationConfig};
use anyhow::Result;
use candle_core::Device;

pub trait ModelForCausalLM {
    fn device(&self) -> &Device;
    fn generate(
        &mut self,
        input_ids: &[u32],
        config: &GenerationConfig,
        mut streamer: Option<&mut dyn TokenStreamer>,
    ) -> Result<Vec<u32>> {
        // a default implementation
        let mut output = Vec::with_capacity(config.max_new_tokens);
        for _ in 0..config.max_new_tokens {
            let next_token = self.generate_next_token(input_ids)?;
            if let Some(streamer) = streamer.as_deref_mut() {
                streamer.append(next_token)?;
            }
            if Some(next_token) == config.eos_token_id {
                break;
            }
            output.push(next_token);
        }
        if let Some(streamer) = streamer.as_deref_mut() {
            streamer.finalize()?;
        }
        Ok(output)
    }

    fn generate_next_token(&self, _input_ids: &[u32]) -> Result<u32> {
        unimplemented!("Implement specific token generation logic")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::streamer::{AsyncTextStreamer, StreamerMessage};
    use anyhow::Result;
    use std::time::Duration;

    struct DummyModel {
        device: Device,
        next_token: u32,
    }

    impl DummyModel {
        fn new(next_token: u32) -> Self {
            Self {
                device: Device::Cpu,
                next_token,
            }
        }
    }

    impl ModelForCausalLM for DummyModel {
        fn device(&self) -> &Device {
            &self.device
        }

        fn generate_next_token(&self, _input_ids: &[u32]) -> Result<u32> {
            Ok(self.next_token)
        }
    }

    #[test]
    fn generate_always_sends_end_message() -> Result<()> {
        let mut model = DummyModel::new(1);
        let config = GenerationConfig::with_max_tokens(3);

        let (mut streamer, receiver) =
            AsyncTextStreamer::new(Box::new(|token_id| Ok(token_id.to_string())));

        let _ = model.generate(&[42], &config, Some(&mut streamer))?;

        let mut end_seen = false;
        for _ in 0..8 {
            match receiver.recv_timeout(Duration::from_millis(200))? {
                StreamerMessage::Token(_) => {}
                StreamerMessage::End => {
                    end_seen = true;
                    break;
                }
            }
        }

        assert!(end_seen);
        Ok(())
    }
}

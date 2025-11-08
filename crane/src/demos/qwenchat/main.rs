use clap::Parser;
use colored::*;
use crane_core::generation::streamer::StreamerMessage;
use crane_core::{
    Msg,
    autotokenizer::AutoTokenizer,
    chat::{Message, Role},
    generation::{
        GenerationConfig, based::ModelForCausalLM, streamer::AsyncTextStreamer,
        streamer::TokenStreamer,
    },
    models::{DType, Device, qwen25::Model as Qwen25Model},
};
use std::io::{self, Write};

#[derive(Parser, Debug)]
#[clap(about, version, author)]
struct Args {
    #[clap(short('m'), long, default_value = "checkpoints/Qwen2.5-0.5B-Instruct")]
    model_path: String,
}

struct ChatCLI {
    history: Vec<Message>,
    tokenizer: AutoTokenizer,
    model: Qwen25Model,
    gen_config: GenerationConfig,
    max_turns: usize,
}

impl ChatCLI {
    fn new(args: Args) -> anyhow::Result<Self> {
        let dtype = DType::F16;
        let device = Device::Cpu;

        let tokenizer = AutoTokenizer::from_pretrained(&args.model_path, None).unwrap();
        let model = Qwen25Model::new(&args.model_path, &device, &dtype)?;

        let gen_config = GenerationConfig {
            max_new_tokens: 235,
            temperature: Some(0.67),
            top_p: Some(1.0),
            repetition_penalty: 1.1,
            repeat_last_n: 1,
            do_sample: false,
            pad_token_id: tokenizer.get_token("<|end_of_text|>"),
            eos_token_id: tokenizer.get_token("<|im_end|>"),
            report_speed: true,
        };

        Ok(Self {
            history: Vec::new(),
            tokenizer,
            model,
            gen_config,
            max_turns: 4,
        })
    }

    fn print_banner(&self) {
        println!("{}", "Welcome to Qwen Chat!".bright_green());
        println!(
            "{}",
            "Type your message below (type 'exit' to quit)\n".bright_green()
        );
    }

    fn get_user_input(&self) -> anyhow::Result<String> {
        print!("{} ", "You:".bold().bright_green());
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        Ok(input.trim().to_string())
    }

    fn generate_response(&mut self, prompt: &str, streamer: &mut dyn TokenStreamer) -> String {
        let input_ids = self.model.prepare_inputs(prompt).unwrap();

        let output_ids = self
            .model
            .generate(&input_ids, &self.gen_config, Some(streamer))
            .unwrap();
        self.tokenizer.decode(&output_ids, false).unwrap()
    }

    fn run(&mut self) -> anyhow::Result<()> {
        self.print_banner();
        loop {
            let input = self.get_user_input()?;
            if input.to_lowercase() == "exit" {
                break;
            }
            self.history.push(Msg!(Role::User, &input));
            let prompt = self
                .tokenizer
                .apply_chat_template(&self.history, true)
                .unwrap();
            print!("{} ", "AI:".bold().bright_magenta());

            let (mut streamer, receiver) = AsyncTextStreamer::new(self.tokenizer.clone());

            // Start a thread to handle streaming tokens
            let handle = std::thread::spawn(move || {
                let mut response_text = String::new();

                for message in receiver {
                    match message {
                        StreamerMessage::Token(token_text) => {
                            // Print token immediately as it arrives
                            print!("{}", token_text);
                            std::io::stdout().flush().unwrap();

                            // Collect for final response
                            response_text.push_str(&token_text);
                        }
                        StreamerMessage::End => {
                            println!(); // New line after generation completes
                            break;
                        }
                    }
                }
                response_text
            });

            // Generate response (this will send tokens to the streamer)
            let _response = self.generate_response(&prompt, &mut streamer);

            // Wait for streaming to complete and get the collected response text
            let response_text = handle.join().unwrap();

            // Add the response to history
            self.history.push(Msg!(Role::Assistant, &response_text));

            if self.history.len() > 2 * self.max_turns {
                self.history
                    .drain(0..(self.history.len() - 2 * self.max_turns));
            }
        }
        println!("{}", "\nGoodbye!".bright_cyan());
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let mut chat = ChatCLI::new(args)?;
    chat.run()
}

use anyhow::Result;
use clap::Parser;

#[tokio::main]
async fn main() -> Result<()> {
    crane_serve::init_logging();
    crane_serve::run(crane_serve::Args::parse()).await
}

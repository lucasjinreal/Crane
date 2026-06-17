#[tokio::main]
async fn main() -> anyhow::Result<()> {
    crane_serve::cli_main().await
}

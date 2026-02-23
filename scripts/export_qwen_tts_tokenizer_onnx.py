import sys
import os
import torch
from transformers import AutoConfig, AutoModel

def main():
    if len(sys.argv) < 3:
        print("Usage: python export_qwen_tts_tokenizer_onnx.py <model_dir> <output_onnx_path>")
        sys.exit(1)

    model_dir = sys.argv[1]
    output_onnx_path = sys.argv[2]
    
    # Needs qwen_tts python package to be importable or in path
    try:
        import qwen_tts
    except ImportError:
        # try to add vendor/Qwen3-TTS to sys.path
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../vendor/Qwen3-TTS")))

    from qwen_tts.core.tokenizer_12hz import Qwen3TTSTokenizerV2Model, Qwen3TTSTokenizerV2Config

    AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
    AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

    print(f"Loading tokenizer model from {model_dir}...")
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    model.eval()

    # The ONNX model should just be the decoder part to turn codes into waveform
    decoder = model.decoder
    decoder.eval()

    # Dummy input for decoder: [batch_size, num_quantizers, seq_len]
    # For 12Hz tokenizer, num_quantizers is 16.
    batch_size = 1
    num_quantizers = model.config.num_quantizers # typically 16
    seq_len = 100 # arbitrary length

    dummy_codes = torch.randint(0, 1024, (batch_size, num_quantizers, seq_len), dtype=torch.long)
    
    # We want to export only `decoder(codes)` which returns [batch_size, out_channels, out_seq_len]
    # Or in our case, the output is `[batch_size, 1, out_seq_len]`
    print(f"Exporting decoder to {output_onnx_path}...")

    torch.onnx.export(
        decoder,
        (dummy_codes,),
        output_onnx_path,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["codes"],
        output_names=["audio_values"],
        dynamic_axes={
            "codes": {0: "batch_size", 2: "seq_len"},
            "audio_values": {0: "batch_size", 2: "out_seq_len"},
        }
    )

    print("ONNX export complete.")

if __name__ == "__main__":
    main()

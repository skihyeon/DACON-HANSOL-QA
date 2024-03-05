# cd llama.cpp

# pip install -r requirements.txt
# make LLAMA_OPENBLAS=1

MODEL_PATH=$(jq -r '.model_path' /configs/gguf_transfer_config.json)
TRANSFER_NAME=$(jq -r '.transfer_model_name' /configs/gguf_transfer_config.json)
QUANTIZE_NAME=$(jq -r '.quantize_model_name' /configs/gguf_transfer_config.json)
QUANTIZE_OPTION=$(jq -r '.quantize_option' /configs/gguf_transfer_config.json)

# make LLAMA_OPENBLAS=1

python /llama.cpp/convert.py "$MODEL_PATH"

llama.cpp/quantize "$MODEL_PATH/$TRANSFER_NAME" "$MODEL_PATH/$QUANTIZE_NAME" "$QUANTIZE_OPTION"`
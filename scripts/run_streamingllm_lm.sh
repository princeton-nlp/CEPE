CONFIGS=(test_ab_32k_prevdoc_16x_4096 test_ab_32k_prevdoc_112x_4096 test_ab_128k_prevdoc_496x_4096 test_pg19_16x_4096 test_pg19_112x_4096 test_pg19_496x_4096 test_proofpile_16x_4096 test_proofpile_112x_4096 test_proofpile_496x_4096 test_codeparrot_16x_4096 test_codeparrot_112x_4096 test_codeparrot_496x_4096)
MODEL_NAME="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="output/Llama-2-7b-hf-streamingllm"
MODEL_CLASS="streamingllm"

for CONFIG in "${CONFIGS[@]}"; do
    if [[ $FILE =~ "eval_ab" || $FILE =~ "test_ab" ]]; then
        DOMAINS=("arxiv" "book")
    elif [[ $FILE =~ "eval_cat" || $FILE =~ "test_cat" ]]; then
        DOMAINS=("arxiv" "book" "c4-rp" "cc" "github" "stackexchange" "wiki")
    else
        DOMAINS=("")
    fi

    for DOMAIN in "${DOMAINS[@]}"; do
        echo "Config file                   = $FILE"
        echo "running evaluation for domain $DOMAIN"
        python eval_lm.py \
            --config configs/$CONFIG \
            --model_name_or_path $MODEL_NAME \
            --model_class $MODEL_CLASS \
            --validation_load_strategy "put_in_decoder" \
            --eval_step_size 2048 \
            --enable_positional_shift True \
            --cache_start_size 4 \
            --cache_recent_size 2044 \
            --validation_domains "$DOMAIN" \
            --output_dir $OUTPUT_DIR \
            --report_to none $OPTIONS
    done
done

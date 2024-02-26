CONFIGS=(test_pg19_16x_4096 test_proofpile_16x_4096 test_codeparrot_16x_4096 test_ab_32k_prevdoc_16x_4096 test_cat_8k_retdoc_8x test_cat_8k_retdoc_20x test_cat_8k_retdoc_50x)

MODEL_NAME="meta-llama/Llama-2-7b-hf"
MODEL_CLASS="replug"
OUTPUT_DIR=output/Llama-2-7b-hf-replug

for CONFIG in "${CONFIGS[@]}"; do
    if [[ $FILE =~ "eval_ab" || $FILE =~ "test_ab" ]]; then
        DOMAINS=("arxiv" "book")
        OPTIONS="--chunk_size 3840"
    elif [[ $FILE =~ "eval_cat" || $FILE =~ "test_cat" ]]; then
        DOMAINS=("arxiv" "book" "c4-rp" "cc" "github" "stackexchange" "wiki")
    else
        DOMAINS=("")
        OPTIONS="--chunk_size 3840"
    fi

    for DOMAIN in "${DOMAINS[@]}"; do
        echo "Config file                   = $FILE"
        echo "running evaluation for domain $DOMAIN"
        python eval_lm.py \
            --config configs/$CONFIG \
            --model_name_or_path $MODEL_NAME \
            --model_class $MODEL_CLASS \
            --validation_load_strategy "put_in_decoder" \
            --validation_domains "$DOMAIN" \
            --output_dir $OUTPUT_DIR \
            --remove_unused_columns False \
            --replug_separate_forward True \
            --report_to none $OPTIONS
    done
done
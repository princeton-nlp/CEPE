CONFIGS=(test_pg19_8x_2048 test_pg19_16x_4096 test_pg19_112x_4096 test_pg19_496x_4096 test_proofpile_8x_2048 test_proofpile_16x_4096 test_proofpile_112x_4096 test_proofpile_496x_4096 test_codeparrot_8x_2048 test_codeparrot_16x_4096 test_codeparrot_112x_4096 test_codeparrot_496x_4096 test_cat_8k_retdoc_8x test_cat_8k_retdoc_20x test_cat_8k_retdoc_50x test_cat_8k_retdoc_100x test_ab_32k_prevdoc_8x_2048 test_ab_32k_prevdoc_16x_4096 test_ab_32k_prevdoc_112x_4096 test_ab_128k_prevdoc_496x_4096)
MODEL_NAME="hyen/CEPE-LLaMA-2-7B"
OUTPUT_DIR="output/CEPE-LLaMA-2-7B"
MODEL_CLASS="context"

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
            --validation_domains "$DOMAIN" \
            --output_dir $OUTPUT_DIR \
            --report_to none $OPTIONS
    done
done

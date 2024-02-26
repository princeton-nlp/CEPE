CONFIGS=(test_pg19_vanilla_2048 test_pg19_8x_2048 test_pg19_16x_4096 test_pg19_112x_4096 test_pg19_496x_4096 test_proofpile_vanilla_2048  test_proofpile_8x_2048 test_proofpile_16x_4096 test_proofpile_112x_4096 test_codeparrot_vanilla_2048 test_codeparrot_8x_2048 test_codeparrot_16x_4096 test_codeparrot_112x_4096 test_cat_8k_vanilla_2048 test_cat_8k_prevdoc_8x test_cat_8k_retdoc_8x test_ab_32k_vanilla_2048 test_ab_32k_prevdoc_8x_2048 test_ab_32k_prevdoc_16x_4096 test_ab_32k_prevdoc_112x_4096 test_cat_8k_retdoc_20x test_cat_8k_retdoc_50x)

MODEL_NAMES=("meta-llama/Llama-2-7b-hf" "togethercomputer/LLaMA-2-7B-32K" "/path/to/NousResearch/Yarn-Llama-2-7b-64k" "/path/to/NousResearch/Yarn-Llama-2-7b-128k")
MODEL_CLASS="vanilla"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    OUTPUT_DIR=output/$(basename -- "$MODEL_NAME")

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
                --validation_domains "$DOMAIN" \
                --output_dir $OUTPUT_DIR \
                --report_to none $OPTIONS
        done
    done
done
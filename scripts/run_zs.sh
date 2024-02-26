CONFIGS=("eval_longqa.yaml" "eval_summ.yaml" "eval_govreport.yaml")
SEEDS=42

MODEL_NAME="hyen/CEPE-LLaMA-2-7B"
OUTPUT_DIR="output/CEPE-LLaMA-2-7B"
MODEL_CLASS="context"

for CONFIG in ${CONFIGS[@]}; do
    OPTIONS="--n_shot_encoder 1 --n_shot_decoder 2 --shot $(expr $N + 2)"

    # multiple seeds due to sampling but this actually doesn't make a big difference on the results
    if [[ $CONFIG == "eval_govreport.yaml" ]]; then
        SEEDS=44
    fi
    for N in 2048 6144 14336 30720; do
        STRAT="concat$N-truncate_left"
        for SEED in $(seq 42 $SEEDS); do
            python eval_downstream.py \
                --config configs/$CONFIG \
                --seed $SEED \
                --model_name_or_path $MODEL_NAME \
                --model_class $MODEL_CLASS \
                --output_dir $OUTPUT_DIR \
                --context_strategy $STRAT \
                --report_to none $OPTIONS
        done
    done
done

MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
OUTPUT_DIR="output/Llama-2-7b-chat-hf"
MODEL_CLASS="vanilla"
for CONFIG in ${CONFIGS[@]}; do

    if [[ $CONFIG == "eval_govreport.yaml" ]]; then
        SEEDS=44
    fi
    for SEED in $(seq 42 44); do
        OPTIONS="--n_test_doc $N"
        python eval_downstream.py \
            --config configs/$CONFIG \
            --seed $SEED \
            --model_name_or_path $MODEL_NAME \
            --model_class $MODEL_CLASS \
            --output_dir $OUTPUT_DIR \
            --report_to none $OPTIONS
    done
done


MODEL_NAME="togethercomputer/Llama-2-7B-32K-Instruct"
OUTPUT_DIR="output/Llama-2-7B-32K-Instruct"
MODEL_CLASS="vanilla"
for CONFIG in ${CONFIGS[@]}; do

    if [[ $CONFIG == "eval_govreport.yaml" ]]; then
        SEEDS=44
    fi
    for SEED in $(seq 42 44); do
        OPTIONS="--n_test_doc $N"
        python eval_downstream.py \
            --config configs/$CONFIG \
            --seed $SEED \
            --model_name_or_path $MODEL_NAME \
            --model_class $MODEL_CLASS \
            --output_dir $OUTPUT_DIR \
            --input_max_length 32768 \
            --report_to none $OPTIONS
    done
done
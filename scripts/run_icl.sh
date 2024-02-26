CONFIG="eval_icl.yaml"

MODEL_NAME="hyen/CEPE-LLaMA-2-7B"
OUTPUT_DIR="output/CEPE-LLaMA-2-7B"
MODEL_CLASS="context"

for N in 18 38; do
    OPTIONS="--n_shot_encoder 1 --n_shot_decoder 2 --shot $(expr $N + 2)"
    for SEED in $(seq 42 44); do
        python eval_downstream.py \
            --config configs/$CONFIG \
            --seed $SEED \
            --model_name_or_path $MODEL_NAME \
            --model_class $MODEL_CLASS \
            --output_dir $OUTPUT_DIR \
            --report_to none $OPTIONS
    done
done

MODEL_NAME="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="output/Llama-2-7b-hf"
MODEL_CLASS="vanilla"
for N in 2 40; do
    for SEED in $(seq 42 44); do
        OPTIONS="--n_test_doc $N"
        python eval_downstream.py \
            --config configs/$CONFIG \
            --seed $SEED \
            --model_name_or_path $MODEL_NAME \
            --model_class $MODEL_CLASS \
            --output_dir $OUTPUT_DIR \
            --shot $N \
            --n_shot_decoder $N \
            --report_to none $OPTIONS
    done
done
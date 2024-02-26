CONFIG="test_qa.yaml"

MODEL_NAME="hyen/CEPE-LLaMA-2-7B"
OUTPUT_DIR="output/CEPE-LLaMA-2-7B"
MODEL_CLASS="context"

for N in 5 10 20 50; do
    OPTIONS="--n_test_doc_encoder 1 --n_test_doc_decoder 10 --n_test_doc $(expr $N + 10)"
    python eval_downstream.py \
        --config configs/$CONFIG \
        --model_name_or_path $MODEL_NAME \
        --model_class $MODEL_CLASS \
        --output_dir $OUTPUT_DIR \
        --report_to none $OPTIONS
done


MODEL_CLASS="vanilla"
for MODEL_NAME in "meta-llama/Llama-2-7b-hf" "togethercomputer/LLaMA-2-7B-32K"; do
    OUTPUT_DIR="output/$(basename -- "$MODEL_NAME")"
    # Note that Llama 2 beyond 20 documents will be truncated, which would give the same results as 20 documents
    for N in 10 15 20 30 60; do
        OPTIONS="--n_test_doc $N"
        python eval_downstream.py \
            --config configs/$CONFIG \
            --model_name_or_path $MODEL_NAME \
            --model_class $MODEL_CLASS \
            --output_dir $OUTPUT_DIR \
            --report_to none $OPTIONS
    done
done

MODEL_NAME="meta-llama/Llama-2-7b-hf"
OUTPUT_DIR="output/$(basename -- $MODEL_NAME)-replug"
MODEL_CLASS="replug"
for N in 5 10 20 30; do
    OPTIONS="--n_test_doc_decoder 0 --n_test_doc_encoder $N --n_test_doc $N"
    python eval_downstream.py \
        --config configs/$CONFIG \
        --model_name_or_path $MODEL_NAME \
        --model_class $MODEL_CLASS \
        --output_dir $OUTPUT_DIR \
        --report_to none $OPTIONS
done
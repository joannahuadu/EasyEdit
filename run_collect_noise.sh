#!/bin/bash

template_yaml="hparams/XSpace/llava_updownqv_co_noise.yaml"

noise_values=(0 0.001 0.05 0.1 0.5)

for noise in "${noise_values[@]}"; do
    
    new_metrics_name="collect_wL24_n${noise}.pt"

    echo "Running with noise=${noise}"

    sed -i "s/^noise: .*/noise: ${noise}/" "$template_yaml"
    sed -i "s/^all_metrics_name: .*/all_metrics_name: '${new_metrics_name}'/" "$template_yaml"

    CUDA_VISIBLE_DEVICES=3 python -m multimodal_edit --model llava --function_name edit_XSpace_LLaVA_VQA_3

    echo "Finished run with noise=${noise}"
    echo "========================================="

done
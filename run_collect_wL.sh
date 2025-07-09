#!/bin/bash

template_yaml="hparams/XSpace/llava_updownqv_co.yaml"

wL_values=(0 8 24 40 56 72)

for wL in "${wL_values[@]}"; do
    
    new_metrics_name="collect_wL${wL}_n0.01.pt"

    echo "Running with wL=${wL}"

    sed -i "s/^wL: .*/wL: ${wL}/" "$template_yaml"
    sed -i "s/^all_metrics_name: .*/all_metrics_name: '${new_metrics_name}'/" "$template_yaml"

    CUDA_VISIBLE_DEVICES=0 python -m multimodal_edit --model llava --function_name edit_XSpace_LLaVA_VQA_2

    echo "Finished run with wL=${wL}"
    echo "========================================="

done
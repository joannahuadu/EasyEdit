#!/bin/bash

template_yaml="hparams/XSpace/llava_updownqv_wL.yaml"

wL_values=(0 8 40 56 72)

for wL in "${wL_values[@]}"; do
    
    new_metrics_name="all_metrics_layer7_updownqv_null_freezeB_null_ds_ep70_th10_sim0.05_wL${wL}_wS8_1e-4_5e-4.jsonl"

    echo "Running with wL=${wL}"

    sed -i "s/^wL: .*/wL: ${wL}/" "$template_yaml"
    sed -i "s/^all_metrics_name: .*/all_metrics_name: '${new_metrics_name}'/" "$template_yaml"

    CUDA_VISIBLE_DEVICES=1 python -m multimodal_edit --model llava --function_name edit_XSpace_LLaVA_VQA_1

    echo "Finished run with wL=${wL}"
    echo "========================================="

done
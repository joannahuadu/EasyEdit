#!/bin/bash

template_yaml="hparams/XSpace/llava_updownqv.yaml"

sim_values=(0.75)

for sim in "${sim_values[@]}"; do
    
    new_metrics_name="all_metrics_layer7_updownqv_null_freezeB_null_ds_ep70_th10_sim${sim}_wL24_wS8_1e-4_5e-4.jsonl"

    echo "Running with sim=${sim}"

    # sed -e "s/^sim: .*/sim: ${sim}/" \
    #     -e "s/^all_metrics_name: .*/all_metrics_name: '${new_metrics_name}'/" \
    #     "$template_yaml" > $template_yaml
    sed -i "s/^sim: .*/sim: ${sim}/" "$template_yaml"
    sed -i "s/^all_metrics_name: .*/all_metrics_name: '${new_metrics_name}'/" "$template_yaml"

    CUDA_VISIBLE_DEVICES=2 python -m multimodal_edit --model llava --function_name edit_XSpace_LLaVA_VQA

    echo "Finished run with sim=${sim}"
    echo "========================================="

done
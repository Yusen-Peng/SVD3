tmux new -s lora
conda activate compress
accelerate launch --config_file configs/accelerate/ddp.yaml --num_processes 1 --num_machines 1 Pi3_main/Pi3_LoRA.py --prune_model /data/wanghaoxuan/SVD_Pi3_cache/Pi3_whitening_only_0.8.safetensors --num_epochs 50 --batch_size 4 --micro_batch_size 1 --learning_rate 1e-4 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 > 50_epochs_train.log 2>&1
# then ctrl+b d to detach
# to reattach: tmux attach -t lora
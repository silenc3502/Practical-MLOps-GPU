pip install deltalake
pip install pandas
pip install mlflow transformers datasets

# CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install transformers[torch]

# Arguments

```text
>>> from transformers import TrainingArguments
>>> TrainingArguments.__init__.__code__.co_varnames
('self', 'output_dir', 'overwrite_output_dir', 'do_train', 'do_eval', 'do_predict', 'eval_strategy', 'prediction_loss_only', 
'per_device_train_batch_size', 'per_device_eval_batch_size', 'per_gpu_train_batch_size', 'per_gpu_eval_batch_size', 
'gradient_accumulation_steps', 'eval_accumulation_steps', 'eval_delay', 'torch_empty_cache_steps', 'learning_rate', 
'weight_decay', 'adam_beta1', 'adam_beta2', 'adam_epsilon', 'max_grad_norm', 'num_train_epochs', 'max_steps', 
'lr_scheduler_type', 'lr_scheduler_kwargs', 'warmup_ratio', 'warmup_steps', 'log_level', 'log_level_replica', 
'log_on_each_node', 'logging_dir', 'logging_strategy', 'logging_first_step', 'logging_steps', 'logging_nan_inf_filter', 
'save_strategy', 'save_steps', 'save_total_limit', 'save_safetensors', 'save_on_each_node', 'save_only_model', 
'restore_callback_states_from_checkpoint', 'no_cuda', 'use_cpu', 'use_mps_device', 'seed', 'data_seed', 'jit_mode_eval', 
'use_ipex', 'bf16', 'fp16', 'fp16_opt_level', 'half_precision_backend', 'bf16_full_eval', 'fp16_full_eval', 'tf32', 'local_rank', 
'ddp_backend', 'tpu_num_cores', 'tpu_metrics_debug', 'debug', 'dataloader_drop_last', 'eval_steps', 'dataloader_num_workers', 
'dataloader_prefetch_factor', 'past_index', 'run_name', 'disable_tqdm', 'remove_unused_columns', 'label_names', 
'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better', 'ignore_data_skip', 'fsdp', 'fsdp_min_num_params', 
'fsdp_config', 'fsdp_transformer_layer_cls_to_wrap', 'accelerator_config', 'parallelism_config', 'deepspeed', 
'label_smoothing_factor', 'optim', 'optim_args', 'adafactor', 'group_by_length', 'length_column_name', 'report_to', 
'ddp_find_unused_parameters', 'ddp_bucket_cap_mb', 'ddp_broadcast_buffers', 'dataloader_pin_memory', 'dataloader_persistent_workers', 
'skip_memory_metrics', 'use_legacy_prediction_loop', 'push_to_hub', 'resume_from_checkpoint', 'hub_model_id', 'hub_strategy', 
'hub_token', 'hub_private_repo', 'hub_always_push', 'hub_revision', 'gradient_checkpointing', 'gradient_checkpointing_kwargs', 
'include_inputs_for_metrics', 'include_for_metrics', 'eval_do_concat_batches', 'fp16_backend', 'push_to_hub_model_id', 
'push_to_hub_organization', 'push_to_hub_token', 'mp_parameters', 'auto_find_batch_size', 'full_determinism', 'torchdynamo', 
'ray_scope', 'ddp_timeout', 'torch_compile', 'torch_compile_backend', 'torch_compile_mode', 'include_tokens_per_second', 
'include_num_input_tokens_seen', 'neftune_noise_alpha', 'optim_target_modules', 'batch_eval_metrics', 'eval_on_start', 
'use_liger_kernel', 'liger_kernel_config', 'eval_use_gather_object', 'average_tokens_across_devices')
```

# After Training

mlflow ui  

# Check UI

http://localhost:5000/  

<img width="544" height="395" alt="image" src="https://github.com/user-attachments/assets/6c352577-1ad5-4e28-bb80-6b60a16d0091" />

<img width="998" height="432" alt="image" src="https://github.com/user-attachments/assets/4d58a8f8-02f1-4ceb-b35f-69c2fb4656bb" />


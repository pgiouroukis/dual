python main.py \
	--output_path 							_experiments_/bart-base/reddit-tifu/random \
	--model_name 							facebook/bart-base \
\
	--active_learning_strategy 				random \
	--active_learning_iterations 			15 \
	--active_learning_samples_per_iteration 10 \
\
	--num_train_epochs 						6 \
	--train_batch_size 						16 \
	--gradient_accumulation_steps 			1 \
	--min_train_steps 						350 \
	--optim 								adamw_torch \
	--learning_rate 						2e-5 \
	--weight_decay 							0.028 \
	--warmup_ratio 							0.1 \
	--num_beams 							4 \
	--train_validation 						0 \
	--train_validation_samples 				0 \
	--eval_batch_size 						25 \
\
	--dataset_hf_name 						reddit_tifu \
	--dataset_hf_config 					long \
	--dataset_train_split 					train \
	--dataset_test_split 					test \
	--dataset_validation_split 				validation \
	--dataset_document_column 				documents \
	--dataset_summary_column 				tldr \
	--eval_samples 							-1 \
	--max_source_length 					512 \
	--max_generation_length 				128 \
\
	--seeds 								42

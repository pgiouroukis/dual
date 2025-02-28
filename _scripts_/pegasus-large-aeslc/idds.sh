python main.py \
	--output_path 							_experiments_/pegasus-large/aeslc/idds \
	--model_name 							google/pegasus-large \
\
	--active_learning_strategy 				idds \
	--active_learning_iterations 			15 \
	--active_learning_samples_per_iteration 10 \
\
	--num_train_epochs 						4 \
	--train_batch_size 						8 \
	--gradient_accumulation_steps 			1 \
	--min_train_steps 						200 \
	--optim 								adamw_torch \
	--learning_rate 						5e-4 \
	--weight_decay 							0.03 \
	--warmup_ratio 							0.1 \
	--num_beams 							4 \
	--train_validation 						0 \
	--train_validation_samples 				0 \
	--eval_batch_size 						16 \
\
	--dataset_hf_name 						aeslc \
	--dataset_train_split 					train \
	--dataset_test_split 					test \
	--dataset_validation_split 				validation \
	--dataset_document_column 				email_body \
	--dataset_summary_column 				subject_line \
	--eval_samples 							-1 \
	--max_source_length 					512 \
	--max_generation_length 				32 \
\
	--embeddings_pt_tensor_path 			_embeddings_/aeslc-tsdae.pt \
\
	--seeds 								42

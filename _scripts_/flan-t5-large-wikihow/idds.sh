python main.py \
	--output_path 							_experiments_/flan-t5-large/wikihow/idds \
	--model_name 							google/flan-t5-large \
\
	--active_learning_strategy 				idds \
	--active_learning_iterations 			15 \
	--active_learning_samples_per_iteration 10 \
\
	--num_train_epochs                      3 \
	--train_batch_size                      4 \
	--min_train_steps                       0 \
	--train_validation                      0 \
	--train_validation_samples             	0 \
	--eval_batch_size                       16 \
	--learning_rate                         3e-5 \
	--gradient_accumulation_steps           1 \
	--optim                                 adafactor \
	--weight_decay                          0.01 \
	--warmup_ratio                          0.1 \
	--num_beams                             3 \
\
	--dataset_hf_name 						wikihow \
	--dataset_hf_config 					all \
    --dataset_data_dir 						_datasets_/wikihow \
	--dataset_train_split 					train \
	--dataset_test_split 					test \
	--dataset_validation_split 				validation \
	--dataset_document_column 				text \
	--dataset_summary_column 				headline \
	--eval_samples 							1500 \
	--max_source_length 					512 \
	--max_generation_length 				256 \
\
	--embeddings_pt_tensor_path 			_embeddings_/wikihow-tsdae.pt \
\
	--seeds 								42

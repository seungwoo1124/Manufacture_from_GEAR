python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method densesparseuniformquantization --quantize_bit 4 --left 0.05 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_4_0.05_gap20.txt
python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method densesparseuniformquantization --quantize_bit 6 --left 0.05 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_6_0.05_gap20.txt

python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method densesparseuniformquantization --quantize_bit 4 --left 0.02 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_4_0.02_gap20.txt
python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method densesparseuniformquantization --quantize_bit 6 --left 0.02 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_6_0.02_gap20.txt


python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method densesparseuniformquantization --quantize_bit 4 --left 0.01 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_4_0.02_gap20.txt
python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method densesparseuniformquantization --quantize_bit 6 --left 0.01 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_6_0.02_gap20.txt

python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method groupquantization --quantize_bit 4 --streaming --streaming_gap 20 > mistral_mmlucot_groupQ_4_0.02_gap20.txt
python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method groupquantization --quantize_bit 6 --streaming --streaming_gap 20 > mistral_mmlucot_groupQ_6_0.02_gap20.txt

python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method outquantize_with_lrap --quantize_bit 4 --left 0.01  --rank 0.05 --rankv 0.05 --loop 3 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_4_0.01_gap20_lrap_0.05.txt
python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method outquantize_with_lrap --quantize_bit 6 --left 0.01  --rank 0.05 --rankv 0.05 --loop 3 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_6_0.01_gap20_lrap_0.05.txt

python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method outquantize_with_lrap --quantize_bit 4 --left 0.01  --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_4_0.01_gap20_lrap_0.02.txt
python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method outquantize_with_lrap --quantize_bit 6 --left 0.01  --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_6_0.01_gap20_lrap_0.02.txt


python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method outquantize_with_lrap --quantize_bit 4 --left 0.02  --rank 0.05 --rankv 0.05 --loop 3 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_4_0.02_gap20_lrap_0.05.txt
python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method outquantize_with_lrap --quantize_bit 6 --left 0.02  --rank 0.05 --rankv 0.05 --loop 3 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_6_0.02_gap20_lrap_0.05.txt

python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method outquantize_with_lrap --quantize_bit 4 --left 0.02  --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_4_0.02_gap20_lrap_0.02.txt
python eval_mmlu_cot.py --model mistralai/Mistral-7B-v0.1  --batch_size 12 --max_new_tokens 256 --attention_number 40 --root_output_dir outputs  --compress_method outquantize_with_lrap --quantize_bit 6 --left 0.02  --rank 0.02 --rankv 0.02 --loop 3 --streaming --streaming_gap 20 > mistral_mmlucot_outlierQ_6_0.02_gap20_lrap_0.02.txt
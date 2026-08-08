# GATE-1 Baseline Reproduction Log

Please fill in this log file after each training session.

| epoch | train_loss | eval_forget_quality | eval_model_utility | timestamp |
| ----- | ---------- | ------------------- | ------------------ | --------- |
| 1     |            |                     |                    |           |
| 2     |            |                     |                    |           |
| 3     |            |                     |                    |           |
| 4     |            |                     |                    |           |
| 5     |            |                     |                    |           |
| 6     |            |                     |                    |           |
| 7     |            |                     |                    |           |
| 8     |            |                     |                    |           |
| 9     |            |                     |                    |           |
| 10    |            |                     |                    |           |

# Rough data outputs(to be cleaned further):

## Output from unlearning:

### First run:

Found existing checkpoints in Drive. Resuming training...
=== Starting NPO Baseline Training ===
2026-08-04 12:17:55.109160: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
[2026-08-04 12:17:58,779] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2026-08-04 12:18:03,361][model][INFO] - Setting pad_token as eos token: <|eot_id|>
[2026-08-04 12:18:08,011][evaluator][INFO] - Evaluations stored in the experiment directory: ./saves/unlearn/GATE1_BASELINE
[2026-08-04 12:18:09,719][trainer][INFO] - NPO Trainer loaded, output_dir: /content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42
0% 0/60 [00:00<?, ?it/s][2026-08-04 12:18:11,202][evaluator][INFO] - **\*** Running TOFU evaluation suite **\***
[2026-08-04 12:18:11,202][evaluator][INFO] - Fine-grained evaluations will be saved to: /content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42/checkpoint-0/evals/TOFU_EVAL.json
[2026-08-04 12:18:11,202][evaluator][INFO] - Aggregated evaluations will be summarised in: /content/drive./My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42/checkpoint-0/evals/TOFU_SUMMARY.json
[2026-08-04 12:18:12,924][metrics][INFO] - Evaluating forget_Q_A_PARA_Prob

.
.
.
.
.

Evaluated 200 examples
[2026-08-04 12:22:20,374][metrics][INFO] - Evaluating forget_truth_ratio
[2026-08-04 12:22:20,375][evaluator][INFO] - Result for metric forget_truth_ratio: 0.4724131809860188
[2026-08-04 12:22:20,388][metrics][INFO] - Skipping forget_quality's precompute forget_truth_ratio, already evaluated.
[2026-08-04 12:22:20,389][metrics][INFO] - Evaluating forget_quality
[2026-08-04 12:22:20,390][metrics][WARNING] - retain_model_logs not provided in reference_logs, setting forget_quality to None
[2026-08-04 12:22:20,390][evaluator][INFO] - Result for metric forget_quality: None
[2026-08-04 12:22:22,332][metrics][INFO] - Evaluating forget_Q_A_Prob

.
.
.
.
.
.

Evaluated 200 examples
[2026-08-04 12:23:02,169][evaluator][INFO] - Result for metric forget_Q_A_Prob: 0.8855382168292999
[2026-08-04 12:23:04,188][metrics][INFO] - Evaluating forget_Q_A_ROUGE

.
.
.
.
.

Evaluated 400 examples
[2026-08-04 12:29:58,807][metrics][INFO] - Evaluating retain_Q_A_ROUGE

# Execution halted, session stopped automatically.

# Full Cell Output:

## Second Run:

No existing checkpoints found. Starting fresh training...
=== Starting NPO Baseline Training ===
2026-08-04 13:27:18.868081: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
[2026-08-04 13:27:21,965] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
/content/open-unlearning/src/trainer/unlearn/rmu.py:12: SyntaxWarning: invalid escape sequence '\.'
module_regex="model\.layers\.7",
/content/open-unlearning/src/trainer/unlearn/rmu.py:13: SyntaxWarning: invalid escape sequence '\.'
trainable_params_regex=["model\.layers\.(5|6|7)\.mlp\.down_proj\.weight"],
config.json: 100% 927/927 [00:00<00:00, 7.92MB/s]
model.safetensors: 100% 2.47G/2.47G [00:24<00:00, 102MB/s]
generation_config.json: 100% 184/184 [00:00<00:00, 1.16MB/s]
tokenizer_config.json: 54.5kB [00:00, 91.5MB/s]
tokenizer.json: 9.09MB [00:00, 42.5MB/s]
special_tokens_map.json: 100% 296/296 [00:00<00:00, 2.00MB/s]
[2026-08-04 13:27:55,809][model][INFO] - Setting pad_token as eos token: <|eot_id|>
README.md: 4.14kB [00:00, 13.0MB/s]
forget05.json: 56.4kB [00:00, 120MB/s]
Generating train split: 100% 200/200 [00:00<00:00, 8210.60 examples/s]
retain90.json: 963kB [00:00, 205MB/s]
Generating train split: 100% 3600/3600 [00:00<00:00, 455064.48 examples/s]
[2026-08-04 13:27:58,338][evaluator][INFO] - Evaluations stored in the experiment directory: ./saves/unlearn/GATE1_BASELINE
[2026-08-04 13:28:00,621][trainer][INFO] - NPO Trainer loaded, output_dir: /content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42
0% 0/60 [00:00<?, ?it/s][2026-08-04 13:28:02,343][evaluator][INFO] - **\*** Running TOFU evaluation suite **\***
[2026-08-04 13:28:02,343][evaluator][INFO] - Fine-grained evaluations will be saved to: /content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42/checkpoint-0/evals/TOFU_EVAL.json
[2026-08-04 13:28:02,343][evaluator][INFO] - Aggregated evaluations will be summarised in: /content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42/checkpoint-0/evals/TOFU_SUMMARY.json

forget05_perturbed.json: 334kB [00:00, 207MB/s]

Generating train split: 100% 200/200 [00:00<00:00, 55319.23 examples/s]
[2026-08-04 13:28:03,244][metrics][INFO] - Evaluating forget_Q_A_PARA_Prob

Calculating loss: 0% 0/100 [00:00<?, ?it/s]
Calculating loss: 1% 1/100 [00:01<02:02, 1.24s/it]
.
.
.
.
.
.
Calculating loss: 94% 94/100 [00:36<00:02, 2.55it/s]
Calculating loss: 95% 95/100 [00:36<00:01, 2.53it/s]
Calculating loss: 96% 96/100 [00:37<00:01, 2.52it/s]
Calculating loss: 97% 97/100 [00:37<00:01, 2.52it/s]
Calculating loss: 98% 98/100 [00:37<00:00, 2.50it/s]
Calculating loss: 99% 99/100 [00:38<00:00, 2.51it/s]
Calculating loss: 100% 100/100 [00:38<00:00, 2.58it/s]
Evaluated 200 examples
[2026-08-04 13:28:42,802][metrics][INFO] - Evaluating forget_Q_A_PERT_Prob

Calculating loss: 0% 0/100 [00:00<?, ?it/s]
Calculating loss: 1% 1/100 [00:01<02:45, 1.67s/it]
.
.
.
.
.
.
Calculating loss: 94% 94/100 [03:17<00:13, 2.21s/it]
Calculating loss: 95% 95/100 [03:19<00:11, 2.23s/it]
Calculating loss: 96% 96/100 [03:21<00:08, 2.24s/it]
Calculating loss: 97% 97/100 [03:23<00:06, 2.25s/it]
Calculating loss: 98% 98/100 [03:26<00:04, 2.26s/it]
Calculating loss: 99% 99/100 [03:28<00:02, 2.25s/it]
Calculating loss: 100% 100/100 [03:30<00:00, 2.11s/it]
Evaluated 200 examples
[2026-08-04 13:32:13,520][metrics][INFO] - Evaluating forget_truth_ratio
[2026-08-04 13:32:13,521][evaluator][INFO] - Result for metric forget_truth_ratio: 0.4724131809860188
[2026-08-04 13:32:13,549][metrics][INFO] - Skipping forget_quality's precompute forget_truth_ratio, already evaluated.
[2026-08-04 13:32:13,550][metrics][INFO] - Evaluating forget_quality
[2026-08-04 13:32:13,551][metrics][WARNING] - retain_model_logs not provided in reference_logs, setting forget_quality to None
[2026-08-04 13:32:13,551][evaluator][INFO] - Result for metric forget_quality: None
[2026-08-04 13:32:14,262][metrics][INFO] - Evaluating forget_Q_A_Prob

Calculating loss: 0% 0/100 [00:00<?, ?it/s]
Calculating loss: 1% 1/100 [00:00<00:36, 2.69it/s]
Calculating loss: 2% 2/100 [00:00<00:37, 2.65it/s]
.
.
.
.
.
.
Calculating loss: 96% 96/100 [00:40<00:01, 2.28it/s]
Calculating loss: 97% 97/100 [00:40<00:01, 2.27it/s]
Calculating loss: 98% 98/100 [00:41<00:00, 2.26it/s]
Calculating loss: 99% 99/100 [00:41<00:00, 2.25it/s]
Calculating loss: 100% 100/100 [00:42<00:00, 2.38it/s]
Evaluated 200 examples
[2026-08-04 13:32:56,340][evaluator][INFO] - Result for metric forget_Q_A_Prob: 0.8855382168292999
[2026-08-04 13:32:57,043][metrics][INFO] - Evaluating forget_Q_A_ROUGE

Calculating text similarity: 0% 0/100 [00:00<?, ?it/s]
Calculating text similarity: 1% 1/100 [00:02<04:45, 2.88s/it]
Calculating text similarity: 2% 2/100 [00:06<05:30, 3.37s/it]
.
.
.
.
.
.
.
Calculating text similarity: 96% 96/100 [05:55<00:16, 4.09s/it]
Calculating text similarity: 97% 97/100 [05:59<00:11, 3.94s/it]
Calculating text similarity: 98% 98/100 [06:03<00:08, 4.05s/it]
Calculating text similarity: 99% 99/100 [06:07<00:03, 3.94s/it]
Calculating text similarity: 100% 100/100 [06:11<00:00, 3.72s/it]
Evaluated 200 examples
[2026-08-04 13:39:08,904][evaluator][INFO] - Result for metric forget_Q_A_ROUGE: 0.8298115008485689

retain_perturbed.json: 641kB [00:00, 35.1MB/s]

Generating train split: 100% 400/400 [00:00<00:00, 72988.85 examples/s]
[2026-08-04 13:39:09,961][metrics][INFO] - Evaluating retain_Q_A_Prob

Calculating loss: 0% 0/200 [00:00<?, ?it/s]
Calculating loss: 0% 1/200 [00:00<01:14, 2.66it/s]
Calculating loss: 1% 2/200 [00:00<01:22, 2.39it/s]
Calculating loss: 2% 3/200 [00:01<01:25, 2.31it/s]
.
.
.
.
.
.
.
.
Calculating loss: 97% 194/200 [01:18<00:02, 2.57it/s]
Calculating loss: 98% 195/200 [01:18<00:02, 2.46it/s]
Calculating loss: 98% 196/200 [01:19<00:01, 2.51it/s]
Calculating loss: 98% 197/200 [01:19<00:01, 2.54it/s]
Calculating loss: 99% 198/200 [01:19<00:00, 2.57it/s]
Calculating loss: 100% 199/200 [01:20<00:00, 2.59it/s]
Calculating loss: 100% 200/200 [01:20<00:00, 2.48it/s]
Evaluated 400 examples
[2026-08-04 13:40:31,496][metrics][INFO] - Evaluating retain_Q_A_ROUGE

Calculating text similarity: 0% 0/200 [00:00<?, ?it/s]
Calculating text similarity: 0% 1/200 [00:02<09:03, 2.73s/it]
Calculating text similarity: 1% 2/200 [00:06<11:10, 3.38s/it]
.
.
.
.
.
.
.
.
.
.
Calculating text similarity: 96% 193/200 [10:57<00:25, 3.59s/it]
Calculating text similarity: 97% 194/200 [11:00<00:20, 3.41s/it]
Calculating text similarity: 98% 195/200 [11:04<00:17, 3.55s/it]
Calculating text similarity: 98% 196/200 [11:07<00:14, 3.53s/it]
Calculating text similarity: 98% 197/200 [11:10<00:10, 3.44s/it]
Calculating text similarity: 99% 198/200 [11:14<00:06, 3.39s/it]
Calculating text similarity: 100% 199/200 [11:17<00:03, 3.31s/it]
Calculating text similarity: 100% 200/200 [11:21<00:00, 3.41s/it]
Evaluated 400 examples
[2026-08-04 13:51:53,831][metrics][INFO] - Evaluating retain_Q_A_PARA_Prob

Calculating loss: 0% 0/200 [00:00<?, ?it/s]
Calculating loss: 0% 1/200 [00:00<01:14, 2.67it/s]
.
.
.
.
.
.
.
.
.
.
Calculating loss: 96% 192/200 [01:21<00:03, 2.37it/s]
Calculating loss: 96% 193/200 [01:21<00:02, 2.44it/s]
Calculating loss: 97% 194/200 [01:21<00:02, 2.37it/s]
Calculating loss: 98% 195/200 [01:22<00:02, 2.32it/s]
Calculating loss: 98% 196/200 [01:22<00:01, 2.29it/s]
Calculating loss: 98% 197/200 [01:23<00:01, 2.26it/s]
Calculating loss: 99% 198/200 [01:23<00:00, 2.24it/s]
Calculating loss: 100% 199/200 [01:24<00:00, 2.24it/s]
Calculating loss: 100% 200/200 [01:24<00:00, 2.36it/s]
Evaluated 400 examples
[2026-08-04 13:53:19,229][metrics][INFO] - Evaluating retain_Q_A_PERT_Prob

Calculating loss: 0% 0/200 [00:00<?, ?it/s]
Calculating loss: 0% 1/200 [00:01<06:12, 1.87s/it]
Calculating loss: 1% 2/200 [00:04<06:54, 2.09s/it]
Calculating loss: 2% 3/200 [00:06<07:05, 2.16s/it]
Calculating loss: 2% 4/200 [00:08<07:09, 2.19s/it]
Calculating loss: 2% 5/200 [00:10<07:12, 2.22s/it]
.
.
.
.
.
.
.
.
.
Calculating loss: 98% 195/200 [06:47<00:10, 2.09s/it]
Calculating loss: 98% 196/200 [06:50<00:08, 2.13s/it]
Calculating loss: 98% 197/200 [06:52<00:06, 2.16s/it]
Calculating loss: 99% 198/200 [06:54<00:04, 2.19s/it]
Calculating loss: 100% 199/200 [06:56<00:02, 2.22s/it]
Calculating loss: 100% 200/200 [06:59<00:00, 2.10s/it]
Evaluated 400 examples
[2026-08-04 14:00:18,443][metrics][INFO] - Evaluating retain_Truth_Ratio

real_authors_perturbed.json: 17.1kB [00:00, 34.1MB/s]

Generating train split: 100% 100/100 [00:00<00:00, 34949.62 examples/s]
[2026-08-04 14:00:19,539][metrics][INFO] - Evaluating ra_Q_A_Prob

Calculating loss: 0% 0/50 [00:00<?, ?it/s]
Calculating loss: 2% 1/50 [00:00<00:08, 5.71it/s]
Calculating loss: 4% 2/50 [00:00<00:08, 5.51it/s]
Calculating loss: 6% 3/50 [00:00<00:08, 5.54it/s]
Calculating loss: 8% 4/50 [00:00<00:08, 5.51it/s]
.
.
.
.
.
.
.
Calculating loss: 92% 46/50 [00:09<00:00, 4.59it/s]
Calculating loss: 94% 47/50 [00:10<00:00, 3.78it/s]
Calculating loss: 96% 48/50 [00:10<00:00, 3.89it/s]
Calculating loss: 98% 49/50 [00:10<00:00, 4.25it/s]
Calculating loss: 100% 50/50 [00:11<00:00, 4.54it/s]
Evaluated 100 examples
[2026-08-04 14:00:31,216][metrics][INFO] - Evaluating ra_Q_A_PERT_Prob

Calculating loss: 0% 0/50 [00:00<?, ?it/s]
Calculating loss: 2% 1/50 [00:00<00:27, 1.80it/s]
Calculating loss: 4% 2/50 [00:01<00:26, 1.80it/s]
Calculating loss: 6% 3/50 [00:01<00:25, 1.82it/s]
Calculating loss: 8% 4/50 [00:02<00:25, 1.80it/s]
.
.
.
.
.
.
.
.
Calculating loss: 92% 46/50 [00:29<00:02, 1.54it/s]
Calculating loss: 94% 47/50 [00:30<00:02, 1.27it/s]
Calculating loss: 96% 48/50 [00:31<00:01, 1.21it/s]
Calculating loss: 98% 49/50 [00:32<00:00, 1.34it/s]
Calculating loss: 100% 50/50 [00:33<00:00, 1.51it/s]
Evaluated 100 examples
[2026-08-04 14:01:04,406][metrics][INFO] - Evaluating ra_Q_A_Prob_normalised
[2026-08-04 14:01:05,229][metrics][INFO] - Evaluating ra_Q_A_ROUGE

Calculating text similarity: 0% 0/50 [00:00<?, ?it/s]
Calculating text similarity: 2% 1/50 [00:01<01:35, 1.96s/it]
Calculating text similarity: 4% 2/50 [00:04<01:38, 2.05s/it]
Calculating text similarity: 6% 3/50 [00:05<01:30, 1.92s/it]
Calculating text similarity: 8% 4/50 [00:08<01:38, 2.15s/it]
.
.
.
.
.
.
.
Calculating text similarity: 94% 47/50 [01:36<00:06, 2.11s/it]
Calculating text similarity: 96% 48/50 [01:38<00:04, 2.33s/it]
Calculating text similarity: 98% 49/50 [01:40<00:02, 2.19s/it]
Calculating text similarity: 100% 50/50 [01:43<00:00, 2.08s/it]
Evaluated 100 examples
[2026-08-04 14:02:49,171][metrics][INFO] - Skipping ra_Truth_Ratio's precompute ra_Q_A_Prob, already evaluated.
[2026-08-04 14:02:49,172][metrics][INFO] - Skipping ra_Truth_Ratio's precompute ra_Q_A_PERT_Prob, already evaluated.
[2026-08-04 14:02:49,172][metrics][INFO] - Evaluating ra_Truth_Ratio

world_facts_perturbed.json: 18.1kB [00:00, 34.1MB/s]

Generating train split: 100% 117/117 [00:00<00:00, 30119.29 examples/s]
[2026-08-04 14:02:50,221][metrics][INFO] - Evaluating wf_Q_A_Prob

Calculating loss: 0% 0/59 [00:00<?, ?it/s]
Calculating loss: 2% 1/59 [00:00<00:10, 5.69it/s]
Calculating loss: 3% 2/59 [00:00<00:10, 5.57it/s]
.
.
.
.
.
.
.
.
Calculating loss: 97% 57/59 [00:10<00:00, 5.34it/s]
Calculating loss: 98% 58/59 [00:11<00:00, 5.34it/s]
Calculating loss: 100% 59/59 [00:11<00:00, 5.29it/s]
Evaluated 117 examples
[2026-08-04 14:03:02,053][metrics][INFO] - Evaluating wf_Q_A_PERT_Prob

Calculating loss: 0% 0/59 [00:00<?, ?it/s]
Calculating loss: 2% 1/59 [00:00<00:31, 1.84it/s]
Calculating loss: 3% 2/59 [00:01<00:30, 1.84it/s]
.
.
.
.
.
.
.
.
.
Calculating loss: 95% 56/59 [00:32<00:01, 1.65it/s]
Calculating loss: 97% 57/59 [00:32<00:01, 1.70it/s]
Calculating loss: 98% 58/59 [00:33<00:00, 1.72it/s]
Calculating loss: 100% 59/59 [00:33<00:00, 1.75it/s]
Evaluated 117 examples
[2026-08-04 14:03:35,769][metrics][INFO] - Evaluating wf_Q_A_Prob_normalised
[2026-08-04 14:03:36,464][metrics][INFO] - Evaluating wf_Q_A_ROUGE

Calculating text similarity: 0% 0/59 [00:00<?, ?it/s]
Calculating text similarity: 2% 1/59 [00:01<01:20, 1.40s/it]
Calculating text similarity: 3% 2/59 [00:04<02:13, 2.35s/it]
Calculating text similarity: 5% 3/59 [00:06<02:05, 2.24s/it]
.
.
.
.
.
.
.
.
.
Calculating text similarity: 95% 56/59 [02:07<00:07, 2.49s/it]
Calculating text similarity: 97% 57/59 [02:09<00:04, 2.22s/it]
Calculating text similarity: 98% 58/59 [02:12<00:02, 2.51s/it]
Calculating text similarity: 100% 59/59 [02:13<00:00, 2.26s/it]
Evaluated 117 examples
[2026-08-04 14:05:49,567][metrics][INFO] - Skipping wf_Truth_Ratio's precompute wf_Q_A_Prob, already evaluated.
[2026-08-04 14:05:49,567][metrics][INFO] - Skipping wf_Truth_Ratio's precompute wf_Q_A_PERT_Prob, already evaluated.
[2026-08-04 14:05:49,568][metrics][INFO] - Evaluating wf_Truth_Ratio
[2026-08-04 14:05:49,568][metrics][INFO] - Evaluating model_utility
[2026-08-04 14:05:49,569][evaluator][INFO] - Result for metric model_utility: 0.5967054518487185

holdout10.json: 131kB [00:00, 27.0MB/s]

Generating train split: 100% 400/400 [00:00<00:00, 186310.01 examples/s]
[2026-08-04 14:05:51,105][metrics][INFO] - Evaluating mia_min_k

0% 0/100 [00:00<?, ?it/s]
1% 1/100 [00:00<00:22, 4.48it/s]
2% 2/100 [00:00<00:23, 4.25it/s]
3% 3/100 [00:00<00:23, 4.21it/s]
4% 4/100 [00:01<00:26, 3.67it/s]
5% 5/100 [00:01<00:25, 3.80it/s]
6% 6/100 [00:01<00:26, 3.55it/s]
7% 7/100 [00:01<00:25, 3.71it/s]
8% 8/100 [00:02<00:26, 3.52it/s]
9% 9/100 [00:02<00:24, 3.67it/s]
10% 10/100 [00:02<00:25, 3.50it/s]
11% 11/100 [00:02<00:24, 3.65it/s]
12% 12/100 [00:03<00:23, 3.76it/s]
13% 13/100 [00:03<00:22, 3.84it/s]
14% 14/100 [00:03<00:22, 3.89it/s]
15% 15/100 [00:03<00:21, 3.93it/s]
16% 16/100 [00:04<00:21, 3.97it/s]
17% 17/100 [00:04<00:20, 3.98it/s]
.
.
.
.
.
.
.
.
.
.
.
89% 89/100 [00:25<00:03, 3.15it/s]
90% 90/100 [00:25<00:03, 3.14it/s]
91% 91/100 [00:26<00:02, 3.15it/s]
92% 92/100 [00:26<00:02, 3.39it/s]
93% 93/100 [00:26<00:01, 3.58it/s]
94% 94/100 [00:26<00:01, 3.44it/s]
95% 95/100 [00:27<00:01, 3.34it/s]
96% 96/100 [00:27<00:01, 3.29it/s]
97% 97/100 [00:27<00:00, 3.28it/s]
98% 98/100 [00:28<00:00, 3.24it/s]
99% 99/100 [00:28<00:00, 3.23it/s]
100% 100/100 [00:28<00:00, 3.47it/s]

0% 0/200 [00:00<?, ?it/s]
0% 1/200 [00:00<00:46, 4.24it/s]
1% 2/200 [00:00<00:48, 4.12it/s]
2% 3/200 [00:00<00:53, 3.66it/s]
2% 4/200 [00:01<00:51, 3.80it/s]
2% 5/200 [00:01<00:50, 3.87it/s]
3% 6/200 [00:01<00:49, 3.91it/s]
4% 7/200 [00:01<00:48, 3.96it/s]
4% 8/200 [00:02<00:52, 3.65it/s]
4% 9/200 [00:02<00:54, 3.48it/s]
5% 10/200 [00:02<00:56, 3.36it/s]
6% 11/200 [00:02<00:53, 3.56it/s]
6% 12/200 [00:03<00:54, 3.45it/s]
6% 13/200 [00:03<00:56, 3.33it/s]
7% 14/200 [00:04<01:00, 3.08it/s]
8% 15/200 [00:04<00:59, 3.11it/s]
8% 16/200 [00:04<00:59, 3.12it/s]
8% 17/200 [00:04<00:58, 3.13it/s]
9% 18/200 [00:05<00:57, 3.15it/s]
10% 19/200 [00:05<00:57, 3.17it/s]
10% 20/200 [00:05<00:56, 3.16it/s]
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
.
92% 183/200 [00:55<00:04, 3.54it/s]
92% 184/200 [00:56<00:04, 3.39it/s]
92% 185/200 [00:56<00:04, 3.32it/s]
93% 186/200 [00:56<00:04, 3.25it/s]
94% 187/200 [00:57<00:04, 3.22it/s]
94% 188/200 [00:57<00:03, 3.19it/s]
94% 189/200 [00:57<00:03, 3.17it/s]
95% 190/200 [00:58<00:03, 3.17it/s]
96% 191/200 [00:58<00:02, 3.15it/s]
96% 192/200 [00:58<00:02, 3.13it/s]
96% 193/200 [00:59<00:02, 3.11it/s]
97% 194/200 [00:59<00:02, 2.93it/s]
98% 195/200 [00:59<00:01, 2.99it/s]
98% 196/200 [01:00<00:01, 3.01it/s]
98% 197/200 [01:00<00:00, 3.05it/s]
99% 198/200 [01:00<00:00, 3.07it/s]
100% 199/200 [01:01<00:00, 3.09it/s]
100% 200/200 [01:01<00:00, 3.26it/s]
[2026-08-04 14:07:21,373][metrics][INFO] - Evaluating privleak
[2026-08-04 14:07:21,373][metrics][WARNING] - retain_model_logs evals not provided for privleak, using default retain auc of 0.5
[2026-08-04 14:07:21,374][evaluator][INFO] - Result for metric privleak: -99.37499998012501
[2026-08-04 14:07:22,203][metrics][INFO] - Evaluating extraction_strength

Calculating ES: 0% 0/100 [00:00<?, ?it/s]
Calculating ES: 1% 1/100 [00:00<00:22, 4.47it/s]
Calculating ES: 2% 2/100 [00:00<00:23, 4.17it/s]
Calculating ES: 3% 3/100 [00:00<00:23, 4.19it/s]
.
.
.
.
.
.
.
.
.
.
Calculating ES: 90% 90/100 [00:25<00:03, 3.12it/s]
Calculating ES: 91% 91/100 [00:26<00:02, 3.14it/s]
Calculating ES: 92% 92/100 [00:26<00:02, 3.36it/s]
Calculating ES: 93% 93/100 [00:26<00:01, 3.55it/s]
Calculating ES: 94% 94/100 [00:26<00:01, 3.42it/s]
Calculating ES: 95% 95/100 [00:27<00:01, 3.33it/s]
Calculating ES: 96% 96/100 [00:27<00:01, 3.26it/s]
Calculating ES: 97% 97/100 [00:27<00:00, 3.23it/s]
Calculating ES: 98% 98/100 [00:28<00:00, 3.22it/s]
Calculating ES: 99% 99/100 [00:28<00:00, 3.19it/s]
Calculating ES: 100% 100/100 [00:28<00:00, 3.46it/s]
Evaluated 200 examples
[2026-08-04 14:07:51,066][evaluator][INFO] - Result for metric extraction_strength: 0.7246319065737944
{'forget_truth_ratio': np.float64(0.4724131809860188), 'forget_Q_A_Prob': np.float64(0.8855382168292999), 'forget_Q_A_ROUGE': np.float64(0.8298115008485689), 'model_utility': np.float64(0.5967054518487185), 'privleak': np.float64(-99.37499998012501), 'extraction_strength': np.float64(0.7246319065737944), 'epoch': 0}
0% 0/60 [39:48<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
Could not estimate the number of tokens of the input, floating-point operations will not be computed
{'loss': 446.6542, 'grad_norm': 6176.0, 'learning_rate': 0.0, 'epoch': 0.16}
{'loss': 447.4654, 'grad_norm': 6784.0, 'learning_rate': 1.6666666666666667e-06, 'epoch': 0.32}
{'loss': 447.8989, 'grad_norm': 8032.0, 'learning_rate': 3.3333333333333333e-06, 'epoch': 0.48}
{'loss': 445.5749, 'grad_norm': 7648.0, 'learning_rate': 5e-06, 'epoch': 0.64}
{'loss': 443.0435, 'grad_norm': 7168.0, 'learning_rate': 6.666666666666667e-06, 'epoch': 0.8}
{'loss': 440.1309, 'grad_norm': 6144.0, 'learning_rate': 8.333333333333334e-06, 'epoch': 0.96}
{'loss': 111.5688, 'grad_norm': 3808.0, 'learning_rate': 1e-05, 'epoch': 1.0}
12% 7/60 [45:48<1:59:02, 134.77s/it][2026-08-04 14:13:50,637][evaluator][INFO] - **\*** Running TOFU evaluation suite **\***
[2026-08-04 14:13:50,643][evaluator][INFO] - Fine-grained evaluations will be saved to: /content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42/checkpoint-7/evals/TOFU_EVAL.json
[2026-08-04 14:13:50,643][evaluator][INFO] - Aggregated evaluations will be summarised in: /content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42/checkpoint-7/evals/TOFU_SUMMARY.json
[2026-08-04 14:13:51,696][metrics][INFO] - Evaluating forget_Q_A_PARA_Prob

Calculating loss: 0% 0/100 [00:00<?, ?it/s]
Calculating loss: 1% 1/100 [00:00<00:40, 2.44it/s]
Calculating loss: 2% 2/100 [00:00<00:42, 2.31it/s]
Calculating loss: 3% 3/100 [00:01<00:39, 2.47it/s]
Calculating loss: 4% 4/100 [00:01<00:40, 2.35it/s]
Calculating loss: 5% 5/100 [00:02<00:41, 2.31it/s]
Calculating loss: 6% 6/100 [00:02<00:39, 2.41it/s]
Calculating loss: 7% 7/100 [00:02<00:37, 2.46it/s]
Calculating loss: 8% 8/100 [00:03<00:38, 2.39it/s]
Calculating loss: 9% 9/100 [00:03<00:38, 2.34it/s]
.
.
.
.
.
.
.
.
Calculating loss: 90% 90/100 [00:38<00:04, 2.16it/s]
Calculating loss: 91% 91/100 [00:39<00:04, 2.17it/s]
Calculating loss: 92% 92/100 [00:39<00:03, 2.18it/s]
Calculating loss: 93% 93/100 [00:40<00:03, 2.31it/s]
Calculating loss: 94% 94/100 [00:40<00:02, 2.27it/s]
Calculating loss: 95% 95/100 [00:41<00:02, 2.25it/s]
Calculating loss: 96% 96/100 [00:41<00:01, 2.25it/s]
Calculating loss: 97% 97/100 [00:42<00:01, 2.23it/s]
Calculating loss: 98% 98/100 [00:42<00:00, 2.22it/s]
Calculating loss: 99% 99/100 [00:42<00:00, 2.22it/s]
Calculating loss: 100% 100/100 [00:43<00:00, 2.30it/s]
Evaluated 200 examples
[2026-08-04 14:14:36,040][metrics][INFO] - Evaluating forget_Q_A_PERT_Prob

Calculating loss: 0% 0/100 [00:00<?, ?it/s]
Calculating loss: 1% 1/100 [00:01<03:06, 1.88s/it]
Calculating loss: 2% 2/100 [00:04<03:20, 2.05s/it]
Calculating loss: 3% 3/100 [00:05<03:09, 1.96s/it]
.
.
.
.
.
.
.
.
.
.
Calculating loss: 90% 90/100 [03:14<00:23, 2.32s/it]
Calculating loss: 91% 91/100 [03:16<00:20, 2.31s/it]
Calculating loss: 92% 92/100 [03:19<00:18, 2.29s/it]
Calculating loss: 93% 93/100 [03:21<00:15, 2.16s/it]
Calculating loss: 94% 94/100 [03:23<00:13, 2.20s/it]
Calculating loss: 95% 95/100 [03:25<00:11, 2.22s/it]
Calculating loss: 96% 96/100 [03:27<00:08, 2.23s/it]
Calculating loss: 97% 97/100 [03:30<00:06, 2.24s/it]
Calculating loss: 98% 98/100 [03:32<00:04, 2.24s/it]
Calculating loss: 99% 99/100 [03:34<00:02, 2.24s/it]
Calculating loss: 100% 100/100 [03:36<00:00, 2.17s/it]
Evaluated 200 examples
[2026-08-04 14:18:12,979][metrics][INFO] - Evaluating forget_truth_ratio
[2026-08-04 14:18:12,994][evaluator][INFO] - Result for metric forget_truth_ratio: 0.48726341613849156
[2026-08-04 14:18:13,109][metrics][INFO] - Skipping forget_quality's precompute forget_truth_ratio, already evaluated.
[2026-08-04 14:18:13,111][metrics][INFO] - Evaluating forget_quality
[2026-08-04 14:18:13,112][metrics][WARNING] - retain_model_logs not provided in reference_logs, setting forget_quality to None
[2026-08-04 14:18:13,112][evaluator][INFO] - Result for metric forget_quality: None
[2026-08-04 14:18:14,186][metrics][INFO] - Evaluating forget_Q_A_Prob

Calculating loss: 0% 0/100 [00:00<?, ?it/s]
Calculating loss: 1% 1/100 [00:00<00:39, 2.51it/s]
Calculating loss: 2% 2/100 [00:00<00:38, 2.57it/s]
Calculating loss: 3% 3/100 [00:01<00:37, 2.61it/s]
Calculating loss: 4% 4/100 [00:01<00:39, 2.43it/s]
Calculating loss: 5% 5/100 [00:01<00:38, 2.50it/s]
Calculating loss: 6% 6/100 [00:02<00:39, 2.39it/s]
Calculating loss: 7% 7/100 [00:02<00:37, 2.46it/s]
Calculating loss: 8% 8/100 [00:03<00:38, 2.39it/s]
Calculating loss: 9% 9/100 [00:03<00:37, 2.45it/s]
Calculating loss: 10% 10/100 [00:04<00:37, 2.38it/s]
Calculating loss: 11% 11/100 [00:04<00:36, 2.45it/s]
Calculating loss: 12% 12/100 [00:04<00:35, 2.50it/s]
.
.
.
.
.
.
.
Calculating loss: 89% 89/100 [00:37<00:05, 2.20it/s]
Calculating loss: 90% 90/100 [00:37<00:04, 2.19it/s]
Calculating loss: 91% 91/100 [00:38<00:04, 2.21it/s]
Calculating loss: 92% 92/100 [00:38<00:03, 2.32it/s]
Calculating loss: 93% 93/100 [00:39<00:02, 2.41it/s]
Calculating loss: 94% 94/100 [00:39<00:02, 2.35it/s]
Calculating loss: 95% 95/100 [00:40<00:02, 2.31it/s]
Calculating loss: 96% 96/100 [00:40<00:01, 2.28it/s]
Calculating loss: 97% 97/100 [00:40<00:01, 2.26it/s]
Calculating loss: 98% 98/100 [00:41<00:00, 2.25it/s]
Calculating loss: 99% 99/100 [00:41<00:00, 2.25it/s]
Calculating loss: 100% 100/100 [00:42<00:00, 2.36it/s]
Evaluated 200 examples
[2026-08-04 14:18:56,556][evaluator][INFO] - Result for metric forget_Q_A_Prob: 0.8235540720820427
[2026-08-04 14:18:57,491][metrics][INFO] - Evaluating forget_Q_A_ROUGE

Calculating text similarity: 0% 0/100 [00:00<?, ?it/s]
Calculating text similarity: 1% 1/100 [00:02<04:47, 2.90s/it]
Calculating text similarity: 2% 2/100 [00:06<05:08, 3.15s/it]
Calculating text similarity: 3% 3/100 [00:08<04:21, 2.69s/it]
Calculating text similarity: 4% 4/100 [00:14<06:21, 3.98s/it]
Calculating text similarity: 5% 5/100 [00:17<05:51, 3.70s/it]
Calculating text similarity: 6% 6/100 [00:23<06:45, 4.32s/it]
.
.
.
.
.
.
.
.
.
Calculating text similarity: 91% 91/100 [05:34<00:40, 4.46s/it]
Calculating text similarity: 92% 92/100 [05:38<00:33, 4.13s/it]
Calculating text similarity: 93% 93/100 [05:41<00:26, 3.75s/it]
Calculating text similarity: 94% 94/100 [05:44<00:22, 3.72s/it]
Calculating text similarity: 95% 95/100 [05:49<00:19, 3.89s/it]
Calculating text similarity: 96% 96/100 [05:53<00:16, 4.00s/it]
Calculating text similarity: 97% 97/100 [05:57<00:12, 4.10s/it]
Calculating text similarity: 98% 98/100 [06:01<00:08, 4.05s/it]
Calculating text similarity: 99% 99/100 [06:05<00:03, 3.98s/it]
Calculating text similarity: 100% 100/100 [06:08<00:00, 3.69s/it]
Evaluated 200 examples
[2026-08-04 14:25:06,417][evaluator][INFO] - Result for metric forget_Q_A_ROUGE: 0.7334026115182186
[2026-08-04 14:25:07,495][metrics][INFO] - Evaluating retain_Q_A_Prob

Calculating loss: 0% 0/200 [00:00<?, ?it/s]
Calculating loss: 0% 1/200 [00:00<01:23, 2.38it/s]
Calculating loss: 1% 2/200 [00:00<01:26, 2.29it/s]
Calculating loss: 2% 3/200 [00:01<01:27, 2.25it/s]
Calculating loss: 2% 4/200 [00:01<01:27, 2.24it/s]
Calculating loss: 2% 5/200 [00:02<01:27, 2.22it/s]
Calculating loss: 3% 6/200 [00:02<01:22, 2.35it/s]
.
.
.
.
.
.
.
.
.
.
Calculating loss: 93% 186/200 [01:15<00:05, 2.44it/s]
Calculating loss: 94% 187/200 [01:15<00:05, 2.49it/s]
Calculating loss: 94% 188/200 [01:15<00:04, 2.52it/s]
Calculating loss: 94% 189/200 [01:16<00:04, 2.42it/s]
Calculating loss: 95% 190/200 [01:16<00:04, 2.35it/s]
Calculating loss: 96% 191/200 [01:17<00:03, 2.43it/s]
Calculating loss: 96% 192/200 [01:17<00:03, 2.49it/s]
Calculating loss: 96% 193/200 [01:18<00:02, 2.53it/s]
Calculating loss: 97% 194/200 [01:18<00:02, 2.56it/s]
Calculating loss: 98% 195/200 [01:18<00:02, 2.45it/s]
Calculating loss: 98% 196/200 [01:19<00:01, 2.49it/s]
Calculating loss: 98% 197/200 [01:19<00:01, 2.53it/s]
Calculating loss: 99% 198/200 [01:19<00:00, 2.56it/s]
Calculating loss: 100% 199/200 [01:20<00:00, 2.58it/s]
Calculating loss: 100% 200/200 [01:20<00:00, 2.47it/s]
Evaluated 400 examples
[2026-08-04 14:26:29,322][metrics][INFO] - Evaluating retain_Q_A_ROUGE

Calculating text similarity: 0% 0/200 [00:00<?, ?it/s]
Calculating text similarity: 0% 1/200 [00:02<09:49, 2.96s/it]
Calculating text similarity: 1% 2/200 [00:06<11:27, 3.47s/it]
Calculating text similarity: 2% 3/200 [00:10<11:48, 3.60s/it]
Calculating text similarity: 2% 4/200 [00:14<12:37, 3.87s/it]
Calculating text similarity: 2% 5/200 [00:19<12:56, 3.98s/it]
.
.
.
.
.
.
.
.
.
Calculating text similarity: 96% 191/200 [10:44<00:32, 3.66s/it]
Calculating text similarity: 96% 192/200 [10:46<00:26, 3.25s/it]
Calculating text similarity: 96% 193/200 [10:49<00:22, 3.24s/it]
Calculating text similarity: 97% 194/200 [10:52<00:19, 3.17s/it]
Calculating text similarity: 98% 195/200 [10:56<00:16, 3.35s/it]
Calculating text similarity: 98% 196/200 [10:59<00:12, 3.24s/it]
Calculating text similarity: 98% 197/200 [11:02<00:09, 3.21s/it]
Calculating text similarity: 99% 198/200 [11:06<00:06, 3.26s/it]
Calculating text similarity: 100% 199/200 [11:09<00:03, 3.23s/it]
Calculating text similarity: 100% 200/200 [11:14<00:00, 3.37s/it]
Evaluated 400 examples
[2026-08-04 14:37:45,161][metrics][INFO] - Evaluating retain_Q_A_PARA_Prob

Calculating loss: 0% 0/200 [00:00<?, ?it/s]
Calculating loss: 0% 1/200 [00:00<01:22, 2.42it/s]
Calculating loss: 1% 2/200 [00:00<01:25, 2.31it/s]
Calculating loss: 2% 3/200 [00:01<01:26, 2.27it/s]
Calculating loss: 2% 4/200 [00:01<01:27, 2.25it/s]
Calculating loss: 2% 5/200 [00:02<01:27, 2.24it/s]
Calculating loss: 3% 6/200 [00:02<01:22, 2.35it/s]
Calculating loss: 4% 7/200 [00:02<01:19, 2.43it/s]
Calculating loss: 4% 8/200 [00:03<01:21, 2.36it/s]
Calculating loss: 4% 9/200 [00:03<01:22, 2.31it/s]
.
.
.
.
.
.
Calculating loss: 96% 192/200 [01:21<00:03, 2.38it/s]
Calculating loss: 96% 193/200 [01:21<00:02, 2.44it/s]
Calculating loss: 97% 194/200 [01:22<00:02, 2.38it/s]
Calculating loss: 98% 195/200 [01:22<00:02, 2.33it/s]
Calculating loss: 98% 196/200 [01:23<00:01, 2.30it/s]
Calculating loss: 98% 197/200 [01:23<00:01, 2.27it/s]
Calculating loss: 99% 198/200 [01:23<00:00, 2.26it/s]
Calculating loss: 100% 199/200 [01:24<00:00, 2.25it/s]
Calculating loss: 100% 200/200 [01:24<00:00, 2.36it/s]
Evaluated 400 examples
[2026-08-04 14:39:10,752][metrics][INFO] - Evaluating retain_Q_A_PERT_Prob

Calculating loss: 0% 0/200 [00:00<?, ?it/s]
Calculating loss: 0% 1/200 [00:01<06:13, 1.88s/it]
Calculating loss: 1% 2/200 [00:04<06:56, 2.10s/it]
Calculating loss: 2% 3/200 [00:06<07:07, 2.17s/it]
Calculating loss: 2% 4/200 [00:08<07:11, 2.20s/it]
.
.
.
.
.
.
.
.
.
.
.
.
Calculating loss: 96% 191/200 [06:40<00:19, 2.21s/it]
Calculating loss: 96% 192/200 [06:42<00:16, 2.11s/it]
Calculating loss: 96% 193/200 [06:44<00:14, 2.04s/it]
Calculating loss: 97% 194/200 [06:46<00:12, 2.02s/it]
Calculating loss: 98% 195/200 [06:48<00:10, 2.09s/it]
Calculating loss: 98% 196/200 [06:50<00:08, 2.14s/it]
Calculating loss: 98% 197/200 [06:52<00:06, 2.17s/it]
Calculating loss: 99% 198/200 [06:55<00:04, 2.19s/it]
Calculating loss: 100% 199/200 [06:57<00:02, 2.22s/it]
Calculating loss: 100% 200/200 [06:59<00:00, 2.10s/it]
Evaluated 400 examples
[2026-08-04 14:46:10,483][metrics][INFO] - Evaluating retain_Truth_Ratio
[2026-08-04 14:46:11,615][metrics][INFO] - Evaluating ra_Q_A_Prob

Calculating loss: 0% 0/50 [00:00<?, ?it/s]
Calculating loss: 2% 1/50 [00:00<00:08, 5.65it/s]
Calculating loss: 4% 2/50 [00:00<00:08, 5.39it/s]
Calculating loss: 6% 3/50 [00:00<00:08, 5.47it/s]
Calculating loss: 8% 4/50 [00:00<00:08, 5.42it/s]
.
.
.
.
.
Calculating loss: 92% 46/50 [00:09<00:00, 4.64it/s]
Calculating loss: 94% 47/50 [00:10<00:00, 3.80it/s]
Calculating loss: 96% 48/50 [00:10<00:00, 3.89it/s]
Calculating loss: 98% 49/50 [00:10<00:00, 4.24it/s]
Calculating loss: 100% 50/50 [00:11<00:00, 4.52it/s]
Evaluated 100 examples
[2026-08-04 14:46:23,374][metrics][INFO] - Evaluating ra_Q_A_PERT_Prob

Calculating loss: 0% 0/50 [00:00<?, ?it/s]
Calculating loss: 2% 1/50 [00:00<00:26, 1.83it/s]
Calculating loss: 4% 2/50 [00:01<00:26, 1.81it/s]
Calculating loss: 6% 3/50 [00:01<00:25, 1.83it/s]
Calculating loss: 8% 4/50 [00:02<00:25, 1.81it/s]
Calculating loss: 10% 5/50 [00:02<00:24, 1.81it/s]
.
.
.
.
.
.
Calculating loss: 88% 44/50 [00:28<00:04, 1.31it/s]
Calculating loss: 90% 45/50 [00:28<00:03, 1.43it/s]
Calculating loss: 92% 46/50 [00:29<00:02, 1.54it/s]
Calculating loss: 94% 47/50 [00:30<00:02, 1.27it/s]
Calculating loss: 96% 48/50 [00:31<00:01, 1.21it/s]
Calculating loss: 98% 49/50 [00:32<00:00, 1.34it/s]
Calculating loss: 100% 50/50 [00:33<00:00, 1.51it/s]
Evaluated 100 examples
[2026-08-04 14:46:56,586][metrics][INFO] - Evaluating ra_Q_A_Prob_normalised
[2026-08-04 14:46:57,533][metrics][INFO] - Evaluating ra_Q_A_ROUGE

Calculating text similarity: 0% 0/50 [00:00<?, ?it/s]
Calculating text similarity: 2% 1/50 [00:02<01:40, 2.06s/it]
Calculating text similarity: 4% 2/50 [00:04<01:40, 2.10s/it]
Calculating text similarity: 6% 3/50 [00:05<01:31, 1.94s/it]
Calculating text similarity: 8% 4/50 [00:08<01:40, 2.19s/it]
Calculating text similarity: 10% 5/50 [00:10<01:30, 2.00s/it]
Calculating text similarity: 12% 6/50 [00:12<01:28, 2.01s/it]
Calculating text similarity: 14% 7/50 [00:14<01:25, 1.99s/it]
.
.
.
.
.
.
Calculating text similarity: 92% 46/50 [01:30<00:07, 1.86s/it]
Calculating text similarity: 94% 47/50 [01:33<00:06, 2.11s/it]
Calculating text similarity: 96% 48/50 [01:35<00:04, 2.06s/it]
Calculating text similarity: 98% 49/50 [01:37<00:01, 1.97s/it]
Calculating text similarity: 100% 50/50 [01:39<00:00, 1.99s/it]
Evaluated 100 examples
[2026-08-04 14:48:36,835][metrics][INFO] - Skipping ra_Truth_Ratio's precompute ra_Q_A_Prob, already evaluated.
[2026-08-04 14:48:36,839][metrics][INFO] - Skipping ra_Truth_Ratio's precompute ra_Q_A_PERT_Prob, already evaluated.
[2026-08-04 14:48:36,839][metrics][INFO] - Evaluating ra_Truth_Ratio
[2026-08-04 14:48:37,591][metrics][INFO] - Evaluating wf_Q_A_Prob

Calculating loss: 0% 0/59 [00:00<?, ?it/s]
Calculating loss: 2% 1/59 [00:00<00:10, 5.62it/s]
Calculating loss: 3% 2/59 [00:00<00:10, 5.59it/s]
.
.
.
.
.
.
.
Calculating loss: 90% 53/59 [00:10<00:01, 5.24it/s]
Calculating loss: 92% 54/59 [00:10<00:00, 5.25it/s]
Calculating loss: 93% 55/59 [00:10<00:00, 5.30it/s]
Calculating loss: 95% 56/59 [00:10<00:00, 5.29it/s]
Calculating loss: 97% 57/59 [00:10<00:00, 5.27it/s]
Calculating loss: 98% 58/59 [00:11<00:00, 5.27it/s]
Calculating loss: 100% 59/59 [00:11<00:00, 5.28it/s]
Evaluated 117 examples
[2026-08-04 14:48:49,484][metrics][INFO] - Evaluating wf_Q_A_PERT_Prob

Calculating loss: 0% 0/59 [00:00<?, ?it/s]
Calculating loss: 2% 1/59 [00:00<00:31, 1.83it/s]
Calculating loss: 3% 2/59 [00:01<00:31, 1.83it/s]
Calculating loss: 5% 3/59 [00:01<00:30, 1.83it/s]
Calculating loss: 7% 4/59 [00:02<00:30, 1.83it/s]
Calculating loss: 8% 5/59 [00:02<00:29, 1.83it/s]
.
.
.
.
.
.
.
Calculating loss: 95% 56/59 [00:32<00:01, 1.64it/s]
Calculating loss: 97% 57/59 [00:32<00:01, 1.68it/s]
Calculating loss: 98% 58/59 [00:33<00:00, 1.72it/s]
Calculating loss: 100% 59/59 [00:33<00:00, 1.74it/s]
Evaluated 117 examples
[2026-08-04 14:49:23,303][metrics][INFO] - Evaluating wf_Q_A_Prob_normalised
[2026-08-04 14:49:24,141][metrics][INFO] - Evaluating wf_Q_A_ROUGE

Calculating text similarity: 0% 0/59 [00:00<?, ?it/s]
Calculating text similarity: 2% 1/59 [00:01<01:20, 1.39s/it]
Calculating text similarity: 3% 2/59 [00:04<02:28, 2.61s/it]
.
.
.
.
.
.
.
Calculating text similarity: 93% 55/59 [02:04<00:11, 2.86s/it]
Calculating text similarity: 95% 56/59 [02:05<00:07, 2.45s/it]
Calculating text similarity: 97% 57/59 [02:07<00:04, 2.16s/it]
Calculating text similarity: 98% 58/59 [02:10<00:02, 2.45s/it]
Calculating text similarity: 100% 59/59 [02:10<00:00, 2.21s/it]
Evaluated 117 examples
[2026-08-04 14:51:34,756][metrics][INFO] - Skipping wf_Truth_Ratio's precompute wf_Q_A_Prob, already evaluated.
[2026-08-04 14:51:34,756][metrics][INFO] - Skipping wf_Truth_Ratio's precompute wf_Q_A_PERT_Prob, already evaluated.
[2026-08-04 14:51:34,756][metrics][INFO] - Evaluating wf_Truth_Ratio
[2026-08-04 14:51:34,757][metrics][INFO] - Evaluating model_utility
[2026-08-04 14:51:34,787][evaluator][INFO] - Result for metric model_utility: 0.5963362552918083
[2026-08-04 14:51:36,168][metrics][INFO] - Evaluating mia_min_k

0% 0/100 [00:00<?, ?it/s]
1% 1/100 [00:00<00:23, 4.22it/s]
2% 2/100 [00:00<00:23, 4.15it/s]
3% 3/100 [00:00<00:23, 4.12it/s]
4% 4/100 [00:01<00:26, 3.64it/s]
5% 5/100 [00:01<00:25, 3.76it/s]
6% 6/100 [00:01<00:26, 3.53it/s]
7% 7/100 [00:01<00:25, 3.69it/s]
8% 8/100 [00:02<00:26, 3.52it/s]
9% 9/100 [00:02<00:24, 3.66it/s]
10% 10/100 [00:02<00:25, 3.48it/s]
11% 11/100 [00:02<00:24, 3.63it/s]
.
.
.
.
.
94% 94/100 [00:26<00:01, 3.41it/s]
95% 95/100 [00:27<00:01, 3.32it/s]
96% 96/100 [00:27<00:01, 3.28it/s]
97% 97/100 [00:27<00:00, 3.26it/s]
98% 98/100 [00:28<00:00, 3.23it/s]
99% 99/100 [00:28<00:00, 3.22it/s]
100% 100/100 [00:28<00:00, 3.47it/s]

0% 0/200 [00:00<?, ?it/s]
0% 1/200 [00:00<00:47, 4.23it/s]
1% 2/200 [00:00<00:47, 4.16it/s]
2% 3/200 [00:00<00:53, 3.65it/s]
2% 4/200 [00:01<00:51, 3.78it/s]
2% 5/200 [00:01<00:50, 3.87it/s]
3% 6/200 [00:01<00:49, 3.93it/s]
4% 7/200 [00:01<00:48, 3.96it/s]
4% 8/200 [00:02<00:52, 3.66it/s]
4% 9/200 [00:02<00:54, 3.49it/s]
5% 10/200 [00:02<00:56, 3.36it/s]
.
.
.
.
.
.
.
.
.
96% 191/200 [00:58<00:02, 3.14it/s]
96% 192/200 [00:58<00:02, 3.13it/s]
96% 193/200 [00:59<00:02, 3.10it/s]
97% 194/200 [00:59<00:02, 2.93it/s]
98% 195/200 [00:59<00:01, 2.98it/s]
98% 196/200 [01:00<00:01, 3.01it/s]
98% 197/200 [01:00<00:00, 3.05it/s]
99% 198/200 [01:00<00:00, 3.07it/s]
100% 199/200 [01:01<00:00, 3.10it/s]
100% 200/200 [01:01<00:00, 3.25it/s]
[2026-08-04 14:53:06,535][metrics][INFO] - Evaluating privleak
[2026-08-04 14:53:06,538][metrics][WARNING] - retain_model_logs evals not provided for privleak, using default retain auc of 0.5
[2026-08-04 14:53:06,538][evaluator][INFO] - Result for metric privleak: -99.1024999801795
[2026-08-04 14:53:07,506][metrics][INFO] - Evaluating extraction_strength

Calculating ES: 0% 0/100 [00:00<?, ?it/s]
Calculating ES: 1% 1/100 [00:00<00:27, 3.66it/s]
Calculating ES: 2% 2/100 [00:00<00:24, 3.97it/s]
Calculating ES: 3% 3/100 [00:00<00:23, 4.08it/s]
Calculating ES: 4% 4/100 [00:01<00:26, 3.62it/s]
Calculating ES: 5% 5/100 [00:01<00:25, 3.76it/s]
Calculating ES: 6% 6/100 [00:01<00:26, 3.54it/s]
Calculating ES: 7% 7/100 [00:01<00:25, 3.70it/s]
Calculating ES: 8% 8/100 [00:02<00:26, 3.52it/s]
Calculating ES: 9% 9/100 [00:02<00:24, 3.68it/s]
Calculating ES: 10% 10/100 [00:02<00:25, 3.50it/s]
Calculating ES: 11% 11/100 [00:02<00:24, 3.66it/s]
Calculating ES: 12% 12/100 [00:03<00:23, 3.78it/s]
.
.
.
.
.
.
.
Calculating ES: 93% 93/100 [00:26<00:01, 3.53it/s]
Calculating ES: 94% 94/100 [00:27<00:01, 3.40it/s]
Calculating ES: 95% 95/100 [00:27<00:01, 3.32it/s]
Calculating ES: 96% 96/100 [00:27<00:01, 3.23it/s]
Calculating ES: 97% 97/100 [00:28<00:00, 3.22it/s]
Calculating ES: 98% 98/100 [00:28<00:00, 3.19it/s]
Calculating ES: 99% 99/100 [00:28<00:00, 3.19it/s]
Calculating ES: 100% 100/100 [00:29<00:00, 3.45it/s]
Evaluated 200 examples
[2026-08-04 14:53:36,526][evaluator][INFO] - Result for metric extraction_strength: 0.5665158436006115
{'forget_truth_ratio': np.float64(0.48726341613849156), 'forget_Q_A_Prob': np.float64(0.8235540720820427), 'forget_Q_A_ROUGE': np.float64(0.7334026115182186), 'model_utility': np.float64(0.5963362552918083), 'privleak': np.float64(-99.1024999801795), 'extraction_strength': np.float64(0.5665158436006115), 'epoch': 1.0}
12% 7/60 [1:25:34<1:59:02, 134.77s/it]^C

# Execution halted, session stopped automatically.

# Full Cell Output:

## Third Run:

Found existing checkpoints in Drive. Resuming training...
=== Starting NPO Baseline Training ===
2026-08-04 15:30:47.034631: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
[2026-08-04 15:30:53,324] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2026-08-04 15:30:59,751][model][INFO] - Setting pad_token as eos token: <|eot_id|>
[2026-08-04 15:31:01,412][evaluator][INFO] - Evaluations stored in the experiment directory: ./saves/unlearn/GATE1_BASELINE
[2026-08-04 15:31:11,412][trainer][INFO] - NPO Trainer loaded, output_dir: /content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42
0% 0/60 [00:00<?, ?it/s][2026-08-04 15:31:13,093][evaluator][INFO] - **\*** Running TOFU evaluation suite **\***
[2026-08-04 15:31:13,094][evaluator][INFO] - Fine-grained evaluations will be saved to: /content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42/checkpoint-0/evals/TOFU_EVAL.json
[2026-08-04 15:31:13,094][evaluator][INFO] - Aggregated evaluations will be summarised in: /content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42/checkpoint-0/evals/TOFU_SUMMARY.json
[2026-08-04 15:31:14,014][metrics][INFO] - Evaluating forget_Q_A_PARA_Prob

Calculating loss: 0% 0/100 [00:00<?, ?it/s]
Calculating loss: 1% 1/100 [00:01<02:04, 1.26s/it]
.
.
.
.
.
.
Calculating loss: 95% 95/100 [00:36<00:01, 2.55it/s]
Calculating loss: 96% 96/100 [00:36<00:01, 2.55it/s]
Calculating loss: 97% 97/100 [00:37<00:01, 2.54it/s]
Calculating loss: 98% 98/100 [00:37<00:00, 2.53it/s]
Calculating loss: 99% 99/100 [00:38<00:00, 2.54it/s]
Calculating loss: 100% 100/100 [00:38<00:00, 2.60it/s]
Evaluated 200 examples
[2026-08-04 15:31:53,277][metrics][INFO] - Evaluating forget_Q_A_PERT_Prob

Calculating loss: 0% 0/100 [00:00<?, ?it/s]
Calculating loss: 1% 1/100 [00:01<02:43, 1.66s/it]
Calculating loss: 2% 2/100 [00:03<02:56, 1.80s/it]
Calculating loss: 3% 3/100 [00:05<02:47, 1.73s/it]
Calculating loss: 4% 4/100 [00:07<02:56, 1.84s/it]
.
.
.
.
.
.
.
.
Calculating loss: 97% 97/100 [03:22<00:06, 2.24s/it]
Calculating loss: 98% 98/100 [03:24<00:04, 2.25s/it]
Calculating loss: 99% 99/100 [03:26<00:02, 2.25s/it]
Calculating loss: 100% 100/100 [03:29<00:00, 2.09s/it]
Evaluated 200 examples
[2026-08-04 15:35:22,540][metrics][INFO] - Evaluating forget_truth_ratio
[2026-08-04 15:35:22,541][evaluator][INFO] - Result for metric forget_truth_ratio: 0.4724131809860188
[2026-08-04 15:35:22,570][metrics][INFO] - Skipping forget_quality's precompute forget_truth_ratio, already evaluated.
[2026-08-04 15:35:22,571][metrics][INFO] - Evaluating forget_quality
[2026-08-04 15:35:22,571][metrics][WARNING] - retain_model_logs not provided in reference_logs, setting forget_quality to None
[2026-08-04 15:35:22,571][evaluator][INFO] - Result for metric forget_quality: None
[2026-08-04 15:35:23,346][metrics][INFO] - Evaluating forget_Q_A_Prob

Calculating loss: 0% 0/100 [00:00<?, ?it/s]
Calculating loss: 1% 1/100 [00:00<00:36, 2.68it/s]
Calculating loss: 2% 2/100 [00:00<00:37, 2.65it/s]
Calculating loss: 3% 3/100 [00:01<00:36, 2.66it/s]
Calculating loss: 4% 4/100 [00:01<00:39, 2.44it/s]
Calculating loss: 5% 5/100 [00:01<00:38, 2.50it/s]
.
.
.
.
.
.
.
.
.
Calculating loss: 96% 96/100 [00:40<00:01, 2.28it/s]
Calculating loss: 97% 97/100 [00:40<00:01, 2.27it/s]
Calculating loss: 98% 98/100 [00:41<00:00, 2.25it/s]
Calculating loss: 99% 99/100 [00:41<00:00, 2.25it/s]
Calculating loss: 100% 100/100 [00:42<00:00, 2.37it/s]
Evaluated 200 examples
[2026-08-04 15:36:05,522][evaluator][INFO] - Result for metric forget_Q_A_Prob: 0.8855382168292999
[2026-08-04 15:36:06,358][metrics][INFO] - Evaluating forget_Q_A_ROUGE

Calculating text similarity: 0% 0/100 [00:00<?, ?it/s]
Calculating text similarity: 1% 1/100 [00:02<04:44, 2.88s/it]
Calculating text similarity: 2% 2/100 [00:06<05:30, 3.37s/it]
Calculating text similarity: 3% 3/100 [00:08<04:31, 2.80s/it]
Calculating text similarity: 4% 4/100 [00:14<06:26, 4.02s/it]
Calculating text similarity: 5% 5/100 [00:17<05:54, 3.73s/it]
.
.
.
.
.
.
.
Calculating text similarity: 97% 97/100 [05:58<00:11, 3.93s/it]
Calculating text similarity: 98% 98/100 [06:03<00:08, 4.04s/it]
Calculating text similarity: 99% 99/100 [06:06<00:03, 3.93s/it]
Calculating text similarity: 100% 100/100 [06:11<00:00, 3.71s/it]
Evaluated 200 examples
[2026-08-04 15:42:17,506][evaluator][INFO] - Result for metric forget_Q_A_ROUGE: 0.8298115008485689
[2026-08-04 15:42:18,366][metrics][INFO] - Evaluating retain_Q_A_Prob

Calculating loss: 0% 0/200 [00:00<?, ?it/s]
Calculating loss: 0% 1/200 [00:00<01:15, 2.64it/s]
Calculating loss: 1% 2/200 [00:00<01:22, 2.39it/s]
Calculating loss: 2% 3/200 [00:01<01:24, 2.32it/s]
.
.
.
.
.
.
Calculating loss: 98% 197/200 [01:19<00:01, 2.54it/s]
Calculating loss: 99% 198/200 [01:19<00:00, 2.57it/s]
Calculating loss: 100% 199/200 [01:20<00:00, 2.59it/s]
Calculating loss: 100% 200/200 [01:20<00:00, 2.48it/s]
Evaluated 400 examples
[2026-08-04 15:43:39,591][metrics][INFO] - Evaluating retain_Q_A_ROUGE

Calculating text similarity: 0% 0/200 [00:00<?, ?it/s]
Calculating text similarity: 0% 1/200 [00:02<09:05, 2.74s/it]
Calculating text similarity: 1% 2/200 [00:06<11:08, 3.38s/it]
Calculating text similarity: 2% 3/200 [00:10<11:16, 3.43s/it]
.
.
.
.
.
.
Calculating text similarity: 97% 194/200 [10:58<00:20, 3.40s/it]
Calculating text similarity: 98% 195/200 [11:02<00:17, 3.53s/it]
Calculating text similarity: 98% 196/200 [11:05<00:14, 3.51s/it]
Calculating text similarity: 98% 197/200 [11:08<00:10, 3.42s/it]
Calculating text similarity: 99% 198/200 [11:12<00:06, 3.39s/it]
Calculating text similarity: 100% 199/200 [11:15<00:03, 3.31s/it]
Calculating text similarity: 100% 200/200 [11:19<00:00, 3.40s/it]
Evaluated 400 examples
[2026-08-04 15:54:59,968][metrics][INFO] - Evaluating retain_Q_A_PARA_Prob

Calculating loss: 0% 0/200 [00:00<?, ?it/s]
Calculating loss: 0% 1/200 [00:00<01:14, 2.65it/s]
Calculating loss: 1% 2/200 [00:00<01:22, 2.39it/s]
Calculating loss: 2% 3/200 [00:01<01:25, 2.31it/s]
.
.
.
.
.
.
.
.

Calculating loss: 98% 196/200 [01:22<00:01, 2.29it/s]
Calculating loss: 98% 197/200 [01:23<00:01, 2.27it/s]
Calculating loss: 99% 198/200 [01:23<00:00, 2.26it/s]
Calculating loss: 100% 199/200 [01:24<00:00, 2.26it/s]
Calculating loss: 100% 200/200 [01:24<00:00, 2.36it/s]
Evaluated 400 examples
[2026-08-04 15:56:25,343][metrics][INFO] - Evaluating retain_Q_A_PERT_Prob

Calculating loss: 0% 0/200 [00:00<?, ?it/s]
Calculating loss: 0% 1/200 [00:01<06:12, 1.87s/it]
Calculating loss: 1% 2/200 [00:04<06:54, 2.09s/it]
.
.
.
.
.
.
.
Calculating loss: 98% 195/200 [06:47<00:10, 2.09s/it]
Calculating loss: 98% 196/200 [06:50<00:08, 2.12s/it]
Calculating loss: 98% 197/200 [06:52<00:06, 2.16s/it]
Calculating loss: 99% 198/200 [06:54<00:04, 2.18s/it]
Calculating loss: 100% 199/200 [06:56<00:02, 2.21s/it]
Calculating loss: 100% 200/200 [06:59<00:00, 2.10s/it]
Evaluated 400 examples
[2026-08-04 16:03:24,358][metrics][INFO] - Evaluating retain_Truth_Ratio
[2026-08-04 16:03:25,135][metrics][INFO] - Evaluating ra_Q_A_Prob

Calculating loss: 0% 0/50 [00:00<?, ?it/s]
Calculating loss: 2% 1/50 [00:00<00:08, 5.82it/s]
Calculating loss: 4% 2/50 [00:00<00:08, 5.52it/s]
Calculating loss: 6% 3/50 [00:00<00:08, 5.53it/s]
Calculating loss: 8% 4/50 [00:00<00:08, 5.42it/s]
.
.
.
.
Calculating loss: 94% 47/50 [00:10<00:00, 3.77it/s]
Calculating loss: 96% 48/50 [00:10<00:00, 3.87it/s]
Calculating loss: 98% 49/50 [00:10<00:00, 4.22it/s]
Calculating loss: 100% 50/50 [00:11<00:00, 4.53it/s]
Evaluated 100 examples
[2026-08-04 16:03:36,887][metrics][INFO] - Evaluating ra_Q_A_PERT_Prob

Calculating loss: 0% 0/50 [00:00<?, ?it/s]
Calculating loss: 2% 1/50 [00:00<00:26, 1.85it/s]
Calculating loss: 4% 2/50 [00:01<00:26, 1.81it/s]
Calculating loss: 6% 3/50 [00:01<00:25, 1.82it/s]
.
.
.
.
Calculating loss: 96% 48/50 [00:31<00:01, 1.21it/s]
Calculating loss: 98% 49/50 [00:31<00:00, 1.35it/s]
Calculating loss: 100% 50/50 [00:33<00:00, 1.51it/s]
Evaluated 100 examples
[2026-08-04 16:04:09,986][metrics][INFO] - Evaluating ra_Q_A_Prob_normalised
[2026-08-04 16:04:10,678][metrics][INFO] - Evaluating ra_Q_A_ROUGE

Calculating text similarity: 0% 0/50 [00:00<?, ?it/s]
Calculating text similarity: 2% 1/50 [00:01<01:34, 1.92s/it]
Calculating text similarity: 4% 2/50 [00:04<01:38, 2.05s/it]
.
.
.
.
.
Calculating text similarity: 90% 45/50 [01:31<00:09, 1.95s/it]
Calculating text similarity: 92% 46/50 [01:33<00:07, 1.81s/it]
Calculating text similarity: 94% 47/50 [01:35<00:06, 2.09s/it]
Calculating text similarity: 96% 48/50 [01:38<00:04, 2.31s/it]
Calculating text similarity: 98% 49/50 [01:40<00:02, 2.18s/it]
Calculating text similarity: 100% 50/50 [01:43<00:00, 2.07s/it]
Evaluated 100 examples
[2026-08-04 16:05:54,271][metrics][INFO] - Skipping ra_Truth_Ratio's precompute ra_Q_A_Prob, already evaluated.
[2026-08-04 16:05:54,271][metrics][INFO] - Skipping ra_Truth_Ratio's precompute ra_Q_A_PERT_Prob, already evaluated.
[2026-08-04 16:05:54,271][metrics][INFO] - Evaluating ra_Truth_Ratio
[2026-08-04 16:05:54,981][metrics][INFO] - Evaluating wf_Q_A_Prob

Calculating loss: 0% 0/59 [00:00<?, ?it/s]
Calculating loss: 2% 1/59 [00:00<00:10, 5.75it/s]
Calculating loss: 3% 2/59 [00:00<00:10, 5.67it/s]
.
.
.
.
.
Calculating loss: 95% 56/59 [00:10<00:00, 5.32it/s]
Calculating loss: 97% 57/59 [00:10<00:00, 5.33it/s]
Calculating loss: 98% 58/59 [00:10<00:00, 5.38it/s]
Calculating loss: 100% 59/59 [00:11<00:00, 5.31it/s]
Evaluated 117 examples
[2026-08-04 16:06:06,774][metrics][INFO] - Evaluating wf_Q_A_PERT_Prob

Calculating loss: 0% 0/59 [00:00<?, ?it/s]
Calculating loss: 2% 1/59 [00:00<00:31, 1.84it/s]
Calculating loss: 3% 2/59 [00:01<00:31, 1.84it/s]
.
.
.
.
.
Calculating loss: 95% 56/59 [00:32<00:01, 1.64it/s]
Calculating loss: 97% 57/59 [00:32<00:01, 1.68it/s]
Calculating loss: 98% 58/59 [00:33<00:00, 1.70it/s]
Calculating loss: 100% 59/59 [00:33<00:00, 1.75it/s]
Evaluated 117 examples
[2026-08-04 16:06:40,498][metrics][INFO] - Evaluating wf_Q_A_Prob_normalised
[2026-08-04 16:06:41,195][metrics][INFO] - Evaluating wf_Q_A_ROUGE

Calculating text similarity: 0% 0/59 [00:00<?, ?it/s]
Calculating text similarity: 2% 1/59 [00:01<01:20, 1.40s/it]
Calculating text similarity: 3% 2/59 [00:04<02:14, 2.35s/it]
Calculating text similarity: 5% 3/59 [00:06<02:05, 2.24s/it]
.
.
.
.
.
Calculating text similarity: 93% 55/59 [02:06<00:11, 2.89s/it]
Calculating text similarity: 95% 56/59 [02:07<00:07, 2.47s/it]
Calculating text similarity: 97% 57/59 [02:09<00:04, 2.21s/it]
Calculating text similarity: 98% 58/59 [02:12<00:02, 2.50s/it]
Calculating text similarity: 100% 59/59 [02:12<00:00, 2.25s/it]
Evaluated 117 examples
[2026-08-04 16:08:54,147][metrics][INFO] - Skipping wf_Truth_Ratio's precompute wf_Q_A_Prob, already evaluated.
[2026-08-04 16:08:54,147][metrics][INFO] - Skipping wf_Truth_Ratio's precompute wf_Q_A_PERT_Prob, already evaluated.
[2026-08-04 16:08:54,147][metrics][INFO] - Evaluating wf_Truth_Ratio
[2026-08-04 16:08:54,148][metrics][INFO] - Evaluating model_utility
[2026-08-04 16:08:54,151][evaluator][INFO] - Result for metric model_utility: 0.5967054518487185
[2026-08-04 16:08:55,568][metrics][INFO] - Evaluating mia_min_k

0% 0/100 [00:00<?, ?it/s]
1% 1/100 [00:00<00:22, 4.43it/s]
2% 2/100 [00:00<00:23, 4.23it/s]
3% 3/100 [00:00<00:23, 4.21it/s]
4% 4/100 [00:01<00:25, 3.69it/s]
.
.
.
.
.
.
.
95% 95/100 [00:27<00:01, 3.34it/s]
96% 96/100 [00:27<00:01, 3.28it/s]
97% 97/100 [00:27<00:00, 3.24it/s]
98% 98/100 [00:28<00:00, 3.23it/s]
99% 99/100 [00:28<00:00, 3.20it/s]
100% 100/100 [00:28<00:00, 3.47it/s]

0% 0/200 [00:00<?, ?it/s]
0% 1/200 [00:00<00:46, 4.24it/s]
1% 2/200 [00:00<00:47, 4.16it/s]
2% 3/200 [00:00<00:54, 3.64it/s]
2% 4/200 [00:01<00:51, 3.79it/s]
2% 5/200 [00:01<00:50, 3.85it/s]
3% 6/200 [00:01<00:49, 3.92it/s]
4% 7/200 [00:01<00:48, 3.97it/s]
4% 8/200 [00:02<00:52, 3.66it/s]
4% 9/200 [00:02<00:54, 3.50it/s]
5% 10/200 [00:02<00:56, 3.35it/s]
6% 11/200 [00:02<00:53, 3.55it/s]
6% 12/200 [00:03<00:54, 3.44it/s]
.
.
.
.
.
.
96% 192/200 [00:58<00:02, 3.14it/s]
96% 193/200 [00:59<00:02, 3.12it/s]
97% 194/200 [00:59<00:02, 2.94it/s]
98% 195/200 [00:59<00:01, 2.99it/s]
98% 196/200 [01:00<00:01, 3.02it/s]
98% 197/200 [01:00<00:00, 3.06it/s]
99% 198/200 [01:00<00:00, 3.08it/s]
100% 199/200 [01:01<00:00, 3.10it/s]
100% 200/200 [01:01<00:00, 3.24it/s]
[2026-08-04 16:10:26,007][metrics][INFO] - Evaluating privleak
[2026-08-04 16:10:26,007][metrics][WARNING] - retain_model_logs evals not provided for privleak, using default retain auc of 0.5
[2026-08-04 16:10:26,007][evaluator][INFO] - Result for metric privleak: -99.37499998012501
[2026-08-04 16:10:26,820][metrics][INFO] - Evaluating extraction_strength

Calculating ES: 0% 0/100 [00:00<?, ?it/s]
Calculating ES: 1% 1/100 [00:00<00:22, 4.44it/s]
Calculating ES: 2% 2/100 [00:00<00:23, 4.15it/s]
Calculating ES: 3% 3/100 [00:00<00:23, 4.17it/s]
Calculating ES: 4% 4/100 [00:01<00:26, 3.67it/s]
.
.
.
.
.
.
Calculating ES: 94% 94/100 [00:26<00:01, 3.44it/s]
Calculating ES: 95% 95/100 [00:27<00:01, 3.35it/s]
Calculating ES: 96% 96/100 [00:27<00:01, 3.28it/s]
Calculating ES: 97% 97/100 [00:27<00:00, 3.25it/s]
Calculating ES: 98% 98/100 [00:28<00:00, 3.23it/s]
Calculating ES: 99% 99/100 [00:28<00:00, 3.19it/s]
Calculating ES: 100% 100/100 [00:28<00:00, 3.47it/s]
Evaluated 200 examples
[2026-08-04 16:10:55,621][evaluator][INFO] - Result for metric extraction_strength: 0.7246319065737944
{'forget_truth_ratio': np.float64(0.4724131809860188), 'forget_Q_A_Prob': np.float64(0.8855382168292999), 'forget_Q_A_ROUGE': np.float64(0.8298115008485689), 'model_utility': np.float64(0.5967054518487185), 'privleak': np.float64(-99.37499998012501), 'extraction_strength': np.float64(0.7246319065737944), 'epoch': 0}
0% 0/60 [39:42<?, ?it/s]`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
Could not estimate the number of tokens of the input, floating-point operations will not be computed
{'loss': 446.6542, 'grad_norm': 6176.0, 'learning_rate': 0.0, 'epoch': 0.16}
{'loss': 447.4654, 'grad_norm': 6784.0, 'learning_rate': 1.6666666666666667e-06, 'epoch': 0.32}
{'loss': 447.8989, 'grad_norm': 8032.0, 'learning_rate': 3.3333333333333333e-06, 'epoch': 0.48}
{'loss': 445.5749, 'grad_norm': 7648.0, 'learning_rate': 5e-06, 'epoch': 0.64}
{'loss': 443.0435, 'grad_norm': 7168.0, 'learning_rate': 6.666666666666667e-06, 'epoch': 0.8}
{'loss': 440.1309, 'grad_norm': 6144.0, 'learning_rate': 8.333333333333334e-06, 'epoch': 0.96}
{'loss': 111.5688, 'grad_norm': 3808.0, 'learning_rate': 1e-05, 'epoch': 1.0}
12% 7/60 [45:40<1:58:43, 134.40s/it][2026-08-04 16:16:53,580][evaluator][INFO] - **\*** Running TOFU evaluation suite **\***
[2026-08-04 16:16:53,584][evaluator][INFO] - Fine-grained evaluations will be saved to: /content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42/checkpoint-7/evals/TOFU_EVAL.json
[2026-08-04 16:16:53,584][evaluator][INFO] - Aggregated evaluations will be summarised in: /content/drive/My Drive/fq-ssiuu-research/checkpoints/c2_npo/seed42/checkpoint-7/evals/TOFU_SUMMARY.json
[2026-08-04 16:16:54,623][metrics][INFO] - Evaluating forget_Q_A_PARA_Prob

Calculating loss: 0% 0/100 [00:00<?, ?it/s]
Calculating loss: 1% 1/100 [00:00<00:40, 2.47it/s]
Calculating loss: 2% 2/100 [00:00<00:42, 2.33it/s]
Calculating loss: 3% 3/100 [00:01<00:38, 2.49it/s]
.
.
.
.
.
.
.
Calculating loss: 96% 96/100 [00:41<00:01, 2.23it/s]
Calculating loss: 97% 97/100 [00:42<00:01, 2.22it/s]
Calculating loss: 98% 98/100 [00:42<00:00, 2.21it/s]
Calculating loss: 99% 99/100 [00:43<00:00, 2.22it/s]
Calculating loss: 100% 100/100 [00:43<00:00, 2.30it/s]
Evaluated 200 examples
[2026-08-04 16:17:38,994][metrics][INFO] - Evaluating forget_Q_A_PERT_Prob

Calculating loss: 0% 0/100 [00:00<?, ?it/s]
Calculating loss: 1% 1/100 [00:01<03:04, 1.87s/it]
Calculating loss: 2% 2/100 [00:04<03:19, 2.04s/it]
Calculating loss: 3% 3/100 [00:05<03:09, 1.95s/it]
.
.
.
.
.
.
.
Calculating loss: 96% 96/100 [03:27<00:08, 2.23s/it]
Calculating loss: 97% 97/100 [03:29<00:06, 2.24s/it]
Calculating loss: 98% 98/100 [03:32<00:04, 2.24s/it]
Calculating loss: 99% 99/100 [03:34<00:02, 2.24s/it]
Calculating loss: 100% 100/100 [03:36<00:00, 2.17s/it]
Evaluated 200 examples
[2026-08-04 16:21:15,622][metrics][INFO] - Evaluating forget_truth_ratio
[2026-08-04 16:21:15,638][evaluator][INFO] - Result for metric forget_truth_ratio: 0.48726341613849156
[2026-08-04 16:21:15,905][metrics][INFO] - Skipping forget_quality's precompute forget_truth_ratio, already evaluated.
[2026-08-04 16:21:15,907][metrics][INFO] - Evaluating forget_quality
[2026-08-04 16:21:15,908][metrics][WARNING] - retain_model_logs not provided in reference_logs, setting forget_quality to None
[2026-08-04 16:21:15,908][evaluator][INFO] - Result for metric forget_quality: None
[2026-08-04 16:21:16,820][metrics][INFO] - Evaluating forget_Q_A_Prob

Calculating loss: 0% 0/100 [00:00<?, ?it/s]
Calculating loss: 1% 1/100 [00:00<00:40, 2.47it/s]
Calculating loss: 2% 2/100 [00:00<00:38, 2.53it/s]
Calculating loss: 3% 3/100 [00:01<00:37, 2.59it/s]
.
.
.
.
.
.
Calculating loss: 93% 93/100 [00:39<00:02, 2.42it/s]
Calculating loss: 94% 94/100 [00:39<00:02, 2.36it/s]
Calculating loss: 95% 95/100 [00:39<00:02, 2.32it/s]
Calculating loss: 96% 96/100 [00:40<00:01, 2.29it/s]
Calculating loss: 97% 97/100 [00:40<00:01, 2.28it/s]
Calculating loss: 98% 98/100 [00:41<00:00, 2.26it/s]
Calculating loss: 99% 99/100 [00:41<00:00, 2.25it/s]
Calculating loss: 100% 100/100 [00:42<00:00, 2.37it/s]
Evaluated 200 examples
[2026-08-04 16:21:59,077][evaluator][INFO] - Result for metric forget_Q_A_Prob: 0.8235540720820427
[2026-08-04 16:22:00,078][metrics][INFO] - Evaluating forget_Q_A_ROUGE

Calculating text similarity: 0% 0/100 [00:00<?, ?it/s]
Calculating text similarity: 1% 1/100 [00:02<04:49, 2.92s/it]
Calculating text similarity: 2% 2/100 [00:06<05:09, 3.16s/it]
Calculating text similarity: 3% 3/100 [00:08<04:23, 2.72s/it]
Calculating text similarity: 4% 4/100 [00:14<06:22, 3.99s/it]
Calculating text similarity: 5% 5/100 [00:17<05:51, 3.71s/it]
Calculating text similarity: 6% 6/100 [00:23<06:44, 4.31s/it]
.
.
.
.
.
Calculating text similarity: 96% 96/100 [05:52<00:16, 4.01s/it]
Calculating text similarity: 97% 97/100 [05:57<00:12, 4.10s/it]
Calculating text similarity: 98% 98/100 [06:01<00:08, 4.05s/it]
Calculating text similarity: 99% 99/100 [06:04<00:03, 3.98s/it]
Calculating text similarity: 100% 100/100 [06:08<00:00, 3.68s/it]
Evaluated 200 examples
[2026-08-04 16:28:08,349][evaluator][INFO] - Result for metric forget_Q_A_ROUGE: 0.7334026115182186
[2026-08-04 16:28:09,571][metrics][INFO] - Evaluating retain_Q_A_Prob

Calculating loss: 0% 0/200 [00:00<?, ?it/s]
Calculating loss: 0% 1/200 [00:00<01:25, 2.33it/s]
Calculating loss: 1% 2/200 [00:00<01:27, 2.27it/s]
.
.
.
.
.
.
.
.
.
Calculating loss: 98% 197/200 [01:19<00:01, 2.53it/s]
Calculating loss: 99% 198/200 [01:20<00:00, 2.56it/s]
Calculating loss: 100% 199/200 [01:20<00:00, 2.59it/s]
Calculating loss: 100% 200/200 [01:20<00:00, 2.47it/s]
Evaluated 400 examples
[2026-08-04 16:29:31,316][metrics][INFO] - Evaluating retain_Q_A_ROUGE

Calculating text similarity: 0% 0/200 [00:00<?, ?it/s]
Calculating text similarity: 0% 1/200 [00:02<09:42, 2.93s/it]
Calculating text similarity: 1% 2/200 [00:06<11:24, 3.46s/it]
.
.
.
.
.
.
.
Calculating text similarity: 99% 198/200 [11:05<00:06, 3.27s/it]
Calculating text similarity: 100% 199/200 [11:08<00:03, 3.23s/it]
Calculating text similarity: 100% 200/200 [11:14<00:00, 3.37s/it]
Evaluated 400 examples
[2026-08-04 16:40:46,952][metrics][INFO] - Evaluating retain_Q_A_PARA_Prob

Calculating loss: 0% 0/200 [00:00<?, ?it/s]
Calculating loss: 0% 1/200 [00:00<01:24, 2.37it/s]
Calculating loss: 1% 2/200 [00:00<01:26, 2.28it/s]
Calculating loss: 2% 3/200 [00:01<01:27, 2.25it/s]
.
.
.
.
.
.
.
Calculating loss: 98% 196/200 [01:23<00:01, 2.30it/s]
Calculating loss: 98% 197/200 [01:23<00:01, 2.27it/s]
Calculating loss: 99% 198/200 [01:23<00:00, 2.25it/s]
Calculating loss: 100% 199/200 [01:24<00:00, 2.24it/s]
Calculating loss: 100% 200/200 [01:24<00:00, 2.36it/s]
Evaluated 400 examples
[2026-08-04 16:42:12,704][metrics][INFO] - Evaluating retain_Q_A_PERT_Prob

Calculating loss: 0% 0/200 [00:00<?, ?it/s]
Calculating loss: 0% 1/200 [00:01<06:13, 1.88s/it]
Calculating loss: 1% 2/200 [00:04<06:54, 2.09s/it]
.
.
.
.
.
.
.
.
.
.
Calculating loss: 98% 197/200 [06:52<00:06, 2.16s/it]
Calculating loss: 99% 198/200 [06:54<00:04, 2.19s/it]
Calculating loss: 100% 199/200 [06:57<00:02, 2.21s/it]
Calculating loss: 100% 200/200 [06:59<00:00, 2.10s/it]
Evaluated 400 examples
[2026-08-04 16:49:11,971][metrics][INFO] - Evaluating retain_Truth_Ratio
[2026-08-04 16:49:12,797][metrics][INFO] - Evaluating ra_Q_A_Prob

Calculating loss: 0% 0/50 [00:00<?, ?it/s]
Calculating loss: 2% 1/50 [00:00<00:08, 5.71it/s]
Calculating loss: 4% 2/50 [00:00<00:08, 5.48it/s]
.
.
.
.
Calculating loss: 94% 47/50 [00:10<00:00, 3.80it/s]
Calculating loss: 96% 48/50 [00:10<00:00, 3.90it/s]
Calculating loss: 98% 49/50 [00:10<00:00, 4.26it/s]
Calculating loss: 100% 50/50 [00:11<00:00, 4.54it/s]
Evaluated 100 examples
[2026-08-04 16:49:24,505][metrics][INFO] - Evaluating ra_Q_A_PERT_Prob

Calculating loss: 0% 0/50 [00:00<?, ?it/s]
Calculating loss: 2% 1/50 [00:00<00:26, 1.84it/s]
.
.
.
.
Calculating loss: 92% 46/50 [00:29<00:02, 1.53it/s]
Calculating loss: 94% 47/50 [00:30<00:02, 1.27it/s]
Calculating loss: 96% 48/50 [00:31<00:01, 1.21it/s]
Calculating loss: 98% 49/50 [00:31<00:00, 1.34it/s]
Calculating loss: 100% 50/50 [00:33<00:00, 1.51it/s]
Evaluated 100 examples
[2026-08-04 16:49:57,602][metrics][INFO] - Evaluating ra_Q_A_Prob_normalised
[2026-08-04 16:49:58,427][metrics][INFO] - Evaluating ra_Q_A_ROUGE

Calculating text similarity: 0% 0/50 [00:00<?, ?it/s]
Calculating text similarity: 2% 1/50 [00:01<01:31, 1.87s/it]
.
.
.
.
Calculating text similarity: 94% 47/50 [01:33<00:06, 2.09s/it]
Calculating text similarity: 96% 48/50 [01:35<00:04, 2.05s/it]
Calculating text similarity: 98% 49/50 [01:36<00:01, 1.96s/it]
Calculating text similarity: 100% 50/50 [01:38<00:00, 1.98s/it]
Evaluated 100 examples
[2026-08-04 16:51:37,238][metrics][INFO] - Skipping ra_Truth_Ratio's precompute ra_Q_A_Prob, already evaluated.
[2026-08-04 16:51:37,238][metrics][INFO] - Skipping ra_Truth_Ratio's precompute ra_Q_A_PERT_Prob, already evaluated.
[2026-08-04 16:51:37,238][metrics][INFO] - Evaluating ra_Truth_Ratio
[2026-08-04 16:51:37,920][metrics][INFO] - Evaluating wf_Q_A_Prob

Calculating loss: 0% 0/59 [00:00<?, ?it/s]
Calculating loss: 2% 1/59 [00:00<00:10, 5.73it/s]
Calculating loss: 3% 2/59 [00:00<00:10, 5.60it/s]
.
.
.
.
.
Calculating loss: 95% 56/59 [00:10<00:00, 5.32it/s]
Calculating loss: 97% 57/59 [00:10<00:00, 5.33it/s]
Calculating loss: 98% 58/59 [00:11<00:00, 5.34it/s]
Calculating loss: 100% 59/59 [00:11<00:00, 5.30it/s]
Evaluated 117 examples
[2026-08-04 16:51:49,750][metrics][INFO] - Evaluating wf_Q_A_PERT_Prob

Calculating loss: 0% 0/59 [00:00<?, ?it/s]
Calculating loss: 2% 1/59 [00:00<00:30, 1.88it/s]
Calculating loss: 3% 2/59 [00:01<00:30, 1.86it/s]
Calculating loss: 5% 3/59 [00:01<00:30, 1.84it/s]
.
.
.
.
.
Calculating loss: 95% 56/59 [00:32<00:01, 1.65it/s]
Calculating loss: 97% 57/59 [00:32<00:01, 1.69it/s]
Calculating loss: 98% 58/59 [00:33<00:00, 1.72it/s]
Calculating loss: 100% 59/59 [00:33<00:00, 1.75it/s]
Evaluated 117 examples
[2026-08-04 16:52:23,485][metrics][INFO] - Evaluating wf_Q_A_Prob_normalised
[2026-08-04 16:52:24,185][metrics][INFO] - Evaluating wf_Q_A_ROUGE

Calculating text similarity: 0% 0/59 [00:00<?, ?it/s]
Calculating text similarity: 2% 1/59 [00:01<01:21, 1.41s/it]
Calculating text similarity: 3% 2/59 [00:04<02:30, 2.63s/it]
.
.
.
.
.
.
Calculating text similarity: 92% 54/59 [02:00<00:13, 2.61s/it]
Calculating text similarity: 93% 55/59 [02:03<00:11, 2.85s/it]
Calculating text similarity: 95% 56/59 [02:05<00:07, 2.44s/it]
Calculating text similarity: 97% 57/59 [02:06<00:04, 2.16s/it]
Calculating text similarity: 98% 58/59 [02:09<00:02, 2.45s/it]
Calculating text similarity: 100% 59/59 [02:10<00:00, 2.21s/it]
Evaluated 117 examples
[2026-08-04 16:54:34,357][metrics][INFO] - Skipping wf_Truth_Ratio's precompute wf_Q_A_Prob, already evaluated.
[2026-08-04 16:54:34,358][metrics][INFO] - Skipping wf_Truth_Ratio's precompute wf_Q_A_PERT_Prob, already evaluated.
[2026-08-04 16:54:34,358][metrics][INFO] - Evaluating wf_Truth_Ratio
[2026-08-04 16:54:34,359][metrics][INFO] - Evaluating model_utility
[2026-08-04 16:54:34,374][evaluator][INFO] - Result for metric model_utility: 0.5963362552918083
[2026-08-04 16:54:35,675][metrics][INFO] - Evaluating mia_min_k

0% 0/100 [00:00<?, ?it/s]
1% 1/100 [00:00<00:23, 4.26it/s]
2% 2/100 [00:00<00:23, 4.20it/s]
3% 3/100 [00:00<00:23, 4.19it/s]
.
.
.
.
.
95% 95/100 [00:27<00:01, 3.32it/s]
96% 96/100 [00:27<00:01, 3.27it/s]
97% 97/100 [00:27<00:00, 3.25it/s]
98% 98/100 [00:28<00:00, 3.22it/s]
99% 99/100 [00:28<00:00, 3.20it/s]
100% 100/100 [00:28<00:00, 3.47it/s]

0% 0/200 [00:00<?, ?it/s]
0% 1/200 [00:00<00:47, 4.16it/s]
1% 2/200 [00:00<00:48, 4.11it/s]
2% 3/200 [00:00<00:54, 3.63it/s]
2% 4/200 [00:01<00:51, 3.80it/s]
.
.
.
.
.
97% 194/200 [00:59<00:02, 2.93it/s]
98% 195/200 [00:59<00:01, 2.98it/s]
98% 196/200 [01:00<00:01, 2.99it/s]
98% 197/200 [01:00<00:00, 3.03it/s]
99% 198/200 [01:00<00:00, 3.05it/s]
100% 199/200 [01:01<00:00, 3.08it/s]
100% 200/200 [01:01<00:00, 3.25it/s]
[2026-08-04 16:56:06,168][metrics][INFO] - Evaluating privleak
[2026-08-04 16:56:06,169][metrics][WARNING] - retain_model_logs evals not provided for privleak, using default retain auc of 0.5
[2026-08-04 16:56:06,169][evaluator][INFO] - Result for metric privleak: -99.1024999801795
[2026-08-04 16:56:07,093][metrics][INFO] - Evaluating extraction_strength

Calculating ES: 0% 0/100 [00:00<?, ?it/s]
Calculating ES: 1% 1/100 [00:00<00:22, 4.33it/s]
Calculating ES: 2% 2/100 [00:00<00:23, 4.19it/s]
.
.
.
.
.
.
Calculating ES: 90% 90/100 [00:25<00:03, 3.11it/s]
Calculating ES: 91% 91/100 [00:26<00:02, 3.13it/s]
Calculating ES: 92% 92/100 [00:26<00:02, 3.36it/s]
Calculating ES: 93% 93/100 [00:26<00:01, 3.54it/s]
Calculating ES: 94% 94/100 [00:27<00:01, 3.42it/s]
Calculating ES: 95% 95/100 [00:27<00:01, 3.32it/s]
Calculating ES: 96% 96/100 [00:27<00:01, 3.26it/s]
Calculating ES: 97% 97/100 [00:28<00:00, 3.24it/s]
Calculating ES: 98% 98/100 [00:28<00:00, 3.21it/s]
Calculating ES: 99% 99/100 [00:28<00:00, 3.18it/s]
Calculating ES: 100% 100/100 [00:29<00:00, 3.45it/s]
Evaluated 200 examples
[2026-08-04 16:56:36,099][evaluator][INFO] - Result for metric extraction_strength: 0.5665158436006115
{'forget_truth_ratio': np.float64(0.48726341613849156), 'forget_Q_A_Prob': np.float64(0.8235540720820427), 'forget_Q_A_ROUGE': np.float64(0.7334026115182186), 'model_utility': np.float64(0.5963362552918083), 'privleak': np.float64(-99.1024999801795), 'extraction_strength': np.float64(0.5665158436006115), 'epoch': 1.0}
12% 7/60 [1:25:23<1:58:43, 134.40s/it]^C

# Execution halted, session stopped due to running out of free GPU quota.

# ...

### We are shifting to kaggle, colab's free resources are too little for our training.

#### Kaggle Notebook training will start from scratch.

## Kaggle Run 1:

=== Step 1: Cloning Repositories ===
fatal: destination path '/kaggle/working/fq-ssiuu' already exists and is not an empty directory.
Already on 'gate/1-baseline-repro'
Your branch is up to date with 'origin/gate/1-baseline-repro'.
fatal: destination path '/kaggle/working/open-unlearning' already exists and is not an empty directory.
M configs/trainer/finetune.yaml
M requirements.txt
M src/evals/lm_eval.py
M src/trainer/**init**.py
Already on 'fq-ssiuu-integration'
Your branch is up to date with 'origin/fq-ssiuu-integration'.
=== Step 2: Patching Dependencies ===
Preparing metadata (setup.py) ... done
=== Step 3: Starting Dual-GPU T4 Training ===
The following values were not passed to `accelerate launch` and had defaults used instead:
`--num_machines` was set to a value of `1`
`--mixed_precision` was set to a value of `'no'`
`--dynamo_backend` was set to a value of `'no'`
To avoid this warning pass in values for each of the problematic parameters or run `accelerate config`.
[2026-08-05 06:29:44,541] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2026-08-05 06:29:44,612] [INFO] [real_accelerator.py:219:get_accelerator] Setting ds_accelerator to cuda (auto detect)
[2026-08-05 06:29:54,275][model][INFO] - Setting pad_token as eos token: <|eot_id|>
[2026-08-05 06:29:54,477][model][INFO] - Setting pad_token as eos token: <|eot_id|>
[2026-08-05 06:29:58,362][evaluator][INFO] - Evaluations stored in the experiment directory: ./saves/unlearn/GATE1_BASELINE
[2026-08-05 06:29:58,371][evaluator][INFO] - Evaluations stored in the experiment directory: ./saves/unlearn/GATE1_BASELINE
/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py:494: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
self.scaler = torch.cuda.amp.GradScaler(**kwargs)
/usr/local/lib/python3.12/dist-packages/accelerate/accelerator.py:494: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
self.scaler = torch.cuda.amp.GradScaler(**kwargs)
[2026-08-05 06:29:59,592][trainer][INFO] - NPO Trainer loaded, output_dir: /kaggle/working/checkpoints/c2_npo/seed42
[2026-08-05 06:29:59,802][trainer][INFO] - NPO Trainer loaded, output_dir: /kaggle/working/checkpoints/c2_npo/seed42
0%| | 0/60 [00:00<?, ?it/s][2026-08-05 06:30:00,954][trainer.base][WARNING] - Custom evaluator can be run with this Trainer only when a single accelerator process is running.
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
Could not estimate the number of tokens of the input, floating-point operations will not be computed
Could not estimate the number of tokens of the input, floating-point operations will not be computed
{'loss': 223.7601, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 0.16}  
{'loss': 223.3218, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 0.32}  
{'loss': 223.4493, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 0.48}  
{'loss': 223.3905, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 0.64}  
{'loss': 223.9659, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 0.8}  
{'loss': 223.6074, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 0.96}  
{'loss': 55.6851, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 1.0}  
 12%|█████▏ | 7/60 [01:10<07:31, 8.52s/it][2026-08-05 06:31:10,999][trainer.base][WARNING] - Custom evaluator can be run with this Trainer only when a single accelerator process is running.
{'loss': 224.4886, 'grad_norm': nan, 'learning_rate': 0.0, 'epoch': 1.16}  
{'loss': 223.6977, 'grad_norm': 2994.0, 'learning_rate': 0.0, 'epoch': 1.32}  
{'loss': 223.4594, 'grad_norm': inf, 'learning_rate': 1.6666666666666667e-06, 'epoch': 1.48}
{'loss': 223.314, 'grad_norm': 3312.0, 'learning_rate': 1.6666666666666667e-06, 'epoch': 1.64}
{'loss': 222.8391, 'grad_norm': 3198.0, 'learning_rate': 3.3333333333333333e-06, 'epoch': 1.8}
{'loss': 217.1892, 'grad_norm': 3924.0, 'learning_rate': 5e-06, 'epoch': 1.96}  
{'loss': 55.3231, 'grad_norm': 1638.0, 'learning_rate': 6.666666666666667e-06, 'epoch': 2.0}
23%|██████████ | 14/60 [02:31<07:14, 9.45s/it][2026-08-05 06:32:32,248][trainer.base][WARNING] - Custom evaluator can be run with this Trainer only when a single accelerator process is running.
{'loss': 145.6496, 'grad_norm': 2834.0, 'learning_rate': 8.333333333333334e-06, 'epoch': 2.16}
{'loss': 111.0035, 'grad_norm': 2356.0, 'learning_rate': 1e-05, 'epoch': 2.32}  
{'loss': 77.1553, 'grad_norm': 1902.0, 'learning_rate': 9.814814814814815e-06, 'epoch': 2.48}
{'loss': 47.321, 'grad_norm': 1627.0, 'learning_rate': 9.62962962962963e-06, 'epoch': 2.64}
{'loss': 48.018, 'grad_norm': 1348.0, 'learning_rate': 9.444444444444445e-06, 'epoch': 2.8}
{'loss': 32.177, 'grad_norm': 574.0, 'learning_rate': 9.25925925925926e-06, 'epoch': 2.96}
{'loss': 7.1426, 'grad_norm': 205.875, 'learning_rate': 9.074074074074075e-06, 'epoch': 3.0}
35%|███████████████ | 21/60 [03:53<06:10, 9.50s/it][2026-08-05 06:33:54,378][trainer.base][WARNING] - Custom evaluator can be run with this Trainer only when a single accelerator process is running.
{'loss': 27.5301, 'grad_norm': 516.0, 'learning_rate': 8.888888888888888e-06, 'epoch': 3.16}
{'loss': 24.9563, 'grad_norm': 336.0, 'learning_rate': 8.703703703703705e-06, 'epoch': 3.32}
{'loss': 25.0582, 'grad_norm': 322.25, 'learning_rate': 8.518518518518519e-06, 'epoch': 3.48}
{'loss': 20.561, 'grad_norm': 270.25, 'learning_rate': 8.333333333333334e-06, 'epoch': 3.64}
{'loss': 23.3456, 'grad_norm': 306.75, 'learning_rate': 8.148148148148148e-06, 'epoch': 3.8}
{'loss': 17.573, 'grad_norm': 239.375, 'learning_rate': 7.962962962962963e-06, 'epoch': 3.96}
{'loss': 4.617, 'grad_norm': 101.0625, 'learning_rate': 7.77777777777778e-06, 'epoch': 4.0}
47%|████████████████████ | 28/60 [05:16<05:05, 9.54s/it][2026-08-05 06:35:16,964][trainer.base][WARNING] - Custom evaluator can be run with this Trainer only when a single accelerator process is running.
{'loss': 17.5668, 'grad_norm': 235.625, 'learning_rate': 7.592592592592594e-06, 'epoch': 4.16}
{'loss': 16.195, 'grad_norm': 243.375, 'learning_rate': 7.4074074074074075e-06, 'epoch': 4.32}
{'loss': 14.1026, 'grad_norm': 254.875, 'learning_rate': 7.222222222222223e-06, 'epoch': 4.48}
{'loss': 13.2727, 'grad_norm': 250.75, 'learning_rate': 7.0370370370370375e-06, 'epoch': 4.64}
{'loss': 14.6596, 'grad_norm': 483.5, 'learning_rate': 6.851851851851853e-06, 'epoch': 4.8}
{'loss': 13.0842, 'grad_norm': 275.0, 'learning_rate': 6.666666666666667e-06, 'epoch': 4.96}
{'loss': 4.7139, 'grad_norm': 190.625, 'learning_rate': 6.481481481481482e-06, 'epoch': 5.0}
58%|█████████████████████████ | 35/60 [06:38<03:59, 9.56s/it][2026-08-05 06:36:39,689][trainer.base][WARNING] - Custom evaluator can be run with this Trainer only when a single accelerator process is running.
{'loss': 9.3003, 'grad_norm': 234.0, 'learning_rate': 6.296296296296297e-06, 'epoch': 5.16}
{'loss': 9.7562, 'grad_norm': 259.25, 'learning_rate': 6.111111111111112e-06, 'epoch': 5.32}
{'loss': 10.5708, 'grad_norm': 251.5, 'learning_rate': 5.925925925925926e-06, 'epoch': 5.48}
{'loss': 8.3831, 'grad_norm': 273.75, 'learning_rate': 5.740740740740741e-06, 'epoch': 5.64}
{'loss': 10.7245, 'grad_norm': 354.75, 'learning_rate': 5.555555555555557e-06, 'epoch': 5.8}
{'loss': 7.3508, 'grad_norm': 229.5, 'learning_rate': 5.370370370370371e-06, 'epoch': 5.96}
{'loss': 1.2911, 'grad_norm': 121.4375, 'learning_rate': 5.185185185185185e-06, 'epoch': 6.0}
70%|██████████████████████████████ | 42/60 [08:01<02:52, 9.60s/it][2026-08-05 06:38:02,839][trainer.base][WARNING] - Custom evaluator can be run with this Trainer only when a single accelerator process is running.
{'loss': 10.5672, 'grad_norm': 314.0, 'learning_rate': 5e-06, 'epoch': 6.16}  
{'loss': 7.8955, 'grad_norm': 237.625, 'learning_rate': 4.814814814814815e-06, 'epoch': 6.32}
{'loss': 7.8776, 'grad_norm': 245.25, 'learning_rate': 4.62962962962963e-06, 'epoch': 6.48}
{'loss': 10.5083, 'grad_norm': 360.25, 'learning_rate': 4.444444444444444e-06, 'epoch': 6.64}
{'loss': 8.1285, 'grad_norm': 258.0, 'learning_rate': 4.2592592592592596e-06, 'epoch': 6.8}
{'loss': 8.9321, 'grad_norm': 308.5, 'learning_rate': 4.074074074074074e-06, 'epoch': 6.96}
{'loss': 1.395, 'grad_norm': 109.125, 'learning_rate': 3.88888888888889e-06, 'epoch': 7.0}
82%|███████████████████████████████████ | 49/60 [09:25<01:46, 9.71s/it][2026-08-05 06:39:26,223][trainer.base][WARNING] - Custom evaluator can be run with this Trainer only when a single accelerator process is running.
{'loss': 6.219, 'grad_norm': 263.0, 'learning_rate': 3.7037037037037037e-06, 'epoch': 7.16}
{'loss': 7.5547, 'grad_norm': 308.0, 'learning_rate': 3.5185185185185187e-06, 'epoch': 7.32}
{'loss': 6.0737, 'grad_norm': 273.25, 'learning_rate': 3.3333333333333333e-06, 'epoch': 7.48}
{'loss': 5.6424, 'grad_norm': 245.625, 'learning_rate': 3.1481481481481483e-06, 'epoch': 7.64}
{'loss': 5.6855, 'grad_norm': 259.75, 'learning_rate': 2.962962962962963e-06, 'epoch': 7.8}
{'loss': 5.9331, 'grad_norm': 271.75, 'learning_rate': 2.7777777777777783e-06, 'epoch': 7.96}
{'loss': 1.475, 'grad_norm': 211.625, 'learning_rate': 2.5925925925925925e-06, 'epoch': 8.0}
93%|████████████████████████████████████████▏ | 56/60 [10:48<00:38, 9.66s/it][2026-08-05 06:40:49,141][trainer.base][WARNING] - Custom evaluator can be run with this Trainer only when a single accelerator process is running.
{'loss': 5.1234, 'grad_norm': 259.0, 'learning_rate': 2.4074074074074075e-06, 'epoch': 8.16}
{'loss': 4.501, 'grad_norm': 255.125, 'learning_rate': 2.222222222222222e-06, 'epoch': 8.32}
{'loss': 5.712, 'grad_norm': 234.375, 'learning_rate': 2.037037037037037e-06, 'epoch': 8.48}
{'loss': 5.4144, 'grad_norm': 291.75, 'learning_rate': 1.8518518518518519e-06, 'epoch': 8.64}
100%|███████████████████████████████████████████| 60/60 [11:43<00:00, 12.23s/it][2026-08-05 06:41:44,481][trainer.base][WARNING] - Custom evaluator can be run with this Trainer only when a single accelerator process is running.
{'train_runtime': 710.7446, 'train_samples_per_second': 2.814, 'train_steps_per_second': 0.084, 'train_loss': 61.08682394226392, 'epoch': 8.64}
100%|███████████████████████████████████████████| 60/60 [11:50<00:00, 11.84s/it]
[2026-08-05 06:41:58,128][trainer.base][WARNING] - Custom evaluator can be run with this Trainer only when a single accelerator process is running.
[rank0]:[W805 06:41:59.200938451 ProcessGroupNCCL.cpp:1553] Warning: WARNING: destroy_process_group() was not called before program exit, which can leak resources. For more info, please see https://pytorch.org/docs/stable/distributed.html#shutdown (function operator())

## Kaggle Run 2:

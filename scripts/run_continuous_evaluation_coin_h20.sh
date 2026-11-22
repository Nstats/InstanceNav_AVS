# CoIN x PSG-Nav E0 second runs (_r2) on the single-GPU H20, 100 frozen episodes per seen split.
# Run each line from the repo root in the psgnav-instance environment, one terminal per line;
# the two run in parallel on GPU 0. To resume an interrupted run, rerun its line with --resume.
bash scripts/run_coin_e0.sh --vlm qwen3.5-flash --gpu 0 --split val_seen --timestamp coin_e0_val_seen_r2
bash scripts/run_coin_e0.sh --vlm qwen3.5-flash --gpu 0 --split val_seen_synonyms --timestamp coin_e0_val_seen_synonyms_r2

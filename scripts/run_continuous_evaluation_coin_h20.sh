# Runs on the single-GPU H20. Run each line from the repo root in the psgnav-instance
# environment, one terminal per line; lines under one heading run in parallel on GPU 0. To
# resume an interrupted run, rerun its line with --resume.

# VLFM base on CoIN val_unseen (100 frozen episodes): VLFM + AVS (E0 / A0 configuration) and
# original VLFM (E1: verification off). The InstanceNav pair runs on the local machine
# (scripts/run_continuous_evaluation_coin.sh). Needs /data/lxj/vlfm with its data/ weights and
# data/vlfm/pointnav_weights_state_dict.pth (tools/export_vlfm_pointnav.py) on this machine.
# Start the VLFM model servers once, before the two runs:
bash scripts/launch_vlfm_servers.sh --gpu 0
bash scripts/run_coin_e0.sh --vlm qwen3.5-flash --gpu 0 --nav-base vlfm --timestamp coin_vlfm_avs_r1
bash scripts/run_coin_e0.sh --vlm qwen3.5-flash --gpu 0 --nav-base vlfm --enable-instance-confirmation false --enable-candidate-approach true --timestamp coin_vlfm_r1

# Done (2026-11-22), CoIN x PSG-Nav E0 second runs (_r2) on the seen splits (SR / SPL):
#bash scripts/run_coin_e0.sh --vlm qwen3.5-flash --gpu 0 --split val_seen --timestamp coin_e0_val_seen_r2   # 15.0 / 6.04
#bash scripts/run_coin_e0.sh --vlm qwen3.5-flash --gpu 0 --split val_seen_synonyms --timestamp coin_e0_val_seen_synonyms_r2   # 13.0 / 6.47

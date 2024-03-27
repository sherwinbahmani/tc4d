seed=0
gpu=0
exp_root_dir=/path/to

############### Trajectory-conditioned generation
scene_setup_path=configs_prompts/a_deer_walking.yaml
# Stage 1
python launch.py --config configs/tc4d_stage_1.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a deer walking" system.scene_setup_path=$scene_setup_path

# Stage 2
ckpt=/path/to/tc4d_stage_1/a_deer_walking@timestamp/ckpts/last.ckpt
python launch.py --config configs/tc4d_stage_2.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a deer walking" system.scene_setup_path=$scene_setup_path system.weights=$ckpt

# Stage 3
ckpt=/path/to/tc4d_stage_2/a_deer_walking@timestamp/ckpts/last.ckpt
python launch.py --config configs/tc4d_stage_3.yaml --train --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a deer walking" system.scene_setup_path=$scene_setup_path system.weights=$ckpt
###############

############### Compositional 4D Scene Generation
# After training multiple stage 3 trajectory-conditioned prompts
# Add ckpts in the compositional config and define the trajectory list, see configs_comp for examples used in the paper
scene_setup_path=configs_comp0/comp0.yaml
ckpt=/path/to/tc4d_stage_2/a_deer_walking@timestamp/ckpts/last.ckpt # Just a dummy input, overwritten by ckpts specified in the comp0 yaml
python launch.py --config configs/tc4d_stage_3.yaml --test --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a deer walking" system.scene_setup_path=$scene_setup_path system.weights=$ckpt
###############

############### High resolution video renderings
# Render high resolution videos used in the paper and project page after training
ckpt=/path/to/tc4d_stage_3/a_deer_walking@timestamp/ckpts/last.ckpt
python launch.py --config configs/tc4d_stage_3_eval.yaml --test --gpu $gpu exp_root_dir=$exp_root_dir seed=$seed system.prompt_processor.prompt="a deer walking" system.scene_setup_path=$scene_setup_path system.weights=$ckpt
###############
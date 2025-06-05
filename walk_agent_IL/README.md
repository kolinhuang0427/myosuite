python -m myosuite.utils.examine_env -e myoLegWalk-v0 -p walk_agent_IL.diffusion_policy_wrapper.DiffusionPolicyForExamineEnv -n 1 -r offscreen -o ./videos -on diffusion_walking

python -m myosuite.utils.examine_env -e myoLegWalk-v0 -p walk_agent_IL.h2o_policy_wrapper.H2OPolicyWrapper -n 1 -r offscreen -o ./videos -on h2o_walking

python data/visualize_sample_gait.py --output ./videos/sample_gait_loops.mp4 --width 640 --height 480 --fps 30 --loops 3 --speed 0.5
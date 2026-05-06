# handy-cleaner-lerobot
TU/e Honors Academy AI Track Project: Handy-Cleaner robot vacuum with LeRobot SO-101 hand.

## Set-up:
	conda activate lerobot
	cd lerobot
	sudo chmod 666 /dev/ttyACM*

### Find ports
	lerobot-find-port

## Calibrate
	lerobot-calibrate \
		--robot.type=so101_follower \
	    	--robot.port=/dev/ttyACM0 \
	    	--robot.id=follower_arm

	lerobot-calibrate \
		--teleop.type=so101_leader \
	    	--teleop.port=/dev/ttyACM1 \
	    	--teleop.id=leader_arm

## Check cameras
	lerobot-find-cameras opencv

## Teleoperate
	lerobot-teleoperate \ 
	    --robot.type=so101_follower \ 
	    --robot.port=/dev/ttyACM0 \
	    --robot.id=follower_arm \ 
	    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
	    --teleop.type=so101_leader \
	    --teleop.port=/dev/ttyACM1 \
	    --teleop.id=leader_arm \
	    --display_data=true

## Record new dataset
	lerobot-record \
	    --robot.type=so101_follower \
	    --robot.port=/dev/ttyACM1 \
	    --robot.id=follower_arm \
	    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
	    --teleop.type=so101_leader \
	    --teleop.port=/dev/ttyACM0 \
	    --teleop.id=leader_arm \
	    --display_data=true \
	    --dataset.repo_id="MrPsarik/PaperBalls" \
	    --dataset.num_episodes=5 \
	    --dataset.single_task="Put a paper ball into the robot container" \
	    --dataset.reset_time_s=30

### Continue recording for existing dataset
	--resume=true

## Replay one record 
	lerobot-replay \
	    --robot.type=so101_follower \
	    --robot.port=/dev/ttyACM1 \
	    --robot.id=follower_arm \
	    --dataset.repo_id=MrPsarik/t \
	    --dataset.episode=0
    
## Run the model
	lerobot-record \
	  --robot.type=so101_follower \
	  --robot.port=/dev/ttyACM0 \
	  --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
	  --robot.id=follower_arm \
	  --display_data=false \
	  --dataset.repo_id=MrPsarik/eval_t \
	  --dataset.single_task="Move yellow block" \
	  --policy.path=MrPsarik/t_policy
  
### Teleop optional if you want to teleoperate in between episodes
	  --teleop.type=so100_leader \
	  --teleop.port=/dev/ttyACM0 \
	  --teleop.id=my_awesome_leader_arm
  
When connecting one arm it could have any port, so if command failes due to the port number try to change it from ACM0 to ACM1.
  
### Removing Cache before each evaluation run
	rm -rf /home/mrpsarik/.cache/huggingface/lerobot/MrPsarik/eval_t

## Train 
	lerobot-train \
	    --dataset.repo_id="MrPsarik/t" \
	    --policy.type=act \
	    --output_dir=outputs/train/t_policy \
	    --job_name=t_policy \
	    --policy.device=cuda \ 
	    --wandb.enable=true \
	    --policy.repo_id="MrPsarik/t_policy" \
	    --save_freq=5000 \
	    --steps=100000

### Continue training 
	lerobot-train \
	    --resume=true \
	    --steps=100000 \
	    --config_path=outputs/train/t_policy/checkpoints/last/pretrained_model/train_config.json





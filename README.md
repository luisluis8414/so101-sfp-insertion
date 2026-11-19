<div align="center">

<h1>SO-101 SFP+ cable insertion</h1>

<p><b>A low-cost SO-101 arm learns a sub-millimetre task from demonstrations: plugging an SFP+ cable into a network card, with ACT and DAgger.</b></p>

<p>
<img src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white" alt="python 3.12">
<img src="https://img.shields.io/badge/LeRobot-ACT%20%2B%20DAgger-FFD21E?logo=huggingface&logoColor=black" alt="LeRobot ACT + DAgger">
<img src="https://img.shields.io/badge/robot-SO--101-C0392B" alt="SO-101">
<img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0">
</p>

<img src="assets/insertion.gif" alt="The policy grasping the cable and inserting it, seen by its three cameras" width="640">

<p>Autonomous run as the policy sees it (side view, close-up of the port, wrist camera), 2× speed.</p>

<p><a href="https://huggingface.co/datasets/luw84/so101_sfp_insertion">Dataset</a> ·
<a href="https://huggingface.co/luw84/act_so101_sfp_insertion">Policy</a> ·
<a href="guide/">Guide: setup, data, training</a></p>

</div>

## Approach

The arm picks up an SFP+ direct-attach cable and pushes it into the second port of a network card until it latches. An SFP+ module is 13.55 mm wide and its cage 14.00 mm, so the port leaves about 0.4 mm of play, and the module needs a push of up to 18 N to latch. The SO-101 is off by several millimetres at the gripper, so the policy has to correct by eye with a close-up camera of the port and slide the module in along the cage.

The training data is about 2.5 hours of teleoperation with the SO-101 leader arm, 166 demonstrations and 142 DAgger corrections. Every episode was rated and trimmed in a small browser-based [episode editor](tools/episode_editor/), and only the good ones were kept.

The policy is ACT (Action Chunking with Transformers) [[1]](#references), trained from scratch on the demonstrations and fine-tuned after each round of corrections. The corrections come from DAgger [[2, 3]](#references): the policy runs, the operator takes over with the leader arm when it is about to fail, and the recovery plus the corrected sub-task is recorded as a new episode, as in RaC [[4]](#references). LeRobot's DAgger loop is extended here so one person can run it alone, with one key to pause and resume and a rewind of the arm for when the pause came too late, similar to Genie Centurion [[5]](#references).

The final policy grasps and inserts the cable in most runs and recovers from a bad grasp or a misaligned approach. The [video](assets/insertions_robot_view.mp4) shows several insertions in a row; in one of them the connector is shifted in the gripper on purpose. It still fails sometimes (see [Limitations](#limitations)), and there is no success rate measured over a fixed protocol.

## Setup on the table

<p align="center"><img src="assets/setup.webp" alt="SO-101 follower, the network card with the SFP+ cable in front of it, and the side and port cameras on the aluminium profile" width="640"></p>

Follower and leader are both standard SO-101 arms with six Feetech STS3215 servos each, and the follower has a [parallel gripper](https://github.com/roboninecom/SO-ARM100-101-102-Parallel-Gripper). Three USB cameras with 130° lenses record at 640×480 MJPG and 30 fps: one shows the workspace from the side, one (`side_nic`) looks along the card at cable height, and one sits on the gripper. The network card is screwed to a 20×20 aluminium profile, with the cable lying in front of it. Training runs on an RTX 4070 Ti SUPER and needs about 6 GB of its 16 GB.

<p align="center"><img src="assets/port_camera.webp" alt="port camera: the module approaching the port, and latched" width="480"></p>

<p align="center"><i>The port camera: the module approaching the port, and latched</i></p>

## Findings

Re-planning at every step is worse on this arm. ACT predicts a chunk of the next 100 actions (3.3 s) at once, and the ACT paper re-plans at every step and averages the overlapping chunks (temporal ensembling). Here that gave slower, stop-and-go motion. The ALOHA arms in the paper follow their targets almost exactly, so starting a new plan from the measured pose costs nothing. The SO-101 lags behind its targets, and that gap between target and actual position is what pushes the cable in. Re-planning often from the measured pose eats the gap away. Executing each chunk for about 50 steps works best.

The policy uses the joint loads, and relies mostly on the port camera. Altering a single input during approach and insertion (400 frames) changes the predicted shoulder and elbow targets by:

| Input altered | change of the predicted targets |
|---|---|
| joint loads replaced | 5 to 6° |
| side camera blanked | 2 to 3° |
| wrist camera blanked | 7 to 10° |
| port close-up blanked | 16 to 21° |

For comparison, the policy's own error against the recorded actions is about 6°. The loads act mainly on the joints that push the cable in.

## Data, policy and guide

- The [dataset](https://huggingface.co/datasets/luw84/so101_sfp_insertion) on the Hugging Face Hub is the training data of the final policy: 111 episodes (58 demonstrations, 53 DAgger corrections), 45 min, three cameras, joint positions and loads. [Browse it online](https://huggingface.co/spaces/lerobot/visualize_dataset?path=/luw84/so101_sfp_insertion/episode_0).
- [luw84/act_so101_sfp_insertion](https://huggingface.co/luw84/act_so101_sfp_insertion) is the final ACT checkpoint.
- The [guide](guide/) covers hardware, installing LeRobot, calibration, downloading data and policy, training, running the policy and recording your own data.
- The [episode editor](tools/episode_editor/) is the browser tool used to rate, trim and delete episodes before training.

## Limitations

The policy is sensitive to light. Sunlight through the window changes brightness, shadows and reflections on the table over the day, and runs in different light were noticeably worse. Brightness and contrast augmentation does not cover moving shadows and reflections.

Teleoperation with the SO-101 is imprecise. The leader uses the same hobby servos, so its readings jump and drift, and backlash, dead zones and lag of the follower add up to millimetres. That limits the quality of both demonstrations and corrections. Better servos or a stiffer arm would help more than more data.

During the final push the policy occasionally stalls, making smaller and smaller corrections. ACT sees only the current frame, so it cannot tell "pushing and sliding in" from "pushing and jammed".

The follower's 7.4 V servos ran on a 5.4 V supply in this build. The policy learned the resulting lag, and a different supply would change the dynamics.

## Credits

[LeRobot](https://github.com/huggingface/lerobot) (Hugging Face, Apache-2.0) provides the robot drivers, recording, ACT and the DAgger loop this project builds on. It is extended here with single-handed DAgger and joint loads in the state. The SO-101 arm is by [TheRobotStudio](https://github.com/TheRobotStudio/SO-ARM100).

## References

1. Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* RSS 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
2. Stéphane Ross, Geoffrey Gordon, Drew Bagnell. *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning.* AISTATS 2011. [arXiv:1011.0686](https://arxiv.org/abs/1011.0686)
3. Michael Kelly, Chelsea Sidrane, Katherine Driggs-Campbell, Mykel J. Kochenderfer. *HG-DAgger: Interactive Imitation Learning with Human Experts.* ICRA 2019. [arXiv:1810.02890](https://arxiv.org/abs/1810.02890)
4. Zheyuan Hu et al. *RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and Correction.* 2025. [arXiv:2509.07953](https://arxiv.org/abs/2509.07953)
5. AgiBot. *Genie Centurion: Accelerating Scalable Real-World Robot Training with Human Rewind-and-Refine Guidance.* 2025. [arXiv:2505.18793](https://arxiv.org/abs/2505.18793)

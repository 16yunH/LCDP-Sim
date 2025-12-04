"""
Data collection script for robot demonstrations
Uses scripted expert to generate training data in simulation.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import numpy as np
import zarr
import h5py
from tqdm import tqdm
import imageio

sys.path.append(str(Path(__file__).parent.parent))

from lcdp.envs.scripted_expert import ScriptedExpert


def collect_episode(
    env,
    expert: ScriptedExpert,
    instruction: str,
    max_steps: int = 200,
    save_video: bool = False
):
    """
    Collect one episode of demonstration data.
    
    Args:
        env: Simulation environment
        expert: Scripted expert policy
        instruction: Language instruction for the task
        max_steps: Maximum steps per episode
        save_video: Whether to save video frames
    Returns:
        Dictionary with episode data or None if failed
    """
    obs, info = env.reset()
    expert.reset()
    
    # Storage for episode data
    observations = []
    actions = []
    rewards = []
    frames = [] if save_video else None
    
    done = False
    step_count = 0
    success = False
    
    while not done and step_count < max_steps:
        # Get current state (privileged information for expert)
        # This assumes env provides this info - adjust based on your env
        ee_pos = info.get('ee_pos', np.zeros(3))
        ee_quat = info.get('ee_quat', np.zeros(4))
        obj_pos = info.get('obj_pos', np.zeros(3))
        gripper_state = info.get('gripper_state', 0.0)
        target_pos = info.get('target_pos', None)
        
        # Compute action from expert
        action = expert.compute_action(
            ee_pos, ee_quat, obj_pos, gripper_state, target_pos
        )
        
        # Store observation and action
        observations.append(obs['rgb'])
        actions.append(action)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        step_count += 1
        
        # Check success
        if info.get('success', False):
            success = True
        
        # Save frame
        if save_video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
    
    # Only return successful episodes (optional)
    if not success:
        return None
    
    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'instruction': instruction,
        'success': success,
        'length': step_count,
        'frames': frames
    }


def save_episodes_zarr(episodes, output_path):
    """Save episodes to Zarr format."""
    root = zarr.open(output_path, mode='w')
    
    # Concatenate all episodes
    all_obs = np.concatenate([ep['observations'] for ep in episodes], axis=0)
    all_actions = np.concatenate([ep['actions'] for ep in episodes], axis=0)
    
    # Save data
    root.create_dataset('observations/rgb', data=all_obs, chunks=(1, 224, 224, 3))
    root.create_dataset('actions', data=all_actions, chunks=(1, 7))
    
    # Save instructions (one per episode)
    instructions = np.array([ep['instruction'] for ep in episodes], dtype=object)
    root.create_dataset('language_instruction', data=instructions)
    
    # Save episode boundaries
    episode_lengths = [ep['length'] for ep in episodes]
    root.create_dataset('episode_lengths', data=np.array(episode_lengths))
    
    print(f"Saved {len(episodes)} episodes to {output_path}")
    print(f"Total transitions: {len(all_actions)}")


def save_episodes_h5(episodes, output_path):
    """Save episodes to HDF5 format."""
    with h5py.File(output_path, 'w') as f:
        # Concatenate all episodes
        all_obs = np.concatenate([ep['observations'] for ep in episodes], axis=0)
        all_actions = np.concatenate([ep['actions'] for ep in episodes], axis=0)
        
        # Save data
        f.create_dataset('observations/rgb', data=all_obs, compression='gzip')
        f.create_dataset('actions', data=all_actions, compression='gzip')
        
        # Save instructions
        instructions = [ep['instruction'].encode('utf-8') for ep in episodes]
        dt = h5py.string_dtype(encoding='utf-8')
        f.create_dataset('language_instruction', data=instructions, dtype=dt)
        
        # Save episode boundaries
        episode_lengths = [ep['length'] for ep in episodes]
        f.create_dataset('episode_lengths', data=np.array(episode_lengths))
    
    print(f"Saved {len(episodes)} episodes to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect robot demonstration data")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/env_config.yaml',
        help='Path to environment config'
    )
    parser.add_argument(
        '--env',
        type=str,
        default='PickCube-v0',
        help='Environment name'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=100,
        help='Number of episodes to collect'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/demonstrations.zarr',
        help='Output file path'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='zarr',
        choices=['zarr', 'h5'],
        help='Output format'
    )
    parser.add_argument(
        '--save-videos',
        action='store_true',
        help='Save episode videos'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='data/videos',
        help='Directory to save videos'
    )
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config file not found: {args.config}, using defaults")
        config = {}
    
    # Create environment
    try:
        import gymnasium as gym
        try:
            import mani_skill2.envs
            env = gym.make(args.env, render_mode="rgb_array")
        except ImportError:
            print("ManiSkill2 not installed!")
            print("Install with: pip install mani-skill2")
            return
    except ImportError:
        print("Gymnasium not installed!")
        return
    
    # Determine task type from env name
    task_type = "pick"
    if "Push" in args.env:
        task_type = "push"
    elif "Stack" in args.env:
        task_type = "stack"
    
    # Create scripted expert
    expert = ScriptedExpert(
        task=task_type,
        noise_scale=config.get('data_collection', {}).get('expert_noise', 0.01)
    )
    
    # Get instruction template
    instruction = f"{task_type.capitalize()} the object"
    if 'env' in config and 'tasks' in config:
        # Use instruction from config if available
        pass
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if args.save_videos:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect episodes
    print(f"Collecting {args.num_episodes} episodes...")
    episodes = []
    success_count = 0
    
    for episode_idx in tqdm(range(args.num_episodes * 2)):  # Collect more to account for failures
        episode_data = collect_episode(
            env=env,
            expert=expert,
            instruction=instruction,
            max_steps=config.get('env', {}).get('max_episode_steps', 200),
            save_video=args.save_videos
        )
        
        if episode_data is not None:
            episodes.append(episode_data)
            success_count += 1
            
            # Save video if requested
            if args.save_videos and episode_data['frames']:
                video_path = video_dir / f"episode_{success_count:04d}.mp4"
                imageio.mimsave(video_path, episode_data['frames'], fps=20)
            
            # Check if we have enough successful episodes
            if success_count >= args.num_episodes:
                break
    
    env.close()
    
    print(f"\nCollected {success_count} successful episodes")
    
    # Save data
    if args.format == 'zarr':
        save_episodes_zarr(episodes, args.output)
    else:
        save_episodes_h5(episodes, args.output)
    
    print("Data collection complete!")


if __name__ == "__main__":
    main()

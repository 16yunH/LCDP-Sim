"""
Evaluation script for Diffusion Policy
Tests trained policy in simulation environment.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
import imageio

sys.path.append(str(Path(__file__).parent.parent))

from lcdp.models.diffusion_policy import DiffusionPolicy


class Evaluator:
    """Evaluation manager for Diffusion Policy"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None, device: str = "cuda"):
        self.device = device
        
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if config_path is not None:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = checkpoint.get('config', None)
            if config is None:
                raise ValueError("No config found in checkpoint and no config_path provided")
        
        self.config = config
        
        # Create model
        print("Creating model...")
        self.model = DiffusionPolicy(
            action_dim=config['model']['action_dim'],
            action_horizon=config['model']['action_horizon'],
            vision_encoder=config['model']['vision_encoder'],
            vision_feature_dim=config['model']['vision_feature_dim'],
            freeze_vision_backbone=config['model']['freeze_vision_backbone'],
            language_model=config['model']['language_model'],
            language_feature_dim=config['model']['language_feature_dim'],
            freeze_language=config['model']['freeze_language'],
            conditioning_type=config['model']['conditioning_type'],
            unet_base_channels=config['model']['unet_base_channels'],
            unet_channel_mult=tuple(config['model']['unet_channel_mult']),
            unet_num_res_blocks=config['model']['unet_num_res_blocks'],
            num_diffusion_steps=config['model']['num_diffusion_steps'],
            beta_schedule=config['model']['beta_schedule'],
            prediction_type=config['model']['prediction_type'],
            dropout=config['model']['dropout'],
            device=device
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print("Model loaded successfully")
    
    def preprocess_observation(self, obs: dict) -> torch.Tensor:
        """
        Preprocess observation from environment.
        
        Args:
            obs: Dictionary with 'rgb' key containing image
        Returns:
            Preprocessed image tensor [1, 3, H, W]
        """
        image = obs['rgb']  # Assume [H, W, 3] uint8
        
        # Resize if needed
        import cv2
        target_size = tuple(self.config['dataset']['image_size'])
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [-1, 1]
        image = image.astype(np.float32) / 127.5 - 1.0
        
        # HWC to CHW
        image = image.transpose(2, 0, 1)
        
        # Add batch dimension
        image = torch.from_numpy(image).unsqueeze(0).float()
        
        return image.to(self.device)
    
    def evaluate_episode(
        self,
        env,
        instruction: str,
        max_steps: int = 200,
        execute_horizon: int = 8,
        num_inference_steps: int = 10,
        render: bool = False,
        save_video: bool = False,
        video_path: str = None
    ) -> dict:
        """
        Evaluate one episode in the environment.
        
        Args:
            env: Gymnasium/ManiSkill environment
            instruction: Language instruction
            max_steps: Maximum steps per episode
            execute_horizon: Number of predicted actions to execute
            num_inference_steps: DDIM sampling steps
            render: Whether to render
            save_video: Whether to save video
            video_path: Path to save video
        Returns:
            Dictionary with episode results
        """
        obs, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0.0
        success = False
        
        frames = [] if save_video else None
        
        with torch.no_grad():
            while not done and step_count < max_steps:
                # Preprocess observation
                image = self.preprocess_observation(obs)
                
                # Get action from policy
                action_sequence = self.model.get_action(
                    images=image,
                    instructions=[instruction],
                    num_inference_steps=num_inference_steps
                )  # [1, action_dim, horizon]
                
                # Execute actions with receding horizon
                actions_to_execute = action_sequence[0, :, :execute_horizon]  # [action_dim, execute_horizon]
                actions_to_execute = actions_to_execute.cpu().numpy().T  # [execute_horizon, action_dim]
                
                # Denormalize if needed
                if self.config['dataset']['normalize_actions']:
                    # You'd need to save action_mean and action_std with checkpoint
                    pass
                
                # Execute each action
                for action in actions_to_execute:
                    if done:
                        break
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                    step_count += 1
                    
                    # Save frame
                    if save_video and frames is not None:
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame)
                    
                    # Check success
                    if info.get('success', False):
                        success = True
                        done = True
                        break
                
                if render:
                    env.render()
        
        # Save video
        if save_video and frames and video_path:
            imageio.mimsave(video_path, frames, fps=20)
            print(f"Video saved to {video_path}")
        
        return {
            'success': success,
            'total_reward': total_reward,
            'steps': step_count,
            'info': info
        }
    
    def evaluate(
        self,
        env_name: str,
        instruction: str,
        num_episodes: int = 50,
        render: bool = False,
        save_videos: bool = False,
        video_dir: str = "videos"
    ):
        """
        Evaluate policy over multiple episodes.
        
        Args:
            env_name: Environment name
            instruction: Language instruction
            num_episodes: Number of episodes to evaluate
            render: Whether to render
            save_videos: Whether to save videos
            video_dir: Directory to save videos
        """
        try:
            import gymnasium as gym
            # Try to import ManiSkill
            try:
                import mani_skill2.envs
                # Determine render mode
                if render and not save_videos:
                    render_mode = "human"
                    print("Using human render mode for interactive visualization")
                else:
                    render_mode = "rgb_array"
                    
                env = gym.make(env_name, render_mode=render_mode)
            except ImportError:
                print("ManiSkill2 not installed, using placeholder environment")
                # For demo purposes, you'd implement a simple env wrapper
                return
        except ImportError:
            print("Gymnasium not installed")
            return
        
        if save_videos:
            Path(video_dir).mkdir(parents=True, exist_ok=True)
        
        results = []
        success_count = 0
        
        print(f"Evaluating on {num_episodes} episodes...")
        for episode_idx in tqdm(range(num_episodes)):
            video_path = None
            if save_videos:
                video_path = os.path.join(video_dir, f"episode_{episode_idx}.mp4")
            
            episode_result = self.evaluate_episode(
                env=env,
                instruction=instruction,
                max_steps=self.config['env'].get('max_episode_steps', 200) if 'env' in self.config else 200,
                execute_horizon=self.config['inference'].get('execute_horizon', 8) if 'inference' in self.config else 8,
                num_inference_steps=self.config['inference'].get('num_inference_steps', 10) if 'inference' in self.config else 10,
                render=render,
                save_video=save_videos,
                video_path=video_path
            )
            
            results.append(episode_result)
            if episode_result['success']:
                success_count += 1
        
        env.close()
        
        # Compute statistics
        success_rate = success_count / num_episodes
        avg_reward = np.mean([r['total_reward'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        
        print("\n" + "="*50)
        print(f"Evaluation Results ({num_episodes} episodes)")
        print("="*50)
        print(f"Success Rate: {success_rate * 100:.1f}%")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Steps: {avg_steps:.1f}")
        print("="*50)
        
        return {
            'success_rate': success_rate,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'results': results
        }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy")
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (if not in checkpoint)'
    )
    parser.add_argument(
        '--env',
        type=str,
        default='PickCube-v0',
        help='Environment name'
    )
    parser.add_argument(
        '--instruction',
        type=str,
        default='Pick up the red cube',
        help='Language instruction'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=50,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render environment'
    )
    parser.add_argument(
        '--save-videos',
        action='store_true',
        help='Save episode videos'
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='videos',
        help='Directory to save videos'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = Evaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )
    
    # Run evaluation
    evaluator.evaluate(
        env_name=args.env,
        instruction=args.instruction,
        num_episodes=args.num_episodes,
        render=args.render,
        save_videos=args.save_videos,
        video_dir=args.video_dir
    )


if __name__ == "__main__":
    main()

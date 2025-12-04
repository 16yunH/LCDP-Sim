"""
Visualization tools for Diffusion Policy
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from typing import List, Optional

sys.path.append(str(Path(__file__).parent.parent))

from lcdp.models.diffusion_policy import DiffusionPolicy


def visualize_action_sequence(
    actions: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Predicted Action Sequence"
):
    """
    Visualize predicted action sequence.
    
    Args:
        actions: [action_dim, horizon] action array
        save_path: Path to save figure (optional)
        title: Plot title
    """
    action_dim, horizon = actions.shape
    action_labels = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper']
    
    fig, axes = plt.subplots(action_dim, 1, figsize=(12, 10))
    if action_dim == 1:
        axes = [axes]
    
    for i, (ax, label) in enumerate(zip(axes, action_labels)):
        ax.plot(actions[i], marker='o', linewidth=2)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_title(title, fontsize=14, fontweight='bold')
        if i == action_dim - 1:
            ax.set_xlabel('Time Step', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved action visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_diffusion_process(
    model: DiffusionPolicy,
    image: torch.Tensor,
    instruction: str,
    num_steps: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualize the denoising process of diffusion model.
    
    Args:
        model: Trained diffusion policy
        image: [1, 3, H, W] observation image
        instruction: Language instruction
        num_steps: Number of denoising steps
        save_path: Path to save animation
    """
    from diffusers import DDIMScheduler
    
    batch_size = 1
    device = image.device
    
    # Encode observations
    obs_features = model.encode_observations(image, [instruction])
    
    # Initialize with noise
    actions = torch.randn(
        batch_size,
        model.action_dim,
        model.action_horizon,
        device=device
    )
    
    # Setup scheduler
    ddim_scheduler = DDIMScheduler(
        num_train_timesteps=model.num_diffusion_steps,
        beta_schedule=model.noise_scheduler.config.beta_schedule,
        prediction_type=model.noise_scheduler.config.prediction_type,
        clip_sample=True
    )
    ddim_scheduler.set_timesteps(num_steps, device=device)
    
    # Store intermediate results
    action_history = [actions[0].cpu().numpy()]
    
    # Denoising loop
    with torch.no_grad():
        for t in ddim_scheduler.timesteps:
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = model.unet(actions, timesteps)
            noise_pred = model.apply_conditioning(
                noise_pred,
                obs_features['vision'],
                obs_features['language']
            )
            
            # Denoise
            actions = ddim_scheduler.step(noise_pred, t, actions).prev_sample
            action_history.append(actions[0].cpu().numpy())
    
    # Create visualization
    fig, axes = plt.subplots(2, (num_steps + 1) // 2 + 1, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, action_seq in enumerate(action_history):
        ax = axes[i]
        ax.imshow(action_seq, aspect='auto', cmap='viridis')
        ax.set_title(f'Step {i}/{num_steps}', fontsize=10)
        ax.set_xlabel('Time Horizon')
        ax.set_ylabel('Action Dim')
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide extra subplots
    for i in range(len(action_history), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Diffusion Denoising Process\nInstruction: "{instruction}"', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved denoising visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_attention_maps(
    model: DiffusionPolicy,
    image: torch.Tensor,
    instruction: str,
    save_path: Optional[str] = None
):
    """
    Visualize attention maps from cross-attention mechanism.
    (Requires model to have cross-attention and expose attention weights)
    """
    # This is a placeholder - actual implementation would require
    # modifying the model to return attention weights
    print("Attention visualization requires model modification to expose weights")
    pass


def compare_predictions(
    model: DiffusionPolicy,
    image: torch.Tensor,
    instructions: List[str],
    save_path: Optional[str] = None
):
    """
    Compare action predictions for different instructions on same observation.
    
    Args:
        model: Trained model
        image: [1, 3, H, W] observation
        instructions: List of instructions to compare
        save_path: Path to save figure
    """
    predictions = []
    
    with torch.no_grad():
        for instruction in instructions:
            actions = model.get_action(
                images=image,
                instructions=[instruction],
                num_inference_steps=10
            )
            predictions.append(actions[0].cpu().numpy())
    
    # Visualize
    n_instructions = len(instructions)
    fig, axes = plt.subplots(n_instructions, 1, figsize=(14, 4 * n_instructions))
    if n_instructions == 1:
        axes = [axes]
    
    for i, (ax, instruction, actions) in enumerate(zip(axes, instructions, predictions)):
        im = ax.imshow(actions, aspect='auto', cmap='viridis')
        ax.set_title(f'"{instruction}"', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Horizon')
        ax.set_ylabel('Action Dimension')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Diffusion Policy")
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--instruction', type=str, default='Pick up the red cube')
    parser.add_argument('--output-dir', type=str, default='visualizations')
    parser.add_argument('--visualize-denoising', action='store_true')
    parser.add_argument('--compare-instructions', nargs='+', default=None)
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create dummy image (in practice, load from env or dataset)
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    
    print("Model loaded. Generating visualizations...")
    
    # Basic action prediction
    # Note: This is a simplified version - you'd load the full model
    print("\n1. Visualizing action sequence...")
    dummy_actions = np.random.randn(7, 16)  # Placeholder
    visualize_action_sequence(
        dummy_actions,
        save_path=output_dir / "action_sequence.png",
        title=f'Action Sequence for: "{args.instruction}"'
    )
    
    print("\n2. Visualizations saved to:", output_dir)
    print("\nNote: For full functionality, load the actual trained model and environment")


if __name__ == "__main__":
    main()

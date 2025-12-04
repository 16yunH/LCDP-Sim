"""
Script to convert real-world robot data into Zarr format for LCDP-Sim training.
"""

import os
import argparse
import numpy as np
import zarr
import cv2
import json
import glob
from tqdm import tqdm
from pathlib import Path

def load_episode_data(episode_dir):
    """
    Load data from a single episode directory.
    Expected structure:
    episode_dir/
        images/
            0.jpg
            1.jpg
            ...
        actions.json  (or .npy)
        instruction.txt
    """
    episode_dir = Path(episode_dir)
    
    # Load images
    image_paths = sorted(glob.glob(str(episode_dir / "images" / "*.jpg")))
    if not image_paths:
        image_paths = sorted(glob.glob(str(episode_dir / "images" / "*.png")))
    
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    
    images = np.stack(images) # [T, H, W, 3]
    
    # Load actions
    # Assuming actions are stored as a list of lists or numpy array
    if (episode_dir / "actions.npy").exists():
        actions = np.load(episode_dir / "actions.npy")
    elif (episode_dir / "actions.json").exists():
        with open(episode_dir / "actions.json", 'r') as f:
            actions = np.array(json.load(f))
    else:
        raise FileNotFoundError(f"No actions file found in {episode_dir}")
        
    # Load instruction
    if (episode_dir / "instruction.txt").exists():
        with open(episode_dir / "instruction.txt", 'r') as f:
            instruction = f.read().strip()
    else:
        instruction = "default instruction"
        
    # Validate lengths
    if len(images) != len(actions):
        print(f"Warning: Mismatch in {episode_dir}: {len(images)} images vs {len(actions)} actions")
        min_len = min(len(images), len(actions))
        images = images[:min_len]
        actions = actions[:min_len]
        
    return {
        'images': images,
        'actions': actions,
        'instruction': instruction
    }

def main():
    parser = argparse.ArgumentParser(description="Convert real robot data to Zarr")
    parser.add_argument('--input_dir', type=str, required=True, help='Root directory containing episode folders')
    parser.add_argument('--output', type=str, required=True, help='Output .zarr file path')
    parser.add_argument('--resize', nargs=2, type=int, default=(224, 224), help='Resize images to (W, H)')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_path = args.output
    
    # Find all episode directories
    # Assuming any directory with an 'images' subdir is an episode
    episode_dirs = [p.parent for p in input_dir.rglob("images") if p.is_dir()]
    episode_dirs = sorted(list(set(episode_dirs)))
    
    if not episode_dirs:
        print(f"No episodes found in {input_dir}. Structure should be: root/ep1/images/, root/ep2/images/...")
        return

    print(f"Found {len(episode_dirs)} episodes")
    
    # Initialize Zarr
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)
    
    # Create datasets
    # We'll append data, so we use expandable arrays or just list and stack at the end if memory allows.
    # For large datasets, we should write sequentially.
    
    # Initialize arrays with chunks
    # We need to know dimensions first, so load first episode
    first_ep = load_episode_data(episode_dirs[0])
    action_dim = first_ep['actions'].shape[1]
    img_h, img_w = args.resize[1], args.resize[0]
    
    print(f"Action dimension: {action_dim}")
    print(f"Image size: {img_w}x{img_h}")
    
    # Create Zarr arrays
    # observations/rgb
    # actions
    # language_instruction
    # episode_ends (optional but good for indexing)
    
    obs_group = root.create_group('observations')
    rgb_arr = obs_group.zeros('rgb', shape=(0, img_h, img_w, 3), chunks=(100, img_h, img_w, 3), dtype='uint8')
    action_arr = root.zeros('actions', shape=(0, action_dim), chunks=(100, action_dim), dtype='float32')
    # Instructions are per episode usually, but dataset.py expects per-timestep or handled via index mapping.
    # The current dataset.py implementation handles:
    # "instructions = self.data.get('language_instruction', None)"
    # and expects it to be a list of strings matching the length of data OR a list of strings per episode?
    # Looking at dataset.py:
    # "instructions = [str(inst) for inst in instructions]"
    # "instructions[0] if len(instructions) == 1 else instructions"
    # It seems it expects a list of strings aligned with the data if it's not length 1.
    # So we should replicate the instruction for every timestep.
    
    inst_arr = root.zeros('language_instruction', shape=(0,), chunks=(100,), dtype=object, object_codec=None)
    
    total_steps = 0
    
    for ep_dir in tqdm(episode_dirs, desc="Converting episodes"):
        ep_data = load_episode_data(ep_dir)
        
        # Resize images
        resized_imgs = []
        for img in ep_data['images']:
            resized_imgs.append(cv2.resize(img, (img_w, img_h)))
        resized_imgs = np.stack(resized_imgs)
        
        num_steps = len(resized_imgs)
        
        # Append to Zarr
        rgb_arr.append(resized_imgs)
        action_arr.append(ep_data['actions'])
        
        # Replicate instruction
        inst_list = [ep_data['instruction']] * num_steps
        inst_arr.append(np.array(inst_list, dtype=object))
        
        total_steps += num_steps
        
    print(f"Conversion complete!")
    print(f"Total steps: {total_steps}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

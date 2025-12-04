"""
Script to download and process ManiSkill2 demonstrations for LCDP-Sim.
Downloads official demos and converts them to the Zarr format with RGB observations.
"""

import os
import argparse
import json
import numpy as np
import h5py
import zarr
import gymnasium as gym
from tqdm import tqdm
import cv2
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download and process ManiSkill2 demos")
    parser.add_argument('--env', type=str, default='PickCube-v0', help='ManiSkill2 environment ID')
    parser.add_argument('--output', type=str, default='data/pick_cube_demos.zarr', help='Output Zarr path')
    parser.add_argument('--num-episodes', type=int, default=None, help='Number of episodes to process (default: all)')
    parser.add_argument('--download-dir', type=str, default='data/maniskill_demos', help='Directory to download raw demos')
    parser.add_argument('--obs-mode', type=str, default='rgbd', help='Observation mode for replay')
    parser.add_argument('--img-size', type=int, default=224, help='Image size for output')
    
    args = parser.parse_args()
    
    # 1. Download Demos
    print(f"Checking for demonstrations for {args.env}...")
    try:
        # Try importing the main function from the script
        from mani_skill2.utils.download_demo import main as download_demo_main
        
        class DownloadArgs:
            def __init__(self, uid, output_dir, quiet=False, download_version=None):
                self.uid = uid
                self.output_dir = output_dir
                self.quiet = quiet
                self.download_version = download_version
        
        # Create download dir
        Path(args.download_dir).mkdir(parents=True, exist_ok=True)
        
        # Download
        download_args = DownloadArgs(uid=args.env, output_dir=args.download_dir)
        download_demo_main(download_args)
        
        # Construct path based on download logic
        # It downloads to output_dir/v0/rigid_body/PickCube-v0/trajectory.h5 usually
        # Let's find it
        demo_path = None
        for path in Path(args.download_dir).rglob("trajectory.h5"):
            if args.env in str(path):
                demo_path = str(path)
                break
                
        if demo_path:
            print(f"Demonstrations located at: {demo_path}")
        else:
            raise FileNotFoundError("Download finished but trajectory.h5 not found")
        
    except ImportError:
        print("Error: mani_skill2 not installed or download_demo failed.")
        print("Please install mani-skill2: pip install mani-skill2")
        return
    except Exception as e:
        print(f"Download failed: {e}")
        # Try to guess path if already exists
        possible_path = Path(args.download_dir) / "v0" / args.env / "trajectory.h5"
        if possible_path.exists():
            print(f"Found existing demo at {possible_path}")
            demo_path = str(possible_path)
        else:
            print("Could not download or find demonstrations.")
            return

    # 2. Process Demos
    print(f"Processing demonstrations from {demo_path}...")
    
    # Open H5 file
    h5_file = h5py.File(demo_path, 'r')
    
    # Get env info
    import json
    if 'env_info' in h5_file.attrs:
        env_info = json.loads(h5_file.attrs['env_info'])
    else:
        # Fallback for some versions of datasets
        print("Warning: 'env_info' attribute not found. Using default configuration.")
        env_info = {}
        
    print(f"Env Info: {env_info}")
    
    # Determine control mode from metadata if possible, else use default
    # ManiSkill2 demos usually record the control mode used
    control_mode = env_info.get('control_mode', 'pd_ee_delta_pose')
    print(f"Using control mode: {control_mode}")
    
    # Create Environment
    # We need 'rgbd' or 'rgb' mode to get images
    # We use the same control mode as the demo
    import mani_skill2.envs
    
    # Handle WSL rendering issue
    import os
    if "WSL_DISTRO_NAME" in os.environ:
        print("Detected WSL environment. Attempting to use CPU rendering (software rasterization).")
        os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
        os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"
        os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
        # For Vulkan (SAPIEN uses Vulkan), we might need llvmpipe
        # But SAPIEN renderer on WSL is tricky.
        # Try to force CPU if possible, but SAPIEN relies on Vulkan.
        # If no GPU is available to WSL, this might fail.
    
    try:
        env = gym.make(
            args.env,
            obs_mode=args.obs_mode,
            control_mode=control_mode,
            render_mode="rgb_array"
        )
    except RuntimeError as e:
        if "ErrorExtensionNotPresent" in str(e) or "Vulkan" in str(e):
            print("\nCRITICAL ERROR: Vulkan initialization failed.")
            print("This is likely because you are running in WSL without proper GPU passthrough or drivers.")
            print("ManiSkill2 requires a Vulkan-capable GPU to render images.")
            print("Please ensure you have installed the latest NVIDIA drivers on Windows and are using WSL2.")
            print("Alternatively, try running this on a native Linux machine or Windows directly.")
            return
        raise e
    
    # Prepare Output Zarr
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    
    # Create datasets
    # We'll use expandable arrays
    img_size = args.img_size
    
    obs_group = root.create_group('observations')
    rgb_arr = obs_group.zeros('rgb', shape=(0, img_size, img_size, 3), chunks=(100, img_size, img_size, 3), dtype='uint8')
    action_arr = root.zeros('actions', shape=(0, env.action_space.shape[0]), chunks=(100, env.action_space.shape[0]), dtype='float32')
    inst_arr = root.zeros('language_instruction', shape=(0,), chunks=(100,), dtype=object, object_codec=None)
    ep_len_arr = root.zeros('episode_lengths', shape=(0,), chunks=(100,), dtype='int64')
    
    # Get list of episodes
    episodes = list(h5_file.keys())
    if args.num_episodes:
        episodes = episodes[:args.num_episodes]
        
    total_steps = 0
    success_count = 0
    
    # Instruction (ManiSkill2 tasks usually have a fixed instruction or we can generate one)
    # For PickCube, it's "Pick up the cube"
    # We can try to get it from env if supported, or hardcode based on task
    default_instruction = "Pick up the target object"
    if "PickCube" in args.env:
        default_instruction = "Pick up the red cube"
    elif "Stack" in args.env:
        default_instruction = "Stack the cube"
    
    for ep_id in tqdm(episodes, desc="Replaying episodes"):
        ep_grp = h5_file[ep_id]
        
        # Load metadata
        # ManiSkill2 demos store metadata in json format in the group attributes or a dataset
        # Usually 'json_meta' dataset? No, usually attributes of the group or passed in reset
        
        # Try to get seed
        # In ManiSkill2 h5 format:
        # ep_grp.attrs['episode_id']
        # But reset usually needs seed.
        # Let's look for 'env_states' to be safe, but reset(seed=...) is better for visual diversity if we had it.
        # The official demos are generated with seeds.
        # We can try to extract seed from ep_id if it's like 'traj_0' -> seed 0? Not always.
        
        # Check if 'json_meta' exists (common in some versions)
        episode_seed = None
        if 'json_meta' in ep_grp.keys():
            import json
            meta = json.loads(ep_grp['json_meta'][()])
            episode_seed = meta.get('episode_seed', None)
            
        if episode_seed is None:
            # Fallback: try to parse from name or just skip if we can't reproduce
            # Some datasets just have state trajectories.
            # If we have env_states, we can use them.
            pass
            
        # Reset Env
        if episode_seed is not None:
            obs, info = env.reset(seed=episode_seed)
        else:
            # If no seed, we might be in trouble for exact reproduction unless we set state
            # Try to load first state
            if 'env_states' in ep_grp:
                env_states = ep_grp['env_states'][()]
                env.reset()
                env.set_state(env_states[0])
                obs = env.get_obs()
            else:
                print(f"Skipping {ep_id}: No seed or env_states found")
                continue
        
        # Replay
        # We need actions
        actions = ep_grp['actions'][()]
        
        episode_imgs = []
        episode_actions = []
        
        # Record first observation
        # Process image
        img = obs['image']['base_camera']['rgb'] # Adjust camera name if needed
        # ManiSkill2 default camera might be 'base_camera' or 'hand_camera' or others.
        # 'rgbd' mode returns a dict of cameras.
        # Let's pick a main camera.
        
        # Find a suitable camera
        cam_name = None
        for name in obs['image'].keys():
            if 'base' in name or 'overhead' in name or 'front' in name:
                cam_name = name
                break
        if cam_name is None:
            cam_name = list(obs['image'].keys())[0]
            
        img = obs['image'][cam_name]['rgb']
        img = cv2.resize(img, (img_size, img_size))
        episode_imgs.append(img)
        
        # Step through actions
        for i, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record image
            img = obs['image'][cam_name]['rgb']
            img = cv2.resize(img, (img_size, img_size))
            episode_imgs.append(img)
            
            episode_actions.append(action)
            
            if terminated or truncated:
                break
                
        # Save episode
        # Note: len(images) = len(actions) + 1 usually (initial obs)
        # But dataset.py expects aligned pairs.
        # Usually we discard the last observation or align them: obs[t], action[t] -> obs[t+1]
        # We'll keep len(actions) observations.
        
        episode_imgs = np.array(episode_imgs[:len(episode_actions)])
        episode_actions = np.array(episode_actions)
        
        # Append to Zarr
        rgb_arr.append(episode_imgs)
        action_arr.append(episode_actions)
        
        # Instructions
        inst_list = [default_instruction] * len(episode_actions)
        inst_arr.append(np.array(inst_list, dtype=object))
        
        ep_len_arr.append(np.array([len(episode_actions)]))
        
        total_steps += len(episode_actions)
        success_count += 1
        
    env.close()
    h5_file.close()
    
    print(f"Processing complete!")
    print(f"Processed {success_count} episodes")
    print(f"Total steps: {total_steps}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()

"""
Unit tests for dataset and data loading
"""

import torch
import numpy as np
import zarr
import tempfile
import os
from pathlib import Path

from lcdp.data.dataset import RobotDataset, collate_fn, create_dataloader


class TestRobotDataset:
    """Test robot dataset"""
    
    def test_dataset_creation(self):
        """Test creating dataset from zarr file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test_data.zarr")
            
            # Create dummy data
            root = zarr.open(data_path, mode='w')
            num_timesteps = 50
            root.create_dataset(
                'observations/rgb',
                data=np.random.randint(0, 255, (num_timesteps, 64, 64, 3), dtype=np.uint8)
            )
            root.create_dataset(
                'actions',
                data=np.random.randn(num_timesteps, 7).astype(np.float32)
            )
            root.create_dataset(
                'language_instruction',
                data=np.array(["Pick up the cube"], dtype=object)
            )
            
            # Create dataset
            dataset = RobotDataset(
                data_path=data_path,
                horizon=16,
                action_horizon=16,
                image_size=(224, 224),
                normalize_actions=True,
                file_format="zarr"
            )
            
            assert len(dataset) > 0
            
            # Test __getitem__
            sample = dataset[0]
            assert 'image' in sample
            assert 'actions' in sample
            assert 'instruction' in sample
            
            assert sample['image'].shape == (3, 224, 224)
            assert sample['actions'].shape == (7, 16)
            assert isinstance(sample['instruction'], str)
    
    def test_action_normalization(self):
        """Test action normalization"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test_data.zarr")
            
            root = zarr.open(data_path, mode='w')
            actions = np.array([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]] * 50, dtype=np.float32)
            
            root.create_dataset('observations/rgb', 
                              data=np.random.randint(0, 255, (50, 64, 64, 3), dtype=np.uint8))
            root.create_dataset('actions', data=actions)
            root.create_dataset('language_instruction', data=np.array(["test"], dtype=object))
            
            dataset = RobotDataset(
                data_path=data_path,
                normalize_actions=True,
                file_format="zarr"
            )
            
            # Check normalization stats exist
            assert hasattr(dataset, 'action_mean')
            assert hasattr(dataset, 'action_std')
            
            # Test denormalization
            normalized = dataset.normalize_action(actions[0])
            denormalized = dataset.denormalize_action(normalized)
            np.testing.assert_allclose(denormalized, actions[0], rtol=1e-5)
    
    def test_collate_fn(self):
        """Test custom collate function"""
        batch = [
            {
                'image': torch.randn(3, 224, 224),
                'actions': torch.randn(7, 16),
                'instruction': 'Instruction 1'
            },
            {
                'image': torch.randn(3, 224, 224),
                'actions': torch.randn(7, 16),
                'instruction': 'Instruction 2'
            }
        ]
        
        collated = collate_fn(batch)
        
        assert collated['image'].shape == (2, 3, 224, 224)
        assert collated['actions'].shape == (2, 7, 16)
        assert len(collated['instruction']) == 2
        assert isinstance(collated['instruction'], list)


class TestDataLoader:
    """Test data loader creation"""
    
    def test_dataloader_creation(self):
        """Test creating a dataloader"""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "test_data.zarr")
            
            # Create dummy data
            root = zarr.open(data_path, mode='w')
            root.create_dataset('observations/rgb', 
                              data=np.random.randint(0, 255, (100, 64, 64, 3), dtype=np.uint8))
            root.create_dataset('actions', 
                              data=np.random.randn(100, 7).astype(np.float32))
            root.create_dataset('language_instruction', 
                              data=np.array(["Pick cube"], dtype=object))
            
            # Create dataloader
            dataloader = create_dataloader(
                data_path=data_path,
                batch_size=4,
                shuffle=True,
                num_workers=0,
                horizon=16,
                action_horizon=16
            )
            
            # Test iteration
            for batch in dataloader:
                assert batch['image'].shape[0] <= 4  # Batch size
                assert batch['image'].shape[1:] == (3, 224, 224)
                assert batch['actions'].shape[1:] == (7, 16)
                break


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

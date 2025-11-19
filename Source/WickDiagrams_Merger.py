import h5py
import os

def merge_hdf5_files(output_path=None, input_paths=None):
    """
    Merge multiple HDF5 files into one, keeping each file's complete structure 
    under its filename as a group.
    Each input file becomes a top-level group in the output file, preserving
    all internal hierarchies, datasets, and attributes.
    """
    def copy_structure(source, dest):
        """Recursively copy all items (datasets, groups, attributes) from source to dest."""
        # Copy attributes
        for attr_name, attr_value in source.attrs.items():
            dest.attrs[attr_name] = attr_value
        
        # Copy all items
        for key in source.keys():
            item = source[key]
            if isinstance(item, h5py.Dataset):
                # Copy dataset with data and attributes
                dest.create_dataset(key, data=item[()], dtype=item.dtype)
                for attr_name, attr_value in item.attrs.items():
                    dest[key].attrs[attr_name] = attr_value
            elif isinstance(item, h5py.Group):
                # Create subgroup and recurse
                subgroup = dest.create_group(key)
                copy_structure(item, subgroup)
    
    with h5py.File(output_path, 'w') as merged_file:
        for input_path in input_paths:
            # Use filename without extension as group name
            group_name = os.path.splitext(os.path.basename(input_path))[0]
            group = merged_file.create_group(group_name)
            
            with h5py.File(input_path, 'r') as source_file:
                # Recursively copy all items from source root to new group
                copy_structure(source_file, group)
    
    print(f'Successfully merged {len(input_paths)} files into {output_path}')
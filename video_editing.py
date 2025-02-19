import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

from sam2.build_sam import build_sam2_video_predictor

def create_masked_image(image, mask, color):
    """
    Create a masked image with colored overlay for the segmented object.
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Convert mask to numpy if it's a tensor
    if hasattr(mask, 'numpy'):
        mask = mask.numpy()
    
    # Get mask dimensions
    h, w = mask.shape[-2:]
    
    # Add alpha to color
    color_with_alpha = np.append(color, 153)  # Alpha=0.6 (153/255)
    color_with_alpha = color_with_alpha / 255.0  # Normalize to [0, 1]
    
    # Reshape mask and create colored mask
    mask_image = mask.reshape(h, w, 1) * color_with_alpha.reshape(1, 1, -1)
    
    # Create the final overlay
    alpha = mask_image[..., 3:4]  # Get alpha channel
    colored_mask = mask_image[..., :3]  # Get RGB channels
    overlay = image * (1 - alpha) + colored_mask * alpha * 255
    
    # Convert to uint8 and then to PIL Image
    overlay_uint8 = overlay.astype(np.uint8)
    return Image.fromarray(overlay_uint8)

def save_combined_binary_mask(masks, output_path):
    """
    Combine multiple masks and save as a single binary PNG file where any masked region is white (255).
    """
    # Convert each mask to numpy if needed
    masks = [mask.numpy() if hasattr(mask, 'numpy') else mask for mask in masks]
    
    # Combine all masks using logical OR
    combined_mask = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        combined_mask = combined_mask | (mask > 0.5)
    
    # Convert to uint8 binary format (0 or 255)
    binary_mask = combined_mask.astype(np.uint8) * 255
    
    # Remove any extra dimensions
    binary_mask = np.squeeze(binary_mask)
    
    # Convert to PIL Image and save as PNG
    mask_image = Image.fromarray(binary_mask, mode='L')
    mask_image.save(output_path, format='PNG')

def get_two_points_from_mask(mask):
    """
    Get two center points from the mask: one at ~1/3 and one at ~2/3 of the mask's height.
    """
    # Find all non-zero points
    rows, cols = np.nonzero(mask)
    if len(rows) == 0 or len(cols) == 0:
        return None
        
    # Get bounding box
    top = rows.min()
    bottom = rows.max()
    
    # Calculate target y-coordinates at 1/3 and 2/3 of height
    height = bottom - top
    y1 = top + height // 3
    y2 = top + (2 * height) // 3
    
    # Find x-coordinates for each y position
    def find_center_x_at_y(y_coord):
        # Get all x-coordinates at this y level
        x_coords = cols[rows == y_coord]
        if len(x_coords) == 0:
            # If no points at this y, look at nearby y coordinates
            for offset in range(1, 6):  # Look up to 5 pixels above and below
                # Look above
                x_coords = cols[rows == (y_coord - offset)]
                if len(x_coords) > 0:
                    break
                # Look below
                x_coords = cols[rows == (y_coord + offset)]
                if len(x_coords) > 0:
                    break
        
        if len(x_coords) == 0:
            return None
            
        # Return center x coordinate
        return (x_coords.min() + x_coords.max()) // 2
    
    # Get x-coordinates for both points
    x1 = find_center_x_at_y(y1)
    x2 = find_center_x_at_y(y2)
    
    if x1 is None or x2 is None:
        return None
    
    return np.array([[x1, y1], [x2, y2]], dtype=np.float32)

def process_video_in_batches(predictor, video_dir, mask_dir, overlay_dir, batch_size=100, start_frame=343,end_frame=1122):
    """
    Process video frames in batches with continuous tracking between batches, starting from frame 343.
    """
    # Initial object configurations
    initial_objects_config = [
        {
            'obj_id': 1,
            'points': np.array([[550, 380], [550, 330]], dtype=np.float32),
            'labels': np.array([1, 1], np.int32),
            'color': np.array([255, 165, 0])  # Orange
        },
        {
            'obj_id': 2,
            'points': np.array([[1250, 570], [1280, 450]], dtype=np.float32),
            'labels': np.array([1, 1], np.int32),
            'color': np.array([0, 255, 255])  # Cyan
        }
    ]
    
    # Scan all JPEG frames and filter for frames >= start_frame
    all_frames = sorted(
        [p for p in os.listdir(video_dir) if p.lower().endswith(('.jpg', '.jpeg'))],
        key=lambda p: int(os.path.splitext(p)[0])
    )
    
    # Filter frames starting from frame 343
    frame_names = [f for f in all_frames if int(os.path.splitext(f)[0]) >= start_frame and int(os.path.splitext(f)[0]) <= end_frame]
    
    if not frame_names:
        print(f"No frames found starting from {start_frame:05d}.jpg")
        return
    
    print(f"Starting processing from frame {start_frame:05d}.jpg")
    print(f"Total frames to process: {len(frame_names)}")
    
    # Calculate number of batches
    num_frames = len(frame_names)
    num_batches = (num_frames + batch_size - 1) // batch_size
    
    # Keep track of last frame's object positions
    current_objects_config = initial_objects_config
    
    # Process each batch
    for batch_idx in range(num_batches):
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")
        
        # Calculate batch frame range
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_frames)
        batch_frames = frame_names[start_idx:end_idx]
        
        # Create temporary directory for batch frames
        batch_dir = Path("temp_batch_frames")
        batch_dir.mkdir(exist_ok=True)
        
        # Create symbolic links to batch frames
        for frame_name in batch_frames:
            src = video_dir / frame_name
            dst = batch_dir / frame_name
            if not dst.exists():
                os.symlink(src, dst)
        
        # Initialize inference state for this batch
        inference_state = predictor.init_state(video_path=str(batch_dir))
        
        # Initialize segmentation for each object using current configurations
        for obj_config in current_objects_config:
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_config['obj_id'],
                points=obj_config['points'],
                labels=obj_config['labels'],
            )
        
        # Process frames in this batch
        video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        
        # Save results for this batch
        last_frame_masks = {}
        for local_idx, frame_name in enumerate(batch_frames):
            frame_path = video_dir / frame_name
            original_image = Image.open(frame_path)
            # Calculate the global frame number by adding start_frame and local_idx
            frame_number = start_idx + local_idx + start_frame
            
            # Get masks for this frame
            frame_masks = video_segments[local_idx]
            
            # Create masked image and collect masks
            final_image = original_image
            all_masks = []
            
            # Keep track of last frame's masks
            if local_idx == len(batch_frames) - 1:
                last_frame_masks = frame_masks
            
            for obj_config in current_objects_config:
                obj_id = obj_config['obj_id']
                if obj_id in frame_masks:
                    mask = frame_masks[obj_id]
                    all_masks.append(mask)
                    final_image = create_masked_image(final_image, mask, obj_config['color'])
            
            # Save mask and overlay using original frame number
            mask_output_path = mask_dir / f"{frame_number:05d}.png"
            overlay_output_path = overlay_dir / f"{frame_number:05d}.jpg"
            
            save_combined_binary_mask(all_masks, mask_output_path)
            final_image.save(overlay_output_path, quality=95)
            
            print(f"Processed frame {frame_number:05d}")
        
        # Update object configurations for next batch using last frame's masks
        if batch_idx < num_batches - 1:  # Skip for last batch
            new_objects_config = []
            for obj_config in current_objects_config:
                obj_id = obj_config['obj_id']
                if obj_id in last_frame_masks:
                    mask = last_frame_masks[obj_id]
                    points = get_two_points_from_mask(mask[0])  # mask[0] to remove batch dimension
                    if points is not None:
                        new_config = obj_config.copy()
                        new_config['points'] = points
                        new_config['labels'] = np.array([1, 1], np.int32)  # Keep both as positive points
                        new_objects_config.append(new_config)
            
            if new_objects_config:  # Only update if we found valid masks
                current_objects_config = new_objects_config
        
        # Clean up temporary batch directory
        for link in batch_dir.iterdir():
            os.unlink(link)
        batch_dir.rmdir()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main(args):
    overlay_dir = Path(args.overlay_dir)
    mask_dir = Path(args.mask_dir)
    video_dir = Path(args.video_dir)
    
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Model configuration
    sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    
    # Initialize predictor
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    # Process video in batches, starting from frame 343
    process_video_in_batches(
        predictor=predictor,
        video_dir=video_dir,
        mask_dir=mask_dir,
        overlay_dir=overlay_dir,
        batch_size=args.batch_size,
        start_frame=0,
        end_frame=1500
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video frame segmentation with SAM2')
    parser.add_argument('--video_dir', type=str, required=True,
                      help='Directory containing video frames')
    parser.add_argument('--mask_dir', type=str, required=True,
                      help='Directory to save segmentation masks')
    parser.add_argument('--overlay_dir', type=str, required=True,
                      help='Directory to save overlay visualizations')
    parser.add_argument('--batch_size', type=int, default=100,
                      help='Number of frames to process in each batch')
    
    args = parser.parse_args()
    main(args)
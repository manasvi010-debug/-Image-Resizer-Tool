# -Image-Resizer-Tool
# Create a comprehensive batch image resizer script with documentation
import os

script_content = '''#!/usr/bin/env python3
"""
Batch Image Resizer and Converter Tool
=======================================

This script resizes and converts all images in a specified folder.

Features:
- Batch process all images in a folder
- Resize by exact dimensions or scale factor
- Convert between formats (JPEG, PNG, BMP, GIF, etc.)
- Preserve or discard aspect ratio
- Save to output folder with original structure
- High-quality resampling (Lanczos filter)

Requirements:
- Python 3.x
- Pillow library: pip install pillow

Author: AI Assistant for VIT Student
Date: November 2025
"""

import os
import sys
from PIL import Image
from pathlib import Path


class ImageResizerTool:
    """Handles batch image resizing and format conversion"""
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    def __init__(self, input_folder, output_folder=None):
        """
        Initialize the image resizer tool.
        
        Args:
            input_folder (str): Path to folder containing images
            output_folder (str): Path to save resized images (default: input_folder/resized)
        """
        self.input_folder = Path(input_folder)
        
        if not self.input_folder.exists():
            raise FileNotFoundError(f"Input folder not found: {input_folder}")
        
        # Set output folder
        if output_folder:
            self.output_folder = Path(output_folder)
        else:
            self.output_folder = self.input_folder / "resized"
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Input folder: {self.input_folder}")
        print(f"Output folder: {self.output_folder}")
    
    def get_image_files(self):
        """
        Get all image files from input folder.
        
        Returns:
            list: List of Path objects for image files
        """
        image_files = []
        for file_path in self.input_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                image_files.append(file_path)
        
        return sorted(image_files)
    
    def resize_by_dimensions(self, target_width, target_height, maintain_aspect=True, 
                            output_format=None, quality=95):
        """
        Resize images to exact dimensions.
        
        Args:
            target_width (int): Target width in pixels
            target_height (int): Target height in pixels
            maintain_aspect (bool): Maintain aspect ratio (default: True)
            output_format (str): Output format (e.g., 'JPEG', 'PNG'). None = keep original
            quality (int): Output quality for JPEG (1-100, default: 95)
        
        Returns:
            dict: Summary of processing results
        """
        image_files = self.get_image_files()
        
        if not image_files:
            print("No images found in input folder!")
            return {"total": 0, "success": 0, "failed": 0}
        
        print(f"\\nFound {len(image_files)} images to process")
        print(f"Target dimensions: {target_width}x{target_height}")
        print(f"Maintain aspect ratio: {maintain_aspect}")
        print(f"Output format: {output_format or 'Original'}\\n")
        
        success_count = 0
        failed_count = 0
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                # Open image
                with Image.open(image_path) as img:
                    original_size = img.size
                    original_format = img.format
                    
                    # Calculate new size
                    if maintain_aspect:
                        # Calculate aspect ratio
                        img.thumbnail((target_width, target_height), Image.LANCZOS)
                        new_size = img.size
                    else:
                        # Resize to exact dimensions
                        img = img.resize((target_width, target_height), Image.LANCZOS)
                        new_size = (target_width, target_height)
                    
                    # Determine output filename and format
                    if output_format:
                        output_name = image_path.stem + '.' + output_format.lower()
                        save_format = output_format.upper()
                    else:
                        output_name = image_path.name
                        save_format = original_format
                    
                    output_path = self.output_folder / output_name
                    
                    # Save image
                    if save_format in ['JPEG', 'JPG']:
                        # Convert RGBA to RGB for JPEG
                        if img.mode in ('RGBA', 'LA', 'P'):
                            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                            img = rgb_img
                        img.save(output_path, format='JPEG', quality=quality, optimize=True)
                    else:
                        img.save(output_path, format=save_format, optimize=True)
                    
                    print(f"[{idx}/{len(image_files)}] ✓ {image_path.name}")
                    print(f"    {original_size[0]}x{original_size[1]} → {new_size[0]}x{new_size[1]}")
                    
                    success_count += 1
            
            except Exception as e:
                print(f"[{idx}/{len(image_files)}] ✗ {image_path.name}")
                print(f"    Error: {str(e)}")
                failed_count += 1
        
        # Summary
        print(f"\\n{'='*60}")
        print(f"Processing complete!")
        print(f"Total images: {len(image_files)}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"{'='*60}\\n")
        
        return {
            "total": len(image_files),
            "success": success_count,
            "failed": failed_count
        }
    
    def resize_by_scale(self, scale_factor, output_format=None, quality=95):
        """
        Resize images by a scale factor.
        
        Args:
            scale_factor (float): Scale factor (e.g., 0.5 = 50%, 2.0 = 200%)
            output_format (str): Output format (e.g., 'JPEG', 'PNG'). None = keep original
            quality (int): Output quality for JPEG (1-100, default: 95)
        
        Returns:
            dict: Summary of processing results
        """
        image_files = self.get_image_files()
        
        if not image_files:
            print("No images found in input folder!")
            return {"total": 0, "success": 0, "failed": 0}
        
        print(f"\\nFound {len(image_files)} images to process")
        print(f"Scale factor: {scale_factor}x ({int(scale_factor * 100)}%)")
        print(f"Output format: {output_format or 'Original'}\\n")
        
        success_count = 0
        failed_count = 0
        
        for idx, image_path in enumerate(image_files, 1):
            try:
                # Open image
                with Image.open(image_path) as img:
                    original_size = img.size
                    original_format = img.format
                    
                    # Calculate new dimensions
                    new_width = int(img.width * scale_factor)
                    new_height = int(img.height * scale_factor)
                    
                    # Resize image
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Determine output filename and format
                    if output_format:
                        output_name = image_path.stem + '.' + output_format.lower()
                        save_format = output_format.upper()
                    else:
                        output_name = image_path.name
                        save_format = original_format
                    
                    output_path = self.output_folder / output_name
                    
                    # Save image
                    if save_format in ['JPEG', 'JPG']:
                        # Convert RGBA to RGB for JPEG
                        if img.mode in ('RGBA', 'LA', 'P'):
                            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                            rgb_img.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                            img = rgb_img
                        img.save(output_path, format='JPEG', quality=quality, optimize=True)
                    else:
                        img.save(output_path, format=save_format, optimize=True)
                    
                    print(f"[{idx}/{len(image_files)}] ✓ {image_path.name}")
                    print(f"    {original_size[0]}x{original_size[1]} → {new_width}x{new_height}")
                    
                    success_count += 1
            
            except Exception as e:
                print(f"[{idx}/{len(image_files)}] ✗ {image_path.name}")
                print(f"    Error: {str(e)}")
                failed_count += 1
        
        # Summary
        print(f"\\n{'='*60}")
        print(f"Processing complete!")
        print(f"Total images: {len(image_files)}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed: {failed_count}")
        print(f"{'='*60}\\n")
        
        return {
            "total": len(image_files),
            "success": success_count,
            "failed": failed_count
        }


def main():
    """Main function with command-line interface"""
    
    print("=" * 60)
    print("Batch Image Resizer and Converter Tool")
    print("=" * 60)
    
    # Check if running from command line with arguments
    if len(sys.argv) > 1:
        # Command-line mode
        if len(sys.argv) < 3:
            print("\\nUsage:")
            print("  Resize by dimensions:")
            print("    python image_resizer.py <input_folder> <width> <height> [output_folder] [format] [quality]")
            print("\\n  Resize by scale:")
            print("    python image_resizer.py <input_folder> scale <factor> [output_folder] [format] [quality]")
            print("\\nExample:")
            print("    python image_resizer.py ./photos 800 600")
            print("    python image_resizer.py ./photos scale 0.5 ./output jpeg 85")
            sys.exit(1)
        
        input_folder = sys.argv[1]
        
        if sys.argv[2].lower() == 'scale':
            # Scale mode
            scale_factor = float(sys.argv[3])
            output_folder = sys.argv[4] if len(sys.argv) > 4 else None
            output_format = sys.argv[5].upper() if len(sys.argv) > 5 else None
            quality = int(sys.argv[6]) if len(sys.argv) > 6 else 95
            
            resizer = ImageResizerTool(input_folder, output_folder)
            resizer.resize_by_scale(scale_factor, output_format, quality)
        else:
            # Dimension mode
            width = int(sys.argv[2])
            height = int(sys.argv[3])
            output_folder = sys.argv[4] if len(sys.argv) > 4 else None
            output_format = sys.argv[5].upper() if len(sys.argv) > 5 else None
            quality = int(sys.argv[6]) if len(sys.argv) > 6 else 95
            
            resizer = ImageResizerTool(input_folder, output_folder)
            resizer.resize_by_dimensions(width, height, maintain_aspect=True, 
                                        output_format=output_format, quality=quality)
    else:
        # Interactive mode
        print("\\nInteractive Mode")
        print("-" * 60)
        
        # Get input folder
        input_folder = input("Enter input folder path: ").strip()
        
        # Get output folder (optional)
        output_folder = input("Enter output folder path (press Enter for default): ").strip()
        output_folder = output_folder if output_folder else None
        
        # Initialize resizer
        try:
            resizer = ImageResizerTool(input_folder, output_folder)
        except FileNotFoundError as e:
            print(f"\\nError: {e}")
            sys.exit(1)
        
        # Choose resize mode
        print("\\nResize mode:")
        print("1. Resize by dimensions (e.g., 800x600)")
        print("2. Resize by scale factor (e.g., 0.5 = 50%)")
        mode = input("Choose mode (1 or 2): ").strip()
        
        # Get format conversion option
        print("\\nFormat conversion (optional):")
        print("Supported: JPEG, PNG, BMP, GIF, TIFF, WEBP")
        output_format = input("Enter output format (press Enter to keep original): ").strip().upper()
        output_format = output_format if output_format else None
        
        # Get quality for JPEG
        quality = 95
        if output_format in ['JPEG', 'JPG']:
            quality_input = input("Enter JPEG quality (1-100, default 95): ").strip()
            quality = int(quality_input) if quality_input else 95
        
        # Process based on mode
        if mode == '1':
            width = int(input("Enter target width (pixels): "))
            height = int(input("Enter target height (pixels): "))
            maintain_aspect = input("Maintain aspect ratio? (y/n, default: y): ").strip().lower() != 'n'
            
            resizer.resize_by_dimensions(width, height, maintain_aspect, output_format, quality)
        
        elif mode == '2':
            scale_factor = float(input("Enter scale factor (e.g., 0.5 for 50%, 2.0 for 200%): "))
            
            resizer.resize_by_scale(scale_factor, output_format, quality)
        
        else:
            print("Invalid mode selected!")
            sys.exit(1)


if __name__ == "__main__":
    main()
'''

# Save the script
with open('image_resizer.py', 'w', encoding='utf-8') as f:
    f.write(script_content)

print("✓ Batch Image Resizer script created: image_resizer.py")
print("\nScript features:")
print("- Resize by exact dimensions or scale factor")
print("- Maintain or discard aspect ratio")
print("- Convert between formats (JPEG, PNG, BMP, GIF, etc.)")
print("- High-quality Lanczos resampling")
print("- Both interactive and command-line modes")
print("- Comprehensive error handling")
print("\nUsage examples:")
print("1. Interactive mode: python image_resizer.py")

## Version History

**v1.0** (November 2025)
- Initial release
- Dimension and scale resizing
- Format conversion
- Interactive and CLI modes
- Comprehensive error handling
'''

# Save README
with open('README_IMAGE_RESIZER.md', 'w', encoding='utf-8') as f:
    f.write(readme_content)

print("✓ Comprehensive README created: README_IMAGE_RESIZER.md")
print("\nDocumentation includes:")
print("- Installation instructions")
print("- Usage examples (interactive and CLI)")
print("- Feature descriptions")
print("- Troubleshooting guide")
print("- Best practices and tips")
print("- Common use cases")

# Create a simplified "quick start" script for beginners
quick_start = '''#!/usr/bin/env python3
"""
QUICK START: Simple Image Resizer
==================================
A beginner-friendly version for quick image resizing.

Usage: 
    python quick_resize.py

Requirements:
    pip install pillow
"""

from PIL import Image
import os
from pathlib import Path


def quick_resize():
    """Simple image resizer with basic options"""
    
    print("=" * 50)
    print("QUICK IMAGE RESIZER")
    print("=" * 50)
    
    # Get folder path
    folder = input("\\nEnter folder path (e.g., ./photos): ").strip()
    folder_path = Path(folder)
    
    if not folder_path.exists():
        print(f"Error: Folder '{folder}' not found!")
        return
    
    # Get all image files
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']:
        images.extend(folder_path.glob(ext))
        images.extend(folder_path.glob(ext.upper()))
    
    if not images:
        print("No images found in folder!")
        return
    
    print(f"\\nFound {len(images)} images")
    
    # Choose resize method
    print("\\nHow do you want to resize?")
    print("1. Make smaller (e.g., 50% size)")
    print("2. Set exact size (e.g., 800x600)")
    choice = input("Enter 1 or 2: ").strip()
    
    # Create output folder
    output_folder = folder_path / "resized"
    output_folder.mkdir(exist_ok=True)
    
    success = 0
    
    if choice == "1":
        # Resize by percentage
        percent = int(input("Enter percentage (e.g., 50 for half size): "))
        scale = percent / 100
        
        print(f"\\nResizing to {percent}%...")
        
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    new_width = int(img.width * scale)
                    new_height = int(img.height * scale)
                    
                    resized = img.resize((new_width, new_height), Image.LANCZOS)
                    
                    output_path = output_folder / img_path.name
                    resized.save(output_path, quality=95, optimize=True)
                    
                    print(f"✓ {img_path.name}")
                    success += 1
            except Exception as e:
                print(f"✗ {img_path.name}: {e}")
    
    elif choice == "2":
        # Resize to exact dimensions
        width = int(input("Enter width (pixels): "))
        height = int(input("Enter height (pixels): "))
        
        print(f"\\nResizing to {width}x{height}...")
        
        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    # Use thumbnail to maintain aspect ratio
                    img.thumbnail((width, height), Image.LANCZOS)
                    
                    output_path = output_folder / img_path.name
                    img.save(output_path, quality=95, optimize=True)
                    
                    print(f"✓ {img_path.name}")
                    success += 1
            except Exception as e:
                print(f"✗ {img_path.name}: {e}")
    
    else:
        print("Invalid choice!")
        return
    
    print(f"\\n{'='*50}")
    print(f"Done! Successfully resized {success}/{len(images)} images")
    print(f"Saved to: {output_folder}")
    print(f"{'='*50}")


if __name__ == "__main__":
    quick_resize()
'''

# Save quick start script
with open('quick_resize.py', 'w', encoding='utf-8') as f:
    f.write(quick_start)

print("✓ Quick start script created: quick_resize.py")
print("\nThis is a simplified version perfect for:")
print("- Beginners learning Python")
print("- Quick one-time resizing tasks")
print("- Understanding the basic workflow")

# Create a usage guide comparison table
import pandas as pd

# Comparison of different usage scenarios
usage_scenarios = {
    'Use Case': [
        'Web optimization',
        'Thumbnail generation',
        'Social media (Instagram)',
        'Social media (Facebook)',
        'Email attachments',
        'Print preparation',
        'Archive storage',
        'Product photos'
    ],
    'Recommended Size': [
        '1920x1080',
        '200x200',
        '1080x1080',
        '1200x630',
        '800x600',
        '3000x2000',
        '50% scale',
        '1000x1000'
    ],
    'Format': [
        'JPEG',
        'JPEG',
        'JPEG',
        'JPEG',
        'JPEG',
        'PNG/TIFF',
        'JPEG',
        'PNG'
    ],
    'Quality': [
        '85',
        '80',
        '90',
        '85',
        '75',
        '95',
        '70',
        '90'
    ],
    'Command': [
        'python image_resizer.py ./photos 1920 1080 ./web jpeg 85',
        'python image_resizer.py ./photos 200 200 ./thumbs jpeg 80',
        'python image_resizer.py ./photos 1080 1080 ./instagram jpeg 90',
        'python image_resizer.py ./photos 1200 630 ./facebook jpeg 85',
        'python image_resizer.py ./photos 800 600 ./email jpeg 75',
        'python image_resizer.py ./photos 3000 2000 ./print png',
        'python image_resizer.py ./photos scale 0.5 ./archive jpeg 70',
        'python image_resizer.py ./photos 1000 1000 ./products png 90'
    ]
}

df_scenarios = pd.DataFrame(usage_scenarios)
df_scenarios.to_csv('image_resizer_usage_scenarios.csv', index=False)

print("Image Resizer - Common Usage Scenarios")
print("=" * 120)
print(df_scenarios.to_string(index=False))
print("\n" + "=" * 120)

# Create technical specifications table
tech_specs = {
    'Resampling Filter': [
        'NEAREST',
        'BILINEAR',
        'BICUBIC',
        'LANCZOS (default)'
    ],
    'Speed': [
        'Fastest',
        'Fast',
        'Medium',
        'Slower'
    ],
    'Quality': [
        'Lowest',
        'Medium',
        'Good',
        'Best'
    ],
    'Use Case': [
        'Quick previews, pixel art',
        'General purpose, web graphics',
        'Photos, general images',
        'High-quality photos, professional work'
    ],
    'When to Use': [
        'Speed is critical, quality not important',
        'Balance between speed and quality',
        'Better quality needed, time available',
        'Maximum quality required (default)'
    ]
}

df_tech = pd.DataFrame(tech_specs)
df_tech.to_csv('resampling_filters_comparison.csv', index=False)

print("\nResampling Filter Comparison")
print("=" * 120)
print(df_tech.to_string(index=False))
print("\n" + "=" * 120)

# Create format comparison table
format_comparison = {
    'Format': ['JPEG', 'PNG', 'WEBP', 'BMP', 'GIF', 'TIFF'],
    'Transparency': ['No', 'Yes', 'Yes', 'No', 'Yes', 'Yes'],
    'Compression': ['Lossy', 'Lossless', 'Both', 'None', 'Lossless', 'Both'],
    'Best For': [
        'Photos, complex images',
        'Graphics, logos, transparency',
        'Modern web (smaller files)',
        'Uncompressed (large files)',
        'Simple animations',
        'Professional/print work'
    ],
    'Average File Size': [
        'Small-Medium',
        'Medium-Large',
        'Small',
        'Very Large',
        'Small-Medium',
        'Large-Very Large'
    ],
    'Browser Support': [
        'All browsers',
        'All browsers',
        'Modern browsers',
        'All browsers',
        'All browsers',
        'Limited'
    ]
}

df_formats = pd.DataFrame(format_comparison)
df_formats.to_csv('image_format_comparison.csv', index=False)

print("\nImage Format Comparison")
print("=" * 120)
print(df_formats.to_string(index=False))
print("\n" + "=" * 120)

print("\n✓ Three CSV files created:")
print("  1. image_resizer_usage_scenarios.csv")
print("  2. resampling_filters_comparison.csv")
print("  3. image_format_comparison.csv")

import plotly.graph_objects as go
import numpy as np

# Define node positions and properties
nodes = {
    'start': {'pos': (0, 10), 'text': 'User runs script', 'color': '#6BB6FF', 'shape': 'stadium'},
    'decision1': {'pos': (0, 9), 'text': 'Interactive or<br>CLI mode?', 'color': '#FFE680', 'shape': 'diamond'},
    'interactive': {'pos': (-2, 8), 'text': 'Prompt for inputs', 'color': '#6BB6FF', 'shape': 'rect'},
    'cli': {'pos': (2, 8), 'text': 'Parse arguments', 'color': '#6BB6FF', 'shape': 'rect'},
    'init': {'pos': (0, 7), 'text': 'Initialize<br>ImageResizerTool', 'color': '#6BB6FF', 'shape': 'rect'},
    'scan': {'pos': (0, 6), 'text': 'Scan folder for<br>images', 'color': '#DDA0DD', 'shape': 'rect'},
    'decision2': {'pos': (0, 5), 'text': 'Images found?', 'color': '#FFE680', 'shape': 'diamond'},
    'error_end': {'pos': (2, 5), 'text': 'Show error<br>message', 'color': '#FF9999', 'shape': 'stadium'},
    'decision3': {'pos': (0, 4), 'text': 'Resize mode?', 'color': '#FFE680', 'shape': 'diamond'},
    'dimensions': {'pos': (-2, 3), 'text': 'Calculate new size<br>(maintain aspect)', 'color': '#6BB6FF', 'shape': 'rect'},
    'scale': {'pos': (2, 3), 'text': 'Calculate new size<br>(scale factor)', 'color': '#6BB6FF', 'shape': 'rect'},
    'open': {'pos': (0, 2), 'text': 'Open image<br>with PIL', 'color': '#DDA0DD', 'shape': 'rect'},
    'resize': {'pos': (0, 1), 'text': 'Resize using<br>Lanczos filter', 'color': '#6BB6FF', 'shape': 'rect'},
    'decision4': {'pos': (0, 0), 'text': 'Format<br>conversion<br>needed?', 'color': '#FFE680', 'shape': 'diamond'},
    'convert': {'pos': (-2, -1), 'text': 'Convert format<br>(RGBA→RGB if JPEG)', 'color': '#6BB6FF', 'shape': 'rect'},
    'keep': {'pos': (2, -1), 'text': 'Keep original<br>format', 'color': '#6BB6FF', 'shape': 'rect'},
    'save': {'pos': (0, -2), 'text': 'Save optimized<br>image', 'color': '#DDA0DD', 'shape': 'rect'},
    'progress': {'pos': (0, -3), 'text': 'Update progress<br>counter', 'color': '#6BB6FF', 'shape': 'rect'},
    'decision5': {'pos': (0, -4), 'text': 'More images?', 'color': '#FFE680', 'shape': 'diamond'},
    'end': {'pos': (0, -5), 'text': 'Display summary<br>(success/failed)', 'color': '#90EE90', 'shape': 'stadium'},
}

# Define connections
connections = [
    ('start', 'decision1'),
    ('decision1', 'interactive', 'Interactive'),
    ('decision1', 'cli', 'CLI'),
    ('interactive', 'init'),
    ('cli', 'init'),
    ('init', 'scan'),
    ('scan', 'decision2'),
    ('decision2', 'error_end', 'No'),
    ('decision2', 'decision3', 'Yes'),
    ('decision3', 'dimensions', 'Dimensions'),
    ('decision3', 'scale', 'Scale'),
    ('dimensions', 'open'),
    ('scale', 'open'),
    ('open', 'resize'),
    ('resize', 'decision4'),
    ('decision4', 'convert', 'Yes'),
    ('decision4', 'keep', 'No'),
    ('convert', 'save'),
    ('keep', 'save'),
    ('save', 'progress'),
    ('progress', 'decision5'),
    ('decision5', 'open', 'Yes'),
    ('decision5', 'end', 'No'),
]

fig = go.Figure()

# Draw connections
for conn in connections:
    start_node = nodes[conn[0]]
    end_node = nodes[conn[1]]
    label = conn[2] if len(conn) > 2 else ''
    
    x0, y0 = start_node['pos']
    x1, y1 = end_node['pos']
    
    # Add line
    fig.add_trace(go.Scatter(
        x=[x0, x1],
        y=[y0, y1],
        mode='lines',
        line=dict(color='#13343B', width=2),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add arrow
    angle = np.arctan2(y1-y0, x1-x0)
    arrow_size = 0.15
    arrow_x = x1 - arrow_size * np.cos(angle)
    arrow_y = y1 - arrow_size * np.sin(angle)
    
    fig.add_annotation(
        x=x1, y=y1,
        ax=arrow_x, ay=arrow_y,
        xref='x', yref='y',
        axref='x', ayref='y',
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor='#13343B'
    )
    
    # Add label if exists
    if label:
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        fig.add_annotation(
            x=mid_x, y=mid_y,
            text=label,
            showarrow=False,
            bgcolor='white',
            font=dict(size=10, color='#13343B')
        )

# Draw nodes
for node_id, node_data in nodes.items():
    x, y = node_data['pos']
    
    # Determine symbol based on shape
    if node_data['shape'] == 'diamond':
        symbol = 'diamond'
        size = 40
    elif node_data['shape'] == 'stadium':
        symbol = 'circle'
        size = 35
    else:
        symbol = 'square'
        size = 35
    
    fig.add_trace(go.Scatter(
        x=[x],
        y=[y],
        mode='markers+text',
        marker=dict(
            size=size,
            color=node_data['color'],
            line=dict(color='#13343B', width=2),
            symbol=symbol
        ),
        text=node_data['text'],
        textposition='middle center',
        textfont=dict(size=9, color='#000000'),
        hoverinfo='skip',
        showlegend=False
    ))

fig.update_layout(
    title='Image Resizing Workflow',
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-3, 3]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-6, 11]),
    plot_bgcolor='#F3F3EE',
    paper_bgcolor='#F3F3EE'
)

fig.write_image('flowchart.png')
fig.write_image('flowchart.svg', format='svg')

import plotly.graph_objects as go
import pandas as pd

# Data
data = {
  "format_quality": ["Original", "JPEG Q100", "JPEG Q95", "JPEG Q85", "JPEG Q75", "PNG Lossless", "WEBP Q85", "BMP"],
  "file_size_mb": [5.0, 2.8, 1.5, 0.9, 0.6, 3.2, 0.7, 5.9]
}

df = pd.DataFrame(data)

# Assign colors based on file size
# Green: < 1.5 MB, Yellow: 1.5-3.0 MB, Red: > 3.0 MB
colors = []
for size in df['file_size_mb']:
    if size < 1.5:
        colors.append('#2E8B57')  # Green
    elif size <= 3.0:
        colors.append('#D2BA4C')  # Yellow
    else:
        colors.append('#DB4545')  # Red

# Create horizontal bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    y=df['format_quality'],
    x=df['file_size_mb'],
    orientation='h',
    marker=dict(color=colors),
    text=[f'{size} MB' for size in df['file_size_mb']],
    textposition='outside',
    cliponaxis=False
))

# Add vertical line at 1.5 MB
fig.add_vline(x=1.5, line_dash="dash", line_color="gray", line_width=2)

# Update layout
fig.update_layout(
    title="Image Format & Quality Comparison",
    xaxis_title="File Size (MB)",
    yaxis_title="Format/Quality",
    showlegend=False
)

fig.update_xaxes(range=[0, max(df['file_size_mb']) * 1.15])

# Save as PNG and SVG
fig.write_image("chart.png")
fig.write_image("chart.svg", format="svg")


print("2. CLI - dimensions: python image_resizer.py ./photos 800 600")
print("3. CLI - scale: python image_resizer.py ./photos scale 0.5")
print("4. CLI - with format: python image_resizer.py ./photos 1920 1080 ./output jpeg 85")

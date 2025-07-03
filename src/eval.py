import os
import argparse
import numpy as np
from PIL import Image

from model import SRCNN
from utils import load_image, preprocess_image, postprocess_image, calculate_psnr

def evaluate_model(args):
    """
    Evaluate the SRCNN model on test images and calculate PSNR scores.
    Compare SRCNN results with bicubic interpolation.
    """
    print("Starting evaluation...")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = SRCNN(input_channels=1)
    model.load_weights(args.model_path)
    
    # Get test images
    test_images = [f for f in os.listdir(args.test_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    # Table for results
    results = []
    
    for image_name in test_images:
        # Load and preprocess image
        img_path = os.path.join(args.test_dir, image_name)
        img = load_image(img_path)
        
        # Process image at different scale factors
        for scale in args.scale_factors:
            bicubic_y, hr_y, (cb, cr) = preprocess_image(img, scale)
            
            # Convert to proper shape for model
            bicubic_input = bicubic_y.reshape(1, 1, *bicubic_y.shape)
            
            # Forward pass
            output = model.forward(bicubic_input).squeeze()
            
            # Calculate PSNR
            psnr_bicubic = calculate_psnr(bicubic_y, hr_y)
            psnr_srcnn = calculate_psnr(output, hr_y)
            improvement = psnr_srcnn - psnr_bicubic
            
            # Save results to table
            results.append([image_name, scale, psnr_bicubic, psnr_srcnn, improvement])
            
            # Save comparison images if requested
            if args.save_images:
                # Create output directory for this scale factor
                scale_dir = os.path.join(args.output_dir, f"x{scale}")
                os.makedirs(scale_dir, exist_ok=True)
                
                # Reconstruct the full RGB images
                bicubic_img = postprocess_image(bicubic_y, cb, cr)
                srcnn_img = postprocess_image(output, cb, cr)
                
                # Save individual images
                img.save(os.path.join(scale_dir, f"{os.path.splitext(image_name)[0]}_original.png"))
                bicubic_img.save(os.path.join(scale_dir, f"{os.path.splitext(image_name)[0]}_bicubic_x{scale}.png"))
                srcnn_img.save(os.path.join(scale_dir, f"{os.path.splitext(image_name)[0]}_srcnn_x{scale}.png"))
    
    # Print results in a table format
    headers = ["Image", "Scale Factor", "PSNR (Bicubic)", "PSNR (SRCNN)", "Improvement"]
    print_table(results, headers)
    
    # Save results to CSV file
    results_file = os.path.join(args.output_dir, "psnr_results.csv")
    with open(results_file, 'w') as f:
        f.write(','.join(headers) + '\n')
        for row in results:
            f.write(','.join([str(item) for item in row]) + '\n')
    
    print(f"Results saved to {results_file}")
    
    # Calculate average results per scale factor
    scale_results = {}
    for row in results:
        scale = row[1]
        if scale not in scale_results:
            scale_results[scale] = {"bicubic": [], "srcnn": [], "improvement": []}
        
        scale_results[scale]["bicubic"].append(row[2])
        scale_results[scale]["srcnn"].append(row[3])
        scale_results[scale]["improvement"].append(row[4])
    
    print("\nAverage Results by Scale Factor:")
    avg_results = []
    for scale, data in scale_results.items():
        avg_bicubic = np.mean(data["bicubic"])
        avg_srcnn = np.mean(data["srcnn"])
        avg_improvement = np.mean(data["improvement"])
        avg_results.append([scale, avg_bicubic, avg_srcnn, avg_improvement])
    
    avg_headers = ["Scale Factor", "Avg PSNR (Bicubic)", "Avg PSNR (SRCNN)", "Avg Improvement"]
    print_table(avg_results, avg_headers)

def print_table(data, headers):
    """Simple function to print a table without external libraries."""
    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in data + [headers]) for i in range(len(headers))]
    
    # Print header
    header_str = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    print(header_str)
    print("-" * len(header_str))
    
    # Print rows
    for row in data:
        row_str = " | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(row))
        print(row_str)

def main():
    parser = argparse.ArgumentParser(description='SRCNN Evaluation')
    
    parser.add_argument('--test_dir', type=str, default='../Set5',
                       help='path to test images')
    parser.add_argument('--output_dir', type=str, default='../eval_results',
                       help='path to save evaluation results')
    parser.add_argument('--model_path', type=str, default='../models/srcnn_final.npy',
                       help='path to pretrained model')
    parser.add_argument('--scale_factors', type=int, nargs='+', default=[2, 3, 4],
                       help='scale factors to evaluate')
    parser.add_argument('--save_images', action='store_true',
                       help='save output images')

    args = parser.parse_args()
    
    evaluate_model(args)

if __name__ == '__main__':
    main() 
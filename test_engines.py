from engine.forensics import ForensicEngine
from engine.fusion import FusionEngine
import sys

def test_analysis(image_path):
    print(f"Analyzing: {image_path}")
    
    # ELA
    ela_img, ela_score = ForensicEngine.run_ela(image_path)
    print(f"ELA Score: {ela_score:.2f}%")
    
    # Noise
    noise_img, noise_score = ForensicEngine.run_noise_analysis(image_path)
    print(f"Noise Score: {noise_score:.2f}%")
    
    # Frequency
    freq_img, freq_score = ForensicEngine.run_frequency_analysis(image_path)
    print(f"Frequency Score: {freq_score:.2f}%")
    
    # Fusion
    results = FusionEngine.calculate_results(ela_score, noise_score, freq_score)
    print("\nFinal Results:")
    print(f"AI-generated: {results['ai_generated']:.1f}%")
    print(f"Photoshop edited: {results['photoshop_edited']:.1f}%")
    print(f"Deepfake: {results['deepfake']:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_analysis(sys.argv[1])
    else:
        print("Please provide an image path.")

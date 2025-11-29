import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import uuid

def generate_visualization(recommendations):
    """Generate visualization for recommendations"""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract data
    algorithms = [r['algorithm'] for r in recommendations]
    confidences = [r['confidence'] for r in recommendations]
    
    # Plot 1: Confidence bar chart
    colors = ['#48bb78' if c > 0.7 else '#ed8936' if c > 0.5 else '#e53e3e' for c in confidences]
    bars = ax1.barh(algorithms, confidences, color=colors, alpha=0.8)
    ax1.set_xlabel('Confidence Score')
    ax1.set_title('Algorithm Recommendations', fontweight='bold', pad=20)
    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, conf in zip(bars, confidences):
        ax1.text(conf + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{conf:.2%}', va='center', fontweight='bold')
    
    # Plot 2: Characteristics radar (if enough data)
    if len(recommendations) >= 3:
        categories = ['Speed', 'Interpretability', 'Accuracy', 'Robustness']
        values = [0.8, 0.7, confidences[0], 0.75]  # Sample values
        
        angles = [n / float(len(categories)) * 2 * 3.14159 for n in range(len(categories))]
        values += values[:1]
        angles += angles[:1]
        
        ax2 = plt.subplot(122, projection='polar')
        ax2.plot(angles, values, 'o-', linewidth=2, color='#4299e1')
        ax2.fill(angles, values, alpha=0.25, color='#4299e1')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_title('Top Algorithm Characteristics', fontweight='bold', pad=20)
    else:
        ax2.text(0.5, 0.5, 'Insufficient data\nfor characteristics', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.axis('off')
    
    plt.tight_layout()
    
    # Save
    filename = f"viz_{uuid.uuid4().hex[:8]}.png"
    output_path = Path("temp") / filename
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path

import torch
import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Core imports
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Visualization imports
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
import matplotlib.patches as mpatches

# Analysis imports
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Topological analysis imports
import dionysus as d
from zigzag_DL import ZIGZAG


@dataclass
class ModelConfig:
    """Configuration for model loading and analysis."""
    model_name: str = "mistralai/Mistral-7B-v0.1"
    device: str = "auto"
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = tempfile.TemporaryDirectory().name


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    selected_layers: List[int] = None
    k_neighbors: int = 2
    barcode_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.selected_layers is None:
            self.selected_layers = [0, 6, 13, 20, 27, 32]
        
        if self.barcode_params is None:
            self.barcode_params = {"knn": 2, "dim": 3}


class ModelLoader:
    """Handles model and tokenizer loading."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._setup_model()
    
    def _setup_model(self):
        """Initialize model and tokenizer."""
        torch.set_grad_enabled(False)
        print("Disabled automatic differentiation")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name, 
            device_map=self.config.device,
            cache_dir=self.config.cache_dir
        )
    
    def _verify_token_positions(self, input_ids: torch.Tensor, prompts: List[str]):
        """Verify that the last two tokens are month and 'is'."""
        month_names = {
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        }
        
        verification_passed = True
        
        for i, prompt in enumerate(prompts):
            # Get the last two tokens
            last_token_id = input_ids[i, -1].item()
            second_last_token_id = input_ids[i, -2].item()
            
            # Decode tokens
            last_token = self.tokenizer.decode([last_token_id]).strip()
            second_last_token = self.tokenizer.decode([second_last_token_id]).strip()
            
            # Check if last token is "is"
            if last_token != "is":
                print(f"Warning: Expected 'is' as last token in prompt {i}, got '{last_token}'")
                print(f"Prompt: {prompt}")
                verification_passed = False
            
            # Check if second-to-last token is a month name
            if second_last_token not in month_names:
                print(f"Warning: Expected month name as second-to-last token in prompt {i}, got '{second_last_token}'")
                print(f"Prompt: {prompt}")
                verification_passed = False
        
        if verification_passed:
            print(f"✓ Token verification successful: All {len(prompts)} prompts have correct token positions (month + 'is')")
    
    def _verify_predictions(self, logits: torch.Tensor, prompts: List[str]):
        """Verify that the model's top predictions are month names."""
        month_names = {
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        }
        
        # Get top predictions for the last position (next token after "is")
        last_position_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        top_token_ids = torch.argmax(last_position_logits, dim=-1)  # [batch_size]
        
        predictions_correct = True
        
        print("\nModel predictions:")
        for i, prompt in enumerate(prompts):
            top_token_id = top_token_ids[i].item()
            predicted_token = self.tokenizer.decode([top_token_id]).strip()
            
            # Extract the starting month from the prompt
            start_month = prompt.split("from ")[1].split(" is")[0]
            
            print(f"Prompt {i:2d}: '{start_month}' + 4 months → Predicted: '{predicted_token}'")
            
            if predicted_token not in month_names:
                print(f"  ⚠️  Warning: Predicted token '{predicted_token}' is not a month name")
                predictions_correct = False
        
        if predictions_correct:
            print(f"\n✓ Prediction verification successful: All {len(prompts)} predictions are month names")
    
    def generate_with_hidden_states(self, prompts: List[str]) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Generate text and extract hidden states."""
        with torch.no_grad():
            input_ids = self.tokenizer(prompts, return_tensors='pt').to(self.device)
            outputs = self.model(
                input_ids=input_ids['input_ids'],
                output_hidden_states=True
            )
            
            # Verify token positions
            self._verify_token_positions(input_ids['input_ids'], prompts)
            
            # Verify model predictions
            self._verify_predictions(outputs.logits, prompts)
            
            # Extract hidden states for last and second-to-last tokens
            last_tokens = torch.stack([
                outputs.hidden_states[idx][:, -1, :] 
                for idx in range(33)
            ], dim=0)
            
            month_tokens = torch.stack([
                outputs.hidden_states[idx][:, -2, :] 
                for idx in range(33)
            ], dim=0)
            
            return last_tokens, month_tokens


class PromptGenerator:
    """Generates prompts for calendar math tasks."""
    
    MONTHS = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    
    @classmethod
    def generate_calendar_prompts(cls, offset_range: Tuple[int, int] = (2, 3)) -> List[str]:
        """Generate calendar math prompts."""
        prompts = []
        start_offset, end_offset = offset_range
        
        for offset in range(start_offset, end_offset):
            for i in range(12):
                from_month = cls.MONTHS[i]
                prompts.append(f"Let's do some calendar math. Four months from {from_month} is")
        
        return prompts


class TopologicalAnalyzer:
    """Handles topological data analysis using persistent homology."""
    
    @staticmethod
    def convert_diagrams_to_numpy(diagrams):
        """Convert dionysus diagrams to numpy arrays."""
        return [
            np.array([[interval.birth, interval.death] for interval in diag]) 
            for diag in diagrams
        ]
    
    @staticmethod
    def compute_zigzag_barcodes(representations: torch.Tensor, 
                               params: Optional[Dict] = None, 
                               show_plots: bool = False, 
                               save_plots: bool = False, 
                               output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Compute zigzag persistent homology barcodes."""
        if params is None:
            params = {"knn": 2, "dim": 3}
        
        token_array = representations.cpu().numpy()
        zigclass = ZIGZAG(params, reps=token_array)
        
        # Generate simplicial complex and compute persistence
        simplices, simplices_padded = zigclass.generate_simplex_tree()
        layers = zigclass.compute_layers_with_intersection(simplices_padded)
        filtration, times = zigclass.compute_filtration_times(simplices, layers)
        zz, diagrams, cells = zigclass.compute_zigzag_persistence(filtration, times)
        
        # Convert to different formats
        diagrams_numpy = TopologicalAnalyzer.convert_diagrams_to_numpy(diagrams)
        converted_diagrams = [np.array(dgm) // 2 for dgm in diagrams_numpy]
        dionysus_diagrams = [d.Diagram(dgm) for dgm in converted_diagrams]
        
        return {
            'diagrams': dionysus_diagrams,
            'raw_diagrams': diagrams_numpy,
            'converted_diagrams': converted_diagrams,
            'simplices': simplices,
            'layers': layers,
            'filtration': filtration,
            'times': times
        }


class Visualizer:
    """Handles visualization of analysis results."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def plot_combined_knn_and_barcodes(self, 
                                     prompts: List[str], 
                                     hidden_states: torch.Tensor,
                                     fig_title: str = "",
                                     show_legend: bool = True) -> plt.Figure:
        """Create combined k-NN and barcode visualization."""
        n_layers, batch_size, hidden_dim = hidden_states.shape
        n_cols = len(self.config.selected_layers)
        
        fig = plt.figure(figsize=(20, 8))
        colors = plt.cm.plasma(np.linspace(0, 1, batch_size))
        cmap = ListedColormap(colors)
        
        # Top row: k-NN plots
        self._plot_knn_row(fig, hidden_states, colors, cmap, n_cols)
        
        # Bottom row: Barcode plot
        self._plot_barcode_row(fig, hidden_states, n_layers, n_cols)
        
        # Add legend and colorbar
        if show_legend:
            self._add_legend(fig, prompts, colors, batch_size)
        
        self._add_colorbar(fig)
        
        return fig
    
    def _plot_knn_row(self, fig, hidden_states, colors, cmap, n_cols):
        """Plot k-NN connectivity graphs for selected layers."""
        for i, layer_idx in enumerate(self.config.selected_layers):
            ax_knn = plt.subplot(2, n_cols, i + 1)
            layer_hidden_states = hidden_states[layer_idx].cpu().numpy()
            
            # Dimensionality reduction
            reducer = PCA(n_components=2)
            embeddings = reducer.fit_transform(layer_hidden_states)
            
            # k-NN computation
            knn = NearestNeighbors(n_neighbors=self.config.k_neighbors, metric="euclidean")
            knn.fit(layer_hidden_states)
            knn_indices = knn.kneighbors(return_distance=False)
            
            # Create edges for visualization
            edges = []
            batch_size = hidden_states.shape[1]
            for j in range(batch_size):
                for neighbor in knn_indices[j]:
                    if j != neighbor:
                        edges.append((embeddings[j], embeddings[neighbor]))
            
            # Plot points and edges
            ax_knn.scatter(embeddings[:, 0], embeddings[:, 1], 
                          c=np.arange(batch_size), cmap=cmap, s=100, edgecolors='black')
            
            edge_collection = LineCollection(edges, color="gray", alpha=0.5, linewidths=1)
            ax_knn.add_collection(edge_collection)
            
            ax_knn.set_title(f"Layer {layer_idx}", fontsize="x-large")
            ax_knn.set_xticks([])
            ax_knn.set_yticks([])
    
    def _plot_barcode_row(self, fig, hidden_states, n_layers, n_cols):
        """Plot persistent homology barcodes."""
        left_pos = 2.2/n_cols * 0.5
        right_pos = 1 - (1.7/n_cols * 0.5)
        width = right_pos - left_pos
        ax_barcode = fig.add_axes([left_pos, 0.1, width, 0.35])
        
        # Compute barcodes
        results = TopologicalAnalyzer.compute_zigzag_barcodes(
            hidden_states, 
            params=self.config.barcode_params,
            show_plots=False, 
            save_plots=False
        )
        
        dionysus_diagrams = results['diagrams']
        
        # Plot barcodes
        if len(dionysus_diagrams) > 1 and len(dionysus_diagrams[1]) > 0:
            d.plot.plot_bars(dionysus_diagrams[1], ax=ax_barcode)
        elif len(dionysus_diagrams) > 0 and len(dionysus_diagrams[0]) > 0:
            d.plot.plot_bars(dionysus_diagrams[0], ax=ax_barcode)
        else:
            ax_barcode.text(0.5, 0.5, 'No features', ha='center', va='center', 
                          transform=ax_barcode.transAxes)
        
        # Styling
        ax_barcode.grid(True, alpha=0.3)
        ax_barcode.set_xlabel("Layers", fontsize="large")
        ax_barcode.set_ylabel("Persistent Features", fontsize="large")
        ax_barcode.tick_params(axis='both', which='major', labelsize='large')
        
        # Add vertical lines for selected layers
        for layer_idx in self.config.selected_layers:
            ax_barcode.axvline(x=layer_idx, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        ax_barcode.set_xlim(-0.25, n_layers + 0.2)
    
    def _add_legend(self, fig, prompts, colors, batch_size):
        """Add legend to the figure."""
        expected_times = [
            f"{int(prompt.split(' ')[4]) + 1} {prompt.split(' ')[5][:-1]}" 
            for prompt in prompts
        ]
        legend_patches = [
            mpatches.Patch(color=colors[i], label=f"{expected_times[i]}") 
            for i in range(batch_size)
        ]
        fig.legend(handles=legend_patches, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  ncol=12, title="Prediction", fontsize="medium", title_fontsize="medium")
    
    def _add_colorbar(self, fig):
        """Add colorbar to the figure."""
        norm = Normalize(vmin=0, vmax=11)
        sm = ScalarMappable(norm=norm, cmap=plt.cm.plasma)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.92, 0.55, 0.015, 0.3])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
        cbar.set_ticks([0, 4, 8, 11])
        cbar.set_ticklabels(['Jan', 'May', 'Sept', 'Dec'])
        cbar.ax.tick_params(labelsize='large')


class ExperimentRunner:
    """Main class to run the analysis experiment."""
    
    def __init__(self, model_config: ModelConfig, analysis_config: AnalysisConfig):
        self.model_config = model_config
        self.analysis_config = analysis_config
        self.model_loader = ModelLoader(model_config)
        self.visualizer = Visualizer(analysis_config)
    
    def run_calendar_analysis(self) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[str]]:
        """Run the complete calendar math analysis."""
        # Generate prompts
        prompts = PromptGenerator.generate_calendar_prompts()
        print(f"Generated {len(prompts)} prompts")
        
        # Generate responses and extract hidden states
        last_tokens, month_tokens = self.model_loader.generate_with_hidden_states(prompts)
        
        print(f"Extracted hidden states: {last_tokens.shape}")
        return prompts, last_tokens, month_tokens
    
    def visualize_results(self, prompts: List[str], hidden_states: torch.Tensor, 
                         title: str = "Analysis Results") -> plt.Figure:
        """Create visualization of the analysis results."""
        return self.visualizer.plot_combined_knn_and_barcodes(
            prompts, hidden_states, fig_title=title, show_legend=False
        )
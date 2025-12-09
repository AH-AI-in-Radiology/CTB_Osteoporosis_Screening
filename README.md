# Automated screening of osteoporosis from non-contrast CT brain imaging

**Paper**: pending publication

This repository contains pre-trained models and a workflow for osteoporosis and low BMD (Bone Mineral Density) screening using non-contrast CT brain imaging. The methodology used a Convolutional Neural Network (CNN), specifically seresnet18, to classify CTB imaging. 

Classification occurs in a 2 step process:
1. Identification of the slice above the lateral ventricles using this [model](https://jasonccai.github.io/HeadCTSegmentation/).
2. Performing classification with 12 slices centered on the slice identified in '1' together with age and sex

There are two models which each address a different screening scenario:
1. Osteoporosis BMD screening (T <= -2.5 is labelled 1, and T > -2.5 is labelled 0)
2. Low BMD screening (T < -1.0 is labelled 1, and T >= -1.0 is labelled 0)

## Repository Structure

- `models.py`: Contains the model architecture
- `utils.py`: Contains utility functions for data preprocessing and model loading
- `demo.ipynb`: Jupyter notebook for running the model on sample data
- `models/`: Directory containing the model configuration and weights (download instructions below)
  - `config.yaml`: Configuration file for the model
  - `model.pt`: Pre-trained model weights
- `data/`: Directory where the data should be places
  - `scans/`: Directory where all the CT Scans should be placed
  - `masks/`: Directory where all the segemented masks should be placed

## Quick Start

### Pre-requisites

- Python 3.10 or higher (tested with Python 3.10)
- The key libraries are:
  - `torch`
  - `monai`
  - `numpy`
  - `timm`

### Download Configuration and Pre-trained Model Weights

1. Navigate to the **Releases** section of this GitHub repository.
2. Select the desired release version.
3. Download the desired model artifacts and rename it to `model.pt`.
4. Place `model.pt` in the `models/` directory of this repository.
5. Place your dicom files in the `data/scans/` directory.
6. Generate the brain segmentation using the instruction provided [here](https://jasonccai.github.io/HeadCTSegmentation/).
7. Ensure the directory structure looks like this:
   ```
   osteoporosis_ct/
   ├── models/
   │   ├── config.yaml
   │   └── model.pt
   ├── data/
   │   ├── scans/
   │   └── masks/
   ├── demo.ipynb
   ├── models.py
   └── utils.py
   ```

### Running the Demo

1. Open the `demo.ipynb` notebook in Jupyter or any compatible notebook environment.
2. Run the cells sequentially to load the model and test it on sample CT images.

## Contributing

Feel free to contribute to this repository by creating pull requests or opening issues to suggest improvements or report bugs.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any questions or suggestions, please contact [[Stefan.KACHEL@austin.org.au](mailto:Stefan.KACHEL@austin.org.au)].

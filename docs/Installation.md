# Installation

This package is intended for installation within an existing CaImAn conda environment. Please refer to the [CaImAn installation instructions](https://caiman.readthedocs.io/en/latest/Installation.html) to set up this environment. 

For Windows users new to CaImAn, see the PDF guide [Note on CaImAn setup on Windows](./Note%20on%20CaImAn%20setup%20on%20Windows.pdf) for step-by-step installation instructions.

Do not install Pluvianus outside of the CaImAn environment!

## From distribution
* To install Pluvianus from PyPI, run the following command in your already set up CaImAn environment:
   ```bash
   pip install pluvianus
   ```

## From Source
Steps:
1. Clone the repository:  
   ```bash
   git clone https://github.com/katonage/pluvianus.git
   ```
2. Navigate to the pluvianus directory:
   ```bash  
   cd pluvianus
   ```
3. Install Pluvianus in editable mode into your CaImAn environment:
   ```bash
   pip install -e .
   ```

## System Requirements
* Refer to CaImAn's system requirements: [CaImAn Installation](https://caiman.readthedocs.io/en/latest/Installation.html)
* At least an HD display is recommended.

## Tested on
* PC, Windows, Anaconda3. 
* CaImAn 1.11.4 - 1.13.1  
* PySide6 6.9.2 - 6.10.2   
(With PySide 6.9.1 the application freezes.)
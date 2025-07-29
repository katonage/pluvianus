import argparse
from pluvianus.pluvianus_app import run_gui

parser = argparse.ArgumentParser(description="Pluvianus GUI")
parser.add_argument("-f", "--file", help="CaImAn results file")
parser.add_argument("-d", "--data", help="Movement corrected data file")
args = parser.parse_args()

if __name__ == "__main__":
    run_gui(file_path=args.file, data_path=args.data)
    
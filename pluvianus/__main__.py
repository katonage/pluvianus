#!/usr/bin/env python3

import argparse

from pluvianus.pluvianus_app import run_gui

def main():
    parser = argparse.ArgumentParser(description="Pluvianus GUI")
    parser.add_argument("-f", "--file", help="CaImAn results file")
    parser.add_argument("-d", "--data", help="Movement corrected data file")
    args = parser.parse_args()

    run_gui(file_path=args.file, data_path=args.data)

if __name__ == "__main__":
    main()
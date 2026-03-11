import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start-month', type=str, required=True)
parser.add_argument('--end-month', type=str, required=True)
args = parser.parse_args()


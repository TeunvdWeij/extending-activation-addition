import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-name', type=str, default="llama2")
parser.add_argument('-i', type=float, default="injection coefficient")
parser.add_argument('-', type=str, default="injection coefficient")

parser.add_argument('-', type=str, default="injection coefficient")


args = parser.parse_args()

print(args)
print("all gucci")
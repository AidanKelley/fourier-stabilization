from coding import save_codes

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("n", action="store")
parser.add_argument("-o", dest="out_file", action="store")

args = parser.parse_args()

n = int(args.n)
out_file = args.out_file

codes = [[i] for i in range(n)]

save_codes(codes, out_file)

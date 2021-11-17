import torch
import csv
import os
import argparse
parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--output-file', dest='output_file',
                    help='The directory used to save the trained models',
                    default='output.tsv', type=str)
args = parser.parse_args()
def average(input):
    return sum(input)/len(input)
dict_data = torch.load(os.path.join(args.save_dir, "excel_data","dict"))
fields = dict_data.keys()
dict_data["avg test acc"] = average(dict_data["avg test acc"])
dict_data["data transferred"] = average(dict_data["data transferred"])
print(dict_data)

if not( os.path.isfile(args.output_file) ):
    with open(args.output_file, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames= fields, delimiter='\t' )
        writer.writeheader()


with open(args.output_file, 'a') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames= fields, delimiter='\t' )
    writer.writerow(dict_data)

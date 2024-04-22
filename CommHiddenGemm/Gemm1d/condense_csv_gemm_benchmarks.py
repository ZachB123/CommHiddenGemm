import csv
import os
import sys
import argparse

# input_file = "benchmarks/nocheck-n8-benchmark.csv"
# output_file = "test.csv"


def reduce_folder(input_folder, output_folder, num_inputs):
    input_folder = "gemm-benchmarks"
    output_folder = "gemm-benchmarks-condensed"
    num_inputs = 5

    os.makedirs(output_folder, exist_ok=True)

    files = os.listdir(input_folder)
    print(files)

    for filename in files:

        data = {}

        # format is: algorithm, processors, m, k, n, throughput, time, memory

        # this assumes the same trials are next to each other in the csv
        with open(f"{input_folder}/{filename}", "r", newline="") as file:
            reader = csv.reader(file)
            for row in reader:
                key = tuple(row[:num_inputs])  # First 5 are like the inputs
                if key not in data:
                    data[key] = {
                        "count": 1,
                        "data": [[float(x) for x in row[num_inputs:]]],
                    }  # Convert remaining columns to floats
                else:
                    data[key]["count"] = data[key]["count"] + 1
                    data[key]["data"].append(
                        [
                            float(row[i + num_inputs])
                            for i in range(len(row[num_inputs:]))
                        ]
                    )

        with open(f"{output_folder}/{filename}", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for key, values in data.items():
                curr_data = values["data"][0]
                for i in range(1, len(values["data"])):
                    for j in range(len(values["data"][i])):
                        curr_data[j] += values["data"][i][j]
                for i in range(len(curr_data)):
                    curr_data[i] = curr_data[i] / values["count"]
                writer.writerow(list(key) + curr_data)


def main():
    parser = argparse.ArgumentParser(description="Reduce CSV arguments")
    parser.add_argument("-i", "--input-folder", type=str, help="Input folder name")
    parser.add_argument("-o", "--output-folder", type=str, help="Output folder name")
    parser.add_argument("-n", "--num-inputs", type=int, help="Number of inputs")

    args = parser.parse_args()

    if not args.input_folder:
        parser.error("Input folder not specified!")
    if not args.output_folder:
        parser.error("Output folder not specified!")
    if args.num_inputs is None:
        parser.error("Number of inputs not specified!")

    print("Input folder:", args.input_folder)
    print("Output folder:", args.output_folder)
    print("Number of inputs:", args.num_inputs)


if __name__ == "__main__":
    print("yo")
    main()

import csv
import os
import sys

# input_file = "basic-benchmarks/python-n8-broadcastbenchmark.csv"
# output_file = "out.csv"

input_folder = "basic-benchmarks"
output_folder = "basic-benchmarks-condensed"

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
            key = tuple(row[:2])  # First 2 are the inputs
            if key not in data:
                data[key] = {
                    "count": 1,
                    "data": [[float(x) for x in row[2:]]],
                }  # Convert remaining columns to floats
            else:
                data[key]["count"] = data[key]["count"] + 1
                data[key]["data"].append(
                    [float(row[i + 2]) for i in range(len(row[2:]))]
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

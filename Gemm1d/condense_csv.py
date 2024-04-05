import csv

input_file = "benchmarks/nocheck-n8-benchmark.csv"
output_file = "test.csv"

data = {}

# format is: algorithm, processors, m, k, n, throughput, time, memory

# this assumes the same trials are next to each other in the csv
with open(input_file, "r", newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        key = tuple(row[:5])  # First 5 are like the inputs
        if key not in data:
            data[key] = {"count": 1, "data": [[float(x) for x in row[5:]]]}  # Convert remaining columns to floats
        else:
            data[key]["count"] = data[key]["count"] + 1
            data[key]["data"].append([float(row[i+5]) for i in range(len(row[5:]))])

with open(output_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for key, values in data.items():
        curr_data = values["data"][0]
        for i in range(1, len(values["data"])):
            for j in range(len(values["data"][i])):
                curr_data[j] += values["data"][i][j]
        for i in range(len(curr_data)):
            curr_data[i] = curr_data[i] / values["count"]
        writer.writerow(list(key) + curr_data)


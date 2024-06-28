import argparse

KEY_MAX_LENGTH = 20

def time_to_nanoseconds(time: str):

    time = time.strip()
    if time.startswith("max"):
        time = time[3:]
        idx = time.find("/")
        time = time[:idx-1]
        time = time.strip()
    else:
        idx = time.find("(")
        time = time[:idx-1]

    split = time.split(" ")
    value = float(split[0])
    unit = split[1].strip()
    if unit == "s":
        return value * 1e9
    elif unit == "ms":
        return value * 1e6
    elif unit == "us":
        return value * 1e3
    elif unit == "ns":
        return value
    else:
        raise Exception("Unknown time unit: " + unit)
    
def time_to_microseconds(time):
    return time_to_nanoseconds(time) / 1e3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bench_output_file", help="The file containing the output of the benchmark")
    parser.add_argument("--other", help=
        "Another file containing the output of the benchmark (optional)."
        "If not supplied, will compare the HOST and DEVICE sections of the main file."
        "If supplied, will compare the content between the main file and the other file.",
        default=""
    )

    args = parser.parse_args()
    file = args.bench_output_file
    other_file = args.other

    host_lines = []
    device_lines = {}

    if other_file == "":
        device_flag = False
        with open(file, "r") as f:
            for line in f:
                if "DEVICE" in line:
                    device_flag = True
                    continue
                if not (":" in line):
                    continue
                split = line.split(":")
                key = split[0].strip()
                value = time_to_microseconds(split[1])
                if device_flag:
                    device_lines[key] = value
                else:
                    host_lines.append((key, value))
    else:
        with open(file, "r") as f:
            for line in f:
                if not (":" in line):
                    continue
                split = line.split(":")
                key = split[0].strip()
                value = time_to_microseconds(split[1])
                # if already in host_lines, update
                found = False
                for i, (k, v) in enumerate(host_lines):
                    if k == key:
                        host_lines[i] = (k, value)
                        found = True
                        break
                if not found:
                    host_lines.append((key, value))
        with open(other_file, "r") as f:
            for line in f:
                if not (":" in line):
                    continue
                split = line.split(":")
                key = split[0].strip()
                value = time_to_microseconds(split[1])
                device_lines[key] = value

    for (key, host_value) in host_lines:
        # find key in device lines
        device_value = device_lines.get(key, None)
        if device_value is None:
            print("Key " + key + " not found in device lines")
            continue
        # print key padded with spaces, right aligned
        print(key.ljust(KEY_MAX_LENGTH), end=" ")
        print(": {:6.2f}x".format(host_value / device_value), end=" ")
        print("({:8.2f} vs {:8.2f} us)".format(host_value, device_value))
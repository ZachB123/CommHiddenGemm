import random
import sys

def generate_random_numbers(count):
    # Generate 'count' random numbers between -10 and 9
    return [random.randint(-10, 9) for _ in range(count)]

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <count>")
        sys.exit(1)

    try:
        count = int(sys.argv[1])
    except ValueError:
        print("Count must be an integer.")
        sys.exit(1)

    if count <= 0:
        print("Count must be a positive integer.")
        sys.exit(1)

    random_numbers = generate_random_numbers(count)
    print("Generated random numbers:", random_numbers)
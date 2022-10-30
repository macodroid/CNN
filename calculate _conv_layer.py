def calculate(n_in, kernel, padding, stride):
    return ((n_in - kernel + 2 * padding) / stride) + 1.0


n_in = float(input("N_in = "))
kernel = float(input("kernel = "))
padding = float(input("padding = "))
stride = float(input("stride = "))

print(f'output is: {calculate(n_in, kernel, padding, stride)}')

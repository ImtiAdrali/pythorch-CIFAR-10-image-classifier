numbers = [10, 20, 30, 40, 50]

number_iterator = iter(numbers)

try:
    print(next(number_iterator))
    print(next(number_iterator))
    print(next(number_iterator))
    print(next(number_iterator))
    print(next(number_iterator))
    print(next(number_iterator))
except StopIteration:
    print("Reached the end of the iterator.")